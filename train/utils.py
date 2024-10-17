import random
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from torch.utils.data import IterableDataset
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import warnings
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
import os
import numpy as np

from transformers import Trainer
import os
import re
from typing import List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk, Dataset
from datasets.builder import DatasetGenerationError
from trl import SFTTrainer, DPOTrainer
import json


def extract_ordered_user_assistant_pairs(dataset, chunk_size=24):
    # Initialize the result list
    res_data = []
    # Create a list of dialogue indices and shuffle it for randomness
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Convert the list of indices into chunks
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

    for chunk_id, chunk in enumerate(chunks):
        # Create a working copy of the chunk to manipulate
        merged_pairs = []
        system_msg = """You are given questions that belong to different conversations, which are highlighted by 'Conversation+ID'. Please answer each question according to its respective conversation context."""
        merged_pairs.append({'role':'system', 'content':system_msg})
        # merged_answers = []
        working_chunk = dataset[chunk]['messages']
        while working_chunk:
            # Randomly select a dialogue from the working chunk
            dialogue_index = random.randrange(len(working_chunk))
            dialogue = working_chunk[dialogue_index]
            # Find and extract the first user-assistant pair in the selected dialogue
            for i in range(0, len(dialogue), 2):  # Step by 2 to get pairs
                if i+1 < len(dialogue) and dialogue[i]['role'] == 'user' and dialogue[i+1]['role'] == 'assistant':
                    dialogue[i]['content'] = f"Convesation {(dialogue_index+1)* (chunk_id+1)}: " + dialogue[i]['content']
                    pair = [dialogue[i], dialogue[i+1]]
                    merged_pairs.extend(pair)

                    # Remove the extracted pair from the dialogue copy
                    del dialogue[i:i+2]
                    break  # Stop after extracting the first pair

            # If the dialogue has no more pairs, remove it from the working chunk
            if not dialogue:
                working_chunk.pop(dialogue_index)
        res_data.append({"messages": merged_pairs})
    return Dataset.from_list(res_data)
    


def merge_dpo_data(dataset, chunk_size=48):
    # Initialize the result list
    res_dpo_data = []
    # Create a list of dialogue indices and shuffle it for randomness
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Convert the list of indices into chunks
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

    for chunk_id, chunk in enumerate(chunks):
        # Create a working copy of the chunk to manipulate
        # system_msg = """You are given questions that belong to different conversations, which are highlighted by 'Conversation+ID'. Please answer each question according to its respective conversation context."""
        # merged_pairs.append({'role':'system', 'content':system_msg})
        # merged_answers = []
        # res_dpo_data = []
        prompt = dataset[chunk]['prompt']
        chosen_chunk = dataset[chunk]['chosen']
        reject_chunk = dataset[chunk]['rejected']
        score_chosen = dataset[chunk]['score_chosen']
        score_rejected = dataset[chunk]['score_rejected']
        score_gap = [chosen - rejected for chosen, rejected in zip(score_chosen, score_rejected)]
        closest_indices = sorted(range(len(score_gap)), key=lambda idx: score_gap[idx], reverse=True)


        merged_prompt = ""
        merged_chose = []
        merged_reject = []
        for i, item in enumerate(prompt):
            merged_prompt += f"Query {i+1}:\n" + item + "\n\n"

        for i, item in enumerate(chosen_chunk):
            merged_chose.append(f"Response {i+1}:\n" + item[-1]['content'] + "\n\n")

        for i, item in enumerate(reject_chunk):
            merged_reject.append(f"Response {i+1}:\n" + item[-1]['content'] + "\n\n")
        
        # merged_pairs.append(
        #     {
        #         'prompt': merged_prompt,
        #         'chosen': [{'content':merged_prompt, 'role':'user'}, {'content': merged_chose, 'role':'assistant'}],
        #         'rejected': [{'content':merged_prompt, 'role':'user'}, {'content': merged_reject, 'role':'assistant'}]
        #     }
        # )
        # print(len(closest_indices), len(chosen_chunk))
        merged_pairs = []
        pair_num = 8
        chunks_indices = [closest_indices[i:i + len(closest_indices)//pair_num] for i in range(0, len(closest_indices), len(closest_indices)//pair_num)]
        # print(len(chunks_indices))
        gather_index_list = []
        merged_pairs.append("".join(merged_chose))
        for index_item in chunks_indices:
            gather_index_list.extend(index_item)
            combined_responses = []
            for i in range(len(chosen_chunk)):
                if i in gather_index_list:
                    combined_responses.append(merged_reject[i])
                else:
                    combined_responses.append(merged_chose[i])
            
            merged_pairs.append("".join(combined_responses))
        print(len(merged_pairs))
        item_dpo_data = []
        for i, item in enumerate(merged_pairs):
            for j in range(i+1, len(merged_pairs)):
                item_dpo_data.append(
                    {
                        'prompt': merged_prompt,
                        'chosen': [{'content':merged_prompt, 'role':'user'}, {'content': item, 'role':'assistant'}],
                        'rejected': [{'content':merged_prompt, 'role':'user'}, {'content': merged_pairs[j], 'role':'assistant'}]
                    }
                )
        res_dpo_data.extend(item_dpo_data[:32])
    return concatenate_datasets([Dataset.from_list(res_dpo_data[:len(res_dpo_data)//2]), Dataset.from_list(res_dpo_data[len(res_dpo_data)//2:])])

def load_json_data(path):
    data = []
    with open(path, "r") as f:
        for item in f:
            data.append(json.loads(item))
    return data

def save_jsonl(data, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def get_long_alpaca(path="/mnt/workspace/data/long_context/LongAlpaca-12k/LongAlpaca-12k.json"):
    with open(path, 'r') as f:
        alpaca_data = json.load(f)
    long_data = alpaca_data[:7320]
    res_list = []
    for item in long_data:
        messages = [{'role':'user', 'content':item['instruction']}, {'role':'assistant', 'content':item['output']}]
        res_list.append({"messages":messages})
    return Dataset.from_list(res_list)

def concat_long_alpaca(train_datasets):
    alpaca_data = get_long_alpaca()
    return concatenate_datasets([train_datasets, alpaca_data])


def concat_long_self_instruct(train_datasets):
    # alpaca_data = get_long_alpaca()
    self_instruct_data = load_json_data("/mnt/workspace/Projects/LongAlign/data_generation/mistral_new/LongInstruct-100k.json")
    # new_data = []
    # for item in self_instruct_data:
    #     messages = item["messages"]
    #     if not messages:
    #         continue
    #     # We add an empty system message if there is none
    #     messages[0]["content"] = item['prompt'] + "\n\n" + messages[0]["content"]
    #     item["messages"] = messages
    #     # We add an empty system message if there is none
    #     # item["messages"][0]["content"] = example['prompt'] + "\n\n" + messages[0]["content"]
    #     new_data.append(item)
    self_instruct_data = Dataset.from_list(self_instruct_data).shuffle(seed=42)
    return concatenate_datasets([train_datasets, self_instruct_data])



def load_and_concat_json_files(folder_path):
    concatenated_list = []  # This will hold all the JSON objects
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')][:1]  # List JSON files in the folder
    
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)  # Get the full path of the JSON file
        data = load_json_data(file_path)
        # with open(file_path, 'r') as file:
        #     data = json.load(file)  # Load the JSON file; assumes the file contains a list of JSON objects
        concatenated_list.extend(data)  # Concatenate the current file's list to the accumulated list
    
    return concatenated_list

def get_long_dpo_dataset(path):
    # data_list = load_and_concat_json_files(path)
    data_list = load_json_data(path)
    return Dataset.from_list(data_list)


def get_dataset(path, splits):
    raw_datasets = DatasetDict()
    for split in splits:
        try:
            # Try first if dataset on a Hub repo
            dataset = load_dataset(path, split=split)
            # dataset = dataset.select(list(range(32)))
        except:
            # If not, check local dataset
            dataset = load_from_disk(os.path.join(path, split))
        if "train" in split:
            raw_datasets["train"] = dataset
        if "test" in split:
            raw_datasets["test"] = dataset
    return raw_datasets

def apply_chat_template(
    example, tokenizer, task: Literal["sft", "generation", "rm", "dpo"] = "sft", assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        # if messages[0]["role"] != "system":
        #     messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if chosen_messages[0]["role"] != "system":
                chosen_messages.insert(0, {"role": "system", "content": ""})
            if rejected_messages[0]["role"] != "system":
                rejected_messages.insert(0, {"role": "system", "content": ""})
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        # print(example.keys())
        # print("--------------------")
        # print("using dpo")
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            # prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            prompt_messages = [ {'role':"user", 'content': example['prompt'] + "\n\n" + example['question']}]
            short_prompt_messages = [ {'role':"user", 'content': example['short_prompt'] + "\n\n" + example['question']}]
            # Insert system message
            # if example["chosen"][0]["role"] != "system":
            #     prompt_messages.insert(0, {"role": "system", "content": ""})
            # else:
            # prompt_messages.insert(0, example["chosen"][0])
            # TODO: handle case where chosen/rejected also have system messages
            # chosen_messages = [{'role':"assistant", 'content': example["chosen"]}]
            # rejected_messages = [ {'role':"assistant", 'content': example["rejected"]}]
            # example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            # example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            example["text_chosen"] = example["chosen"]
            example["text_rejected"] = example["rejected"]
            example["text_prompt"] = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            example["text_short_prompt"] = tokenizer.apply_chat_template(
                short_prompt_messages, tokenize=False, add_generation_prompt=True
            )
            # example["text_chosen"] = _strip_prefix(example["text_chosen"], assistant_prefix)
            # example["text_rejected"] = _strip_prefix(example["text_rejected"], assistant_prefix)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example



class CustomTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        PREFIX_CHECKPOINT_DIR = "checkpoint"
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            logger.warning(
                f"Checkpoint destination directory {output_dir} already exists and is non-empty."
                "Saving will proceed but saved results may be invalid."
            )
            staging_output_dir = output_dir
        else:
            staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
        self.save_model(staging_output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(staging_output_dir)
            # Save RNG state
            self._save_rng_state(staging_output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir
        TRAINER_STATE_NAME = "trainer_state.json"
        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(staging_output_dir)

        # Place checkpoint in final location after all saving is finished.
        # First wait for everyone to finish writing
        self.args.distributed_state.wait_for_everyone()
        # Then go through the rewriting process starting on process 0
        if staging_output_dir != output_dir:
            with self.args.main_process_first(
                desc="Renaming model checkpoint folder to true location", local=self.args.save_on_each_node
            ):
                if self.args.should_save and os.path.exists(staging_output_dir):
                    os.rename(staging_output_dir, output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

def pad_to_length(tensor: torch.Tensor, length: int, pad_value, dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )

class CustomSFTTrainer(SFTTrainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        PREFIX_CHECKPOINT_DIR = "checkpoint"
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            logger.warning(
                f"Checkpoint destination directory {output_dir} already exists and is non-empty."
                "Saving will proceed but saved results may be invalid."
            )
            staging_output_dir = output_dir
        else:
            staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
        self.save_model(staging_output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(staging_output_dir)
            # Save RNG state
            self._save_rng_state(staging_output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir
        TRAINER_STATE_NAME = "trainer_state.json"
        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(staging_output_dir)

        # Place checkpoint in final location after all saving is finished.
        # First wait for everyone to finish writing
        self.args.distributed_state.wait_for_everyone()
        # Then go through the rewriting process starting on process 0
        if staging_output_dir != output_dir:
            with self.args.main_process_first(
                desc="Renaming model checkpoint folder to true location", local=self.args.save_on_each_node
            ):
                if self.args.should_save and os.path.exists(staging_output_dir):
                    os.rename(staging_output_dir, output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


from torch.utils.data import Sampler

class LongDPOSampler(Sampler):
    def __init__(self, dataset, pair_num=32):
        super().__init__(dataset)
        self.dataset = dataset
        self.pair_num = pair_num

        # Ensure the dataset size is a multiple of the domain size
        assert len(self.dataset) % self.pair_num == 0, \
            "Dataset size must be a multiple of the domain size."

        # Calculate the number of domains
        self.num_pairs = len(self.dataset) // self.pair_num

    def __iter__(self):
        # Generate indices for each domain
        pair_indices = [list(range(i * self.pair_num, (i + 1) * self.pair_num)) for i in range(self.num_pairs)]
        
        # Optionally shuffle the domain indices here if you want different domain order each epoch
        random.shuffle(pair_indices)

        # Flatten the list of lists
        indices = [index for sublist in pair_indices for index in sublist]

        return iter(indices)

    def __len__(self):
        return len(self.dataset)



class CustomDPOTrainer(DPOTrainer):
    def _get_train_sampler(self):
        return LongDPOSampler(self.train_dataset)
    
    def concatenated_forward(
        self, model, batch
    ):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    @staticmethod
    def concatenated_inputs(
        batch,
        is_encoder_decoder,
        label_pad_token_id,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}
        max_length = 28000
        # print(f"padding to {max_length}")
        # if is_encoder_decoder:
        #     max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        # else:
        #     max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        # from transformers.utils import pad_to_length
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch


    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        PREFIX_CHECKPOINT_DIR = "checkpoint"
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            logger.warning(
                f"Checkpoint destination directory {output_dir} already exists and is non-empty."
                "Saving will proceed but saved results may be invalid."
            )
            staging_output_dir = output_dir
        else:
            staging_output_dir = os.path.join(run_dir, f"tmp-{checkpoint_folder}")
        self.save_model(staging_output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(staging_output_dir)
            # Save RNG state
            self._save_rng_state(staging_output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir
        TRAINER_STATE_NAME = "trainer_state.json"
        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(staging_output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(staging_output_dir)

        # Place checkpoint in final location after all saving is finished.
        # First wait for everyone to finish writing
        self.args.distributed_state.wait_for_everyone()
        # Then go through the rewriting process starting on process 0
        if staging_output_dir != output_dir:
            with self.args.main_process_first(
                desc="Renaming model checkpoint folder to true location", local=self.args.save_on_each_node
            ):
                if self.args.should_save and os.path.exists(staging_output_dir):
                    os.rename(staging_output_dir, output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


class SaveDeepSpeedPeftModelCallback(TrainerCallback):
    def __init__(self, trainer, save_steps=500):
        self.trainer = trainer
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if (state.global_step + 1) % self.save_steps == 0:
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(self.trainer.deepspeed)
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(args.output_dir, state_dict=state_dict)
            self.trainer.accelerator.wait_for_everyone()
        return control


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            shuffle (bool): If true, the samples in each buffer are suffled. Default is `True`.
            add_eos_token (bool): If true, each buffer is delimited with eos token. Default is `True`.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=64,
        chars_per_token=3.6,
        content_field="content",
        shuffle=True,
        add_eos_token=True,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.shuffle = shuffle
        self.add_eos_token = add_eos_token
        # print(f"Max Buffer: {self.max_buffer_size}")

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.add_eos_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=40):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens
import os
def load_data(data_path, prefix="sampled"):
    json_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.startswith(prefix) and f.endswith('.jsonl')]
    print(json_files)
    from datasets import concatenate_datasets, Features, Value, Dataset
    import json
    features = Features({
        'text': Value('string'),
    })
    dataset_list = []
    all_data = []
    for file in json_files:
        with open(file, 'r') as f:
            json_data = []
            for item in f:
                json_data.append(json.loads(item))
        for item in json_data:
            if 'text' in item.keys():
                all_data.append({'text':item['text']})
    import random
    random.seed(42)
    random.shuffle(all_data)
    dataset = Dataset.from_list(all_data)

    return dataset


# def load_data(data_path, prefix="sampled"):
#     json_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.startswith(prefix) and f.endswith('.jsonl')]
#     dataset = load_dataset('json', data_files=json_files, split='train', streaming=True)
#     return dataset

def create_datasets(tokenizer, training_args, data_path):
    # dataset = load_from_disk(data_path)
    # from IPython import embed; embed()
    train_dataset, valid_dataset = None, None
    if training_args.do_train:
        train_data = load_data(data_path, prefix="sampled")
        # print(f"Size of the train set: {len(train_data)}")
        column_names = train_data.column_names
        dataset_text_field = "text" if "text" in column_names else column_names[0]
        # chars_per_token = chars_token_ratio(train_data, tokenizer, dataset_text_field)
        # print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
        train_dataset = ConstantLengthDataset(
            tokenizer,
            train_data,
            infinite=True,
            seq_length=training_args.model_max_length,
            chars_per_token=3.4,
            num_of_sequences=256,
            content_field=dataset_text_field,
            shuffle=True,
            add_eos_token=False,
        )
    if training_args.do_eval or training_args.do_predict:
        valid_data = load_data(data_path, "test")
        # valid_data = test_dataset
        column_names = valid_data.column_names
        dataset_text_field = "text" if "text" in column_names else column_names[0]
        # chars_per_token = chars_token_ratio(valid_data, tokenizer, dataset_text_field)
        print(f"Size of the validation set: {len(valid_data)}")
        valid_dataset = ConstantLengthDataset(
            tokenizer,
            valid_data,
            infinite=False,
            seq_length=training_args.model_max_length,
            chars_per_token=3.4,
            num_of_sequences=2,
            content_field=dataset_text_field,
            shuffle=False,
            add_eos_token=False,
        )

    return train_dataset, valid_dataset


def create_and_prepare_model(args):
    device_map = None
    bnb_config = None
    load_in_8bit = args.use_8bit_qunatization

    if args.use_4bit_qunatization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_qunatization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16 and args.use_4bit_qunatization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        device_map = "auto"  # {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=load_in_8bit,
        quantization_config=bnb_config,
        device_map=device_map,
        use_cache=not args.use_gradient_checkpointing,
        trust_remote_code=True,
        use_flash_attention_2=args.use_flash_attn
    )

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(","),
        )
        if (args.use_4bit_qunatization or args.use_8bit_qunatization) and args.use_peft_lora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_gradient_checkpointing)

        if args.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


def peft_module_casting_to_bf16(model, args):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)