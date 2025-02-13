
import sys
# sys.path.append("/mnt/workspace/Projects/LongAlign/train")
# from utils import apply_chat_template

from transformers import AutoTokenizer
import warnings
from datasets import Dataset
import numpy as np
import re

import random
import json
import os

count = 0

def format_labels(examples):
    tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/ckpts/Qwen/Qwen2-7B-Instruct")
    # tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    ignore_index = -100
    instruction_template = "<|im_start|>user"
    response_template = '<|im_start|>assistant'
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    instruction_token_ids = tokenizer.encode(instruction_template, add_special_tokens=False)

    batch = tokenizer(examples, padding=True, truncation=False, return_tensors="pt")
    labels = batch["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    for i in range(len(examples)):
        response_token_ids_idxs = []
        human_token_ids_idxs = []

        for assistant_idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
            # find the indexes of the start of a response.
            if (
                response_token_ids
                == batch["labels"][i][assistant_idx : assistant_idx + len(response_token_ids)].tolist()
            ):
                response_token_ids_idxs.append(assistant_idx + len(response_token_ids))

        if len(response_token_ids_idxs) == 0:
            warnings.warn(
                f"Could not find response key `{response_template}` in the "
                f'following instance: {tokenizer.decode(batch["input_ids"][i][-10:])} '
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            batch["labels"][i, :] = ignore_index

        human_token_ids = instruction_token_ids
        for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
            # find the indexes of the start of a human answer.
            if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                human_token_ids_idxs.append(human_idx)

        if len(human_token_ids_idxs) == 0:
            warnings.warn(
                f"Could not find instruction key `{instruction_template}` in the "
                f'following instance: {tokenizer.decode(batch["input_ids"][i][:100])} '
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            batch["labels"][i, :] = ignore_index

        if (
            len(human_token_ids_idxs) > 0
            and len(response_token_ids_idxs) > 0
            and human_token_ids_idxs[0] > response_token_ids_idxs[0]
        ):
            human_token_ids_idxs = [0] + human_token_ids_idxs

        for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
            # Make pytorch loss function ignore all non response tokens
            if idx != 0:
                batch["labels"][i, start:end] = ignore_index
            else:
                batch["labels"][i, :end] = ignore_index

        if len(response_token_ids_idxs) < len(human_token_ids_idxs):
            batch["labels"][i, human_token_ids_idxs[-1] :] = ignore_index
    return batch


def build_chat_format(prompt, question, answer):
    user_prompt = prompt + "\n\n" + question if prompt else question
    return [
        {'role':"user", 'content': user_prompt},
        {'role':"assistant", 'content': answer}
    ]


def apply_chat_template(
    example, tokenizer, task="dpo", assistant_prefix="<|assistant|>\n"
):
    batch = {}
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if task == "dpo":
        # print(example.keys())
        # print("--------------------")
        # print("using dpo")
        if all(k in example.keys() for k in ("chosens", "rejecteds")):
            # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
            # prompt_messages = [[msg for msg in example["chosen"] if msg["role"] == "user"][0]]
            chosen_prompt_messages = []
            rejected_prompt_messages = []
            chosen_short_prompt_messages = []
            rejected_short_prompt_messages = []
            for i in range(len(example['questions'])):
                prompt = example['prompt'] if i == 0 else ""
                question = example['questions'][i]
                chosen_answer = example["chosens"][i]
                rejected_answer = example["rejecteds"][i]
                chosen_prompt_messages.extend(build_chat_format(prompt, question, chosen_answer))
                rejected_prompt_messages.extend(build_chat_format(prompt, question, rejected_answer))

                short_prompt = example['short_prompts'][i]
                # Note it is append here
                chosen_short_prompt_messages.append(build_chat_format(short_prompt, question, chosen_answer))
                rejected_short_prompt_messages.append(build_chat_format(short_prompt, question, rejected_answer))
            
            chosen_prompt_input = tokenizer.apply_chat_template(
                chosen_prompt_messages, tokenize=False, add_generation_prompt=True
            )
            rejected_prompt_input = tokenizer.apply_chat_template(
                rejected_prompt_messages, tokenize=False, add_generation_prompt=True
            )
            
            chosen_short_prompt_input = [tokenizer.apply_chat_template(
                item, tokenize=False, add_generation_prompt=True
            ) for item in chosen_short_prompt_messages]
            rejected_short_prompt_input = [tokenizer.apply_chat_template(
                item, tokenize=False, add_generation_prompt=True
            ) for item in rejected_short_prompt_messages]
            long_batch_chosen = format_labels([chosen_prompt_input])
            long_batch_rejected = format_labels([rejected_prompt_input])
            short_batch_chosen = format_labels(chosen_short_prompt_input)
            short_batch_rejected = format_labels(rejected_short_prompt_input)

            # short_batch = format_labels(chosen_short_prompt_input + rejected_short_prompt_input)
            for index in range(2):
                tmp_batch_chosen = long_batch_chosen if index == 0 else short_batch_chosen
                tmp_batch_rejected = long_batch_rejected if index == 0 else short_batch_rejected
                # len_chosen = 1 if index == 0 else len(chosen_short_prompt_messages)
                for k, toks in {
                    "chosen_": tmp_batch_chosen,
                    "rejected_": tmp_batch_rejected,
                }.items():
                    for type_key, tokens in toks.items():
                        tokens = tokens.tolist()
                        if type_key == "token_type_ids":
                            continue
                        if index == 1:
                            batch[f"ref_{k}{type_key}"] = tokens
                        else:
                            batch[f"{k}{type_key}"] = tokens
            return batch

            
            
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )


def load_json_data(path):
    data = []
    with open(path, "r") as f:
        for item in f:
            data.append(json.loads(item))
    return data



def check_repetitions_kmp(string, repetition_threshold=3):
    """
    Check if a string contains many repetitions of substrings using KMP algorithm concepts.

    Parameters:
    string (str): The input string to check.
    repetition_threshold (int): The minimum number of repetitions to be considered as many.

    Returns:
    bool: True if many repetitions are found, False otherwise.
    """
    
    # Length of the string
    length = len(string)
    if length <= 1:
        return False
    
    # Compute the KMP table (prefix function)
    pi = [0] * length
    k = 0
    for i in range(1, length):
        while k > 0 and string[k] != string[i]:
            k = pi[k - 1]
        if string[k] == string[i]:
            k += 1
        pi[i] = k
    
    # Check for the longest prefix which is also a suffix
    longest_prefix_suffix_length = pi[-1]
    
    if longest_prefix_suffix_length == 0:
        return False
    
    # Check if it repeats enough times
    repeating_unit_length = length - longest_prefix_suffix_length
    if length % repeating_unit_length != 0:
        return False
    
    repetitions = length // repeating_unit_length
    return repetitions >= repetition_threshold

def build_tokenized_answer(prompt, answer):
    """
    Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
    It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    """

    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

    # Prepare input tokens for token by token comparison
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )
def tokenize_row(feature):
    """Tokenize a single row from a DPO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + chosen or prompt + rejected responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
        the sum of the length of the prompt and the chosen/rejected response, with
        label_pad_token_id  for the prompt tokens.
    """
    global count
    # count += 1
    # print(count)
    batch = {}
    # try:
    # prompt = feature["prompt"] if not ref_mode else feature["short_prompt"]
    chosen = feature["chosen"]
    rejected = feature["rejected"]

    for index, prompt in enumerate([feature["prompt"], feature["short_prompt"]]):
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = build_tokenized_answer(prompt, rejected)

            # add BOS token to head of prompt
#             prompt_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
#             chosen_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
#             rejected_tokens["prompt_input_ids"] = [tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

#             prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
#             chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
#             rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer
            chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > 256000:
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-256000 :]

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > 256000:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: 1024]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                -100
            ] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                -100
            ] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    if index == 1:
                        batch[f"ref_{k}{type_key}"] = tokens
                    else:
                        batch[f"{k}{type_key}"] = tokens
    # except:
    #     return {}
    return batch



from tqdm import tqdm
def filter_censored_responses(data, censored_phrases):
    """
    Filters out responses that contain any of the censored phrases.

    Parameters:
    responses (list of str): List of responses to filter.
    censored_phrases (list of str): List of phrases indicating a response is censored.

    Returns:
    list of str: List of non-censored responses.
    """
    responses = [item['chosen'] for item in data]
    rejected_res = [item['rejected'] for item in data]
    print(len(responses), len(rejected_res))
    # responses = responses
    filtered_responses = []
    results_items = []
    for i, response in tqdm(enumerate(rejected_res)):
        if not any(phrase.lower() in response.lower() for phrase in censored_phrases):
            if not any(phrase2.lower() in responses[i].lower() for phrase2 in censored_phrases):
            # filtered_responses.append(response)
                if len(response) > 10 and len(responses[i]) > 10:
                    if not check_repetitions_kmp(response) and not check_repetitions_kmp(responses[i]):
                        results_items.append(data[i])
    # for i, response in enumerate(responses):
    #     if not any(phrase.lower() in response.lower() for phrase in censored_phrases):
    #         # filtered_responses.append(response)
    #         if len(response) > 0:
    #             results_items.append(data[i])
    return results_items



import random
from collections import defaultdict
def sample_by_prompt_id(data_list, samples_per_group=2):
    # Convert dataset to list of dictionaries
    # data_list = dataset.to_list()  # Convert to Python dict for easy manipulation
    
    # Group by prompt_id
    groups = defaultdict(list)
    for item in data_list:
        # item = {key: data_list[key][idx] for key in data_list}
        groups[item['prompt_id']].append(item)
    
    # Sample 2 items from each group
    sampled_data = []
    for prompt_id, items in groups.items():
        # if len(items) < samples_per_group:
        #     continue
        if len(items) > samples_per_group:
            sampled_data.extend(random.sample(items, samples_per_group))
        else:
            sampled_data.extend(items)  # Include all if less than samples_per_group
    
    # Convert back to the dataset format if necessary (Assuming all columns are of the same length)
    # sampled_dataset = []
    # for key in dataset.features.keys():
    #     sampled_dataset[key] = [item[key] for item in sampled_data]

    return sampled_data


def group_by_prompt_id(data_list):
    # Convert dataset to list of dictionaries
    # data_list = dataset.to_list()  # Convert to Python dict for easy manipulation
    
    # Group by prompt_id
    groups = defaultdict(list)
    for item in data_list:
        # item = {key: data_list[key][idx] for key in data_list}
        groups[item['prompt_id']].append(item)
        assert item['prompt'] == groups[item['prompt_id']][0]['prompt']
    
#     # Sample 2 items from each group
#     sampled_data = []
#     for prompt_id, items in groups.items():
#         # if len(items) < samples_per_group:
#         #     continue
#         if len(items) > samples_per_group:
#             sampled_data.extend(random.sample(items, samples_per_group))
#         else:
#             sampled_data.extend(items)  # Include all if less than samples_per_group
    
    # Convert back to the dataset format if necessary (Assuming all columns are of the same length)
    # sampled_dataset = []
    # for key in dataset.features.keys():
    #     sampled_dataset[key] = [item[key] for item in sampled_data]
    group_data_list = []
    count = 0
    for prompt_id, items in groups.items():
        tmp_item = {}
        if len(items) != 4:
            count += 1
            continue
        tmp_item['prompt'] = items[0]['prompt']
        tmp_item['short_prompts'] = [item['short_prompt'] for item in items]
        tmp_item['questions'] = [item['question'] for item in items]
        tmp_item['chosens'] = [item['chosen'] for item in items]
        tmp_item['rejecteds'] = [item['rejected'] for item in items]
        group_data_list.append(tmp_item)
    print(count)
    return group_data_list

# Example usage

censored_phrases = [
    "can't assist",
    "content is not available",
    "As an AI assistant",
    "I am unable to provide",
    "Sorry",
    "That topic is outside my scope",
    "My guidelines prevent me from addressing",
    "I can't help",
    "I am not sure how to get that information",
    "I'm unable to",
    "I can't fulfill this request",
    "I don't have the capability to provide",
    "This topic is restricted",
    "I cannot discuss this subject",
    "This information is not accessible",
    "I'm not permitted to provide",
    "I have no data on this issue",
    "My training data does not cover this",
    "Access to this information is restricted",
    "This query is beyond my capabilities",
    "Sorry, I can't delve into that",
    "This goes beyond my operational boundaries",
    "I cannot support this inquiry",
    "I'm programmed not to respond to this",
    "This matter is beyond my scope",
    "My responses are limited on this topic",
    "I have no input on that",
    "My algorithm restricts me from answering",
    "I'm sorry, but that's beyond my reach",
    "My response capabilities don’t include this topic",
    "This falls outside my answer scope",
    "I don't cover that area",
    "I can't engage with this subject matter",
    "I'm not the right resource for that information",
    "This content is beyond my purview",
    "I don't handle this type of query",
    "My training limits me from assisting on this",
    "This issue is beyond my functionality",
    "I am restricted from providing insights on this",
]

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Generate instruction')
    # parser.add_argument('--domain', type=str, default='Book', help='Domain of the text', required=True)
    parser.add_argument("--split", type=int, default=0)
    # parser.add_argument('--model_name', type=str, default='mistral', help='Model Name', required=True)
    parser.add_argument('--model_path', type=str, default='EleutherAI/gpt-neo-2.7B', help='Model Path', required=True)
    parser.add_argument('--data_path', type=str, default='data', help='Input file', required=True)
    parser.add_argument('--save_path', type=str, default='save_data', help='Save Folder', required=True)
    return parser.parse_args()





import sys


if __name__ == "__main__":
    # split = 0
    # split = int(sys.argv[1])
    args = parse_args()
    split = args.split
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)


    data = load_json_data(f"{args.data_path}/split_{split}.json")


    data = filter_censored_responses(data, censored_phrases)

    data = group_by_prompt_id(data)

    train_dataset = Dataset.from_list(data)
    column_names = list(train_dataset.features)
    train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=20,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    train_dataset = train_dataset.filter(lambda item: len(item.keys())>0 and len(item["ref_chosen_input_ids"]) < 32*1024 and len(item["ref_rejected_input_ids"]) < 32*1024, num_proc=20)
    train_dataset.save_to_disk(f"{args.save_path}/split_{split}")
    print(len(train_dataset))