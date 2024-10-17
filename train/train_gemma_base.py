
import sys
# sys.path.append("/mnt/workspace/Projects/CLEX")

import copy
from dataclasses import dataclass, field, asdict
import pathlib

import torch
import transformers
from transformers import Trainer, is_torch_tpu_available, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from argument import *
import numpy as np

import logging
import wandb
logger = logging.getLogger(__name__)
from utils import create_datasets, CustomTrainer



IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import os
# os.environ["WANDB_MODE"] = "disabled"



local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()



def calculate_perplexity(model, tokenizer, dataset, batch_size, config):
    model.eval()

    total_ppl = 0
    total_tokens = 0
    total_loss = 0.0
    # total_loss
    # from accelerate import Accelerator
    # accelerator = Accelerator()
    # device = accelerator.device
    # model = model.to(device)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    from tqdm import tqdm
    for batch in tqdm(dataloader):
        past = None
        input_ids = batch["input_ids"]
        chunks = [input_ids[i:i + 1] for i in range(0, len(input_ids), 1)]
        batch_loss = 0.0
        batch_tokens = 0
        for chunk in chunks:
            # input_ids = tokenizer.encode(chunk, add_special_tokens=True, truncation=True, padding="max_length", max_length=max_length)
            # attention_mask = [1] * len(input_ids)

            input_ids = torch.tensor(chunk).unsqueeze(0)

            input_ids = input_ids.to(device="cuda" if torch.cuda.is_available() else "cpu")
          

            with torch.no_grad():
                outputs = model(input_ids, past_key_values=past, use_cache=True, return_dict=True)

            
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            # Flatten the tokens
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), input_ids.view(-1))
            # print(loss.tolist())
            # batch_ppl = torch.exp(loss)
            chunk_tokens = input_ids.ne(tokenizer.pad_token_id).sum().item()

            batch_loss += loss.item() * chunk_tokens
            batch_tokens += chunk_tokens
            # Store the past key and value for the next iteration
            past = outputs.past_key_values
        total_loss += batch_loss
        total_tokens += batch_tokens
        print(batch_loss/batch_tokens)
    average_ppl = np.exp(total_loss / total_tokens)
    return average_ppl


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank


    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        # _fast_init=False
    )
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        # padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )
    # tokenizer.pad_token = tokenizer.unk_token

    # from dataset import WikiDataset
    # dataset = WikiDataset(tokenizer=tokenizer, model_args=model_args, data_args=data_args, training_args=training_args)

    # model.eval()
  
    # perplexity = calculate_perplexity(model, tokenizer, dataset.predict_dataset, 1, config)
    # print("Perplexity:", perplexity)
    
    train_dataset, valid_dataset = create_datasets(tokenizer, training_args, data_args.data_path)
    # model.eval()
  
    # perplexity = calculate_perplexity(model, tokenizer, dataset.predict_dataset, 1, config)
    # print("Perplexity:", perplexity)
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    trainer = CustomTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        # compute_metrics=dataset.compute_metrics if training_args.do_predict and not is_torch_tpu_available() else None,
        # data_collator=dataset.data_collator,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
        # if training_args.do_predict and not is_torch_tpu_available()
        # else None,
    )
    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        # if trainer.is_fsdp_enabled:
        #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        
        # model.config.use_cache = True
        trainer.save_state()
        # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        if trainer.is_deepspeed_enabled:
            trainer.save_model()
        else:
            trainer_save_model_safe(trainer)
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_dataset = dataset.predict_dataset
        predictions= trainer.predict(predict_dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", predictions.metrics)
        trainer.save_metrics("predict", predictions.metrics)


if __name__ == "__main__":
    train()
