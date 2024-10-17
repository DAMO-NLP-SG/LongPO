
import sys

import copy
from dataclasses import dataclass, field, asdict
import pathlib

import torch
import transformers
from transformers import Trainer, is_torch_tpu_available, AutoModelForCausalLM, set_seed, AutoConfig
from transformers.trainer_pt_utils import LabelSmoother
from argument import *
import numpy as np
from accelerate import Accelerator
import logging
import datasets
import wandb
logger = logging.getLogger(__name__)
from trl import SFTTrainer
from ring_trainer import RingSFTTrainer
from datasets import load_dataset, load_from_disk

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import os
# os.environ["WANDB_MODE"] = "disabled"
from utils import apply_chat_template, CustomTrainer, get_dataset, CustomSFTTrainer, concat_long_alpaca, concat_long_self_instruct, get_long_dpo_dataset
from longdpo_trainer import (
    LongDPOTrainer,
    LongDPORingTrainer,
    LongDPOUlyssesTrainer,
    LongDPOJointUlyssesTrainer,
    LongDPOFullJointUlyssesTrainer,
    LongDPOFullMTJointUlyssesTrainer,
    LongDPOFullMTFixedJointUlyssesTrainer,
    LongSFTJointUlyssesTrainer,
    LongSFTKLJointUlyssesTrainer,

)

# from ring_monkey_patch import replace_attn_with_ring_attn

# replace_attn_with_ring_attn()

# from ulysses.monkey_patch_llama3 import replace_attn_with_ring_attn
from ulysses.monkey_patch_mistral import replace_attn_with_sequence_parallel_attn

replace_attn_with_sequence_parallel_attn()

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






def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    set_seed(training_args.seed)

    # accelerator = Accelerator()

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # from CLEX import LlamaForCausalLM, CLEXLlamaConfig


    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path
    )
    config.rope_theta = model_args.rope_theta
    # config.sliding_window = training_args.model_max_length
    
    # from transformers import MistralForCausalLM
    # device_map = {"": os.environ.get('LOCAL_RANK', '0')}
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
        trust_remote_code=True,
        use_flash_attention_2=True,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False


    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.ref_model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        use_flash_attention_2=True,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    ref_model.config.use_cache = False

    # ref_model = model
    # model.model.clex_layer.proj_func.reset_parameters()
    # for name, param in model.named_parameters():
    #     if "clex" in name:
    #         param.requires_grad = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if "mistral" in model_args.model_name_or_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
    elif "llama" in model_args.model_name_or_path.lower():
        tokenizer.pad_token = "<|end_of_text|>"
    # DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    # tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    # raw_datasets = get_dataset(data_args.data_path, splits=["train_sft", "test_sft"])
    train_dataset = load_from_disk(data_args.data_path).shuffle(seed=42)
    # keys = train_dataset.column_names
    # def convert_format(example):
    #     for key in keys:
    #         if len(example[key]) == 1:
    #             example[key] = example[key][0]
    #     return example

    # train_dataset = train_dataset.map(convert_format, num_proc=60)
    eval_dataset = None
    # raw_datasets['train'] = concat_long_self_instruct(raw_datasets['train'])
    # #####################
    # # Apply chat template
    # #####################
    column_names = list(train_dataset.features)
    # train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "dpo"}, num_proc=40)
    # train_dataset = train_dataset.map(
    #     apply_chat_template,
    #     fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
    #     num_proc=40,
    #     remove_columns=column_names,
    #     desc="Formatting comparisons with prompt template",
    # )
    # train_dataset = train_dataset.rename_columns(
    #     {"text_prompt": "prompt", "text_short_prompt": "short_prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    # )
    # model.eval()

    # print("Perplexity:", perplexity)

    logger.info("*** Model loaded! ***")

    # ref_model = model
    
    # TRAINER_CLASS = LongDPOFullMTFixedJointUlyssesTrainer
    TRAINER_CLASS = LongDPOFullMTJointUlyssesTrainer
    ########################
    # Initialize the Trainer
    ########################
    trainer = TRAINER_CLASS(
        model,
        ref_model,
        beta=model_args.dpo_beta,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        max_prompt_length=training_args.model_max_length,
        dataset_num_proc=96
    )


    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    accelerator = Accelerator()
    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tags": ["sft"],
        }
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

        if training_args.push_to_hub is True:
            logger.info("Pushing to hub...")
            trainer.push_to_hub()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
