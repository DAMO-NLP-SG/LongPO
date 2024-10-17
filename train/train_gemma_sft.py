
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

from datasets import load_dataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import os
os.environ["WANDB_MODE"] = "disabled"
from utils import apply_chat_template, CustomTrainer, get_dataset, CustomSFTTrainer, concat_long_alpaca, concat_long_self_instruct

# from ring_monkey_patch import replace_attn_with_ring_attn

# replace_attn_with_ring_attn()

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


def build_clex_args(config, model_args):
    config.log_scale = model_args.log_scale
    config._flash_attn_2_enabled = model_args.use_flashattn
    config.rope_scaling = {
        "type": model_args.scaling_type,
        "max_factor": model_args.max_factor,
        "param_factor": model_args.param_factor,
        "factor": 1,
        "time_dt": model_args.time_dt
    }
    
# def calculate_perplexity(model, tokenizer, dataset, batch_size):
#     model.eval()

#     total_ppl = 0
#     total_tokens = 0

#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

#     for batch in dataloader:
#         # encoded_batch = tokenizer.batch_encode_plus(
#         #     batch["text"], padding=True, truncation=True, return_tensors="pt"
#         # )
#         # print(batch["input_ids"])
#         input_ids = torch.tensor(batch["input_ids"]).unsqueeze(dim=0).to("cuda")
#         # attention_mask = batch["attention_mask"]

#         with torch.no_grad():
#             outputs = model(input_ids)

#         loss = outputs.loss
#         batch_ppl = torch.exp(loss)
#         batch_tokens = input_ids.ne(tokenizer.pad_token_id).sum().item()

#         total_ppl += batch_ppl.item() * batch_tokens
#         total_tokens += batch_tokens

#     average_ppl = total_ppl / total_tokens
#     return average_ppl


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
    # config.rope_theta = 1e6
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
    # model.model.clex_layer.proj_func.reset_parameters()
    # for name, param in model.named_parameters():
    #     if "clex" in name:
    #         param.requires_grad = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True,
    )
    if "mistral" in model_args.model_name_or_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.pad_token = tokenizer.unk_token
    # DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    # tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    raw_datasets = get_dataset(data_args.data_path, splits=["train_sft", "test_sft"])

    # raw_datasets['train'] = concat_long_self_instruct(raw_datasets['train'])
    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"}, num_proc=40)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    # model.eval()
    # perplexity = calculate_perplexity(model, tokenizer, dataset.predict_dataset, 1, config)
    # print("Perplexity:", perplexity)

    logger.info("*** Model loaded! ***")

    ########################
    # Initialize the Trainer
    ########################
    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.model_max_length,
        tokenizer=tokenizer,
        packing=True,
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
    accelerator = Accelerator()
    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

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
