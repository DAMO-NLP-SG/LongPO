
# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datasets
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import DPODataCollatorWithPadding, disable_dropout_in_model, pad_to_length, peft_module_casting_to_bf16
from trl import DPOTrainer

from transformers.utils import is_datasets_available, is_torch_tpu_available, is_sagemaker_mp_enabled, is_apex_available
import os
# logger = logging.get_logger(__name__)
from flash_attn.losses.cross_entropy import CrossEntropyLoss

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


# if is_wandb_available():
#     import wandb

if is_deepspeed_available():
    import deepspeed



class LongDPOTrainer(DPOTrainer):

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.is_encoder_decoder = False
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid",
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the DPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the DPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name
        self.reference_free = reference_free

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOTrainer's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.dataset_num_proc = dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        # with PartialState().local_main_process_first():
            # tokenize the dataset
        # train_dataset = train_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)

        Trainer.__init__(
            self,
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
    

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        # deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        # config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        # if model is not None:
        #     if hasattr(model, "config"):
        #         hidden_size = (
        #             max(model.config.hidden_sizes)
        #             if getattr(model.config, "hidden_sizes", None)
        #             else getattr(model.config, "hidden_size", None)
        #         )
        #         if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
        #             # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
        #             # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
        #             config_kwargs.update(
        #                 {
        #                     "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
        #                     "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
        #                     "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
        #                 }
        #             )

        # # If ZeRO-3 is used, we shard both the active and reference model.
        # # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        # if config_kwargs["zero_optimization"]["stage"] != 3:
        #     config_kwargs["zero_optimization"]["stage"] = 2
        # model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        # model.eval()
        return self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        # return model

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

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
            # logger.warning(
            #     f"Checkpoint destination directory {output_dir} already exists and is non-empty."
            #     "Saving will proceed but saved results may be invalid."
            # )
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
    
    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        # prompt = feature["prompt"] if not ref_mode else feature["short_prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]
        
        for index, prompt in enumerate([feature["prompt"], feature["short_prompt"]]):
            if not self.is_encoder_decoder:
                # Check issues below for more details
                #  1. https://github.com/huggingface/trl/issues/907
                #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
                #  3. https://github.com/LianjiaTech/BELLE/issues/337

                if not isinstance(prompt, str):
                    raise ValueError(f"prompt should be an str but got {type(prompt)}")
                prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
                prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

                if not isinstance(chosen, str):
                    raise ValueError(f"chosen should be an str but got {type(chosen)}")
                chosen_tokens = self.build_tokenized_answer(prompt, chosen)

                if not isinstance(rejected, str):
                    raise ValueError(f"rejected should be an str but got {type(rejected)}")
                rejected_tokens = self.build_tokenized_answer(prompt, rejected)

                # add BOS token to head of prompt
                # prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
                # chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
                # rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

                # prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
                # chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
                # rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

                # add EOS token to end of answer
                chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
                chosen_tokens["attention_mask"].append(1)

                rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
                rejected_tokens["attention_mask"].append(1)

                longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

                # if combined sequence is too long, truncate the prompt
                for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                    if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                        if self.truncation_mode == "keep_start":
                            for k in ["prompt_input_ids", "prompt_attention_mask"]:
                                answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                        elif self.truncation_mode == "keep_end":
                            for k in ["prompt_input_ids", "prompt_attention_mask"]:
                                answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                        else:
                            raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

                # if that's still too long, truncate the response
                for answer_tokens in [chosen_tokens, rejected_tokens]:
                    if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                        for k in ["input_ids", "attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

                # Create labels
                chosen_sequence_tokens = {
                    k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
                }
                rejected_sequence_tokens = {
                    k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
                }
                chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
                chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                    self.label_pad_token_id
                ] * len(chosen_tokens["prompt_input_ids"])
                rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
                rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                    self.label_pad_token_id
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

            else:
                chosen_tokens = self.tokenizer(
                    chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
                )
                rejected_tokens = self.tokenizer(
                    rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
                )
                prompt_tokens = self.tokenizer(
                    prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
                )

                batch["chosen_labels"] = chosen_tokens["input_ids"]
                batch["rejected_labels"] = rejected_tokens["input_ids"]
                batch["prompt_input_ids"] = prompt_tokens["input_ids"]
                batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

                if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                    batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                        labels=batch["rejected_labels"]
                    )
                    batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                        labels=batch["chosen_labels"]
                    )

        return batch


    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        # compute reference logps
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(
                    self.model
                ).disable_adapter() if self.is_peft_model else nullcontext():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
        ref_mode: bool = False,
    ) -> Dict[str, torch.LongTensor]:
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
        prefix = "ref_" if ref_mode else ""
        # print(batch[f"{prefix}chosen_input_ids"].shape)
        if is_encoder_decoder:
            max_length = max(batch[f"{prefix}chosen_input_ids"].shape[1], batch[f"{prefix}rejected_input_ids"].shape[1])
            # max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch[f"{prefix}chosen_input_ids"].shape[1], batch[f"{prefix}rejected_input_ids"].shape[1])
            # max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        if max_length % 16 != 0:
            max_length += 16 - (max_length % 16)
        if ref_mode:
            print("ref_length:", max_length)
        else:
            print("long length:", max_length)
        for k in batch:
            # if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            if k.startswith(f"{prefix}chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace(f"{prefix}chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            # if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            if k.startswith(f"{prefix}rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace(f"{prefix}rejected", "concatenated")
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
        

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

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

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], ref_mode=False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
            ref_mode=ref_mode,
        )
        prefix = "ref_" if ref_mode else ""
        len_chosen = batch[f"{prefix}chosen_labels"].shape[0]
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

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch, ref_mode=False)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch, ref_mode=True)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch, ref_mode=True)
        # if self.args.use_ring_attention:
        torch.distributed.all_reduce(policy_chosen_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(policy_rejected_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(reference_chosen_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(reference_rejected_logps, op=torch.distributed.ReduceOp.SUM)
        # print("gather sucessfully, begin computing loss.")
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # if reference_output in batch use that otherwise use the reference model
        if "reference_output" in batch:
            reference_output = batch["reference_output"]
        else:
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    reference_output = self.model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            else:
                reference_output = self.ref_model.generate(
                    input_ids=batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"], policy_output_decoded, ref_output_decoded
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output
    
   


class LongDPORingTrainer(LongDPOTrainer):


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            # dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return DataLoader(train_dataset, **dataloader_params)

    
    def extract_local(self, value, rank, world_size, device, dim=1):
        if value is None:
            return None
        value_chunks = value.chunk(2 * world_size, dim=dim)
        local_value = torch.cat(
            [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
        )
        return local_value.to(device)


    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], ref_mode=False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
            ref_mode=ref_mode,
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

        inputs = {
            "input_ids": concatenated_batch["concatenated_input_ids"],
            "attention_mask": concatenated_batch["concatenated_attention_mask"],
        }
        if self.args.use_ring_attention:
            position_ids = inputs.get("position_ids", None)
            if position_ids is None:
                seq_length = inputs["input_ids"].size(1)
                inputs['position_ids'] = torch.arange(seq_length).unsqueeze(0).expand(inputs["input_ids"].shape[0], -1)
                # position_ids = (
                #     torch.arange(self.args.seq_length).unsqueeze(0).expand(input_ids.shape[0], -1)
                # )
            for obj in inputs:
                if obj in ['input_ids', 'labels', 'position_ids', 'attention_mask']:
                    inputs[obj] = self.extract_local(
                        inputs[obj],
                        self.accelerator.process_index,
                        self.accelerator.num_processes, 
                        self.accelerator.device
                    )

        all_logits = model(
            **inputs,
        ).logits
        local_labels = self.extract_local(
            concatenated_batch["concatenated_labels"],
            self.accelerator.process_index,
            self.accelerator.num_processes, 
            self.accelerator.device
        )
        all_logps = self.get_batch_logps(
            all_logits,
            local_labels,
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        # print("begin backward")
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps



class LongDPOUlyssesTrainer(LongDPOTrainer):

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            # dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return DataLoader(train_dataset, **dataloader_params)

    
    
    def extract_local(self, value, rank, world_size, device, dim=1):
        if value is None:
            return None
        value_chunks = value.chunk(world_size, dim=dim)
        local_value = value_chunks[rank]
        return local_value.to(device)


    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], ref_mode=False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
            ref_mode=ref_mode,
        )
        # len_chosen = batch["chosen_labels"].shape[0]
        prefix = "ref_" if ref_mode else ""
        len_chosen = batch[f"{prefix}chosen_labels"].shape[0]
        # print("len chosen: ", len_chosen)
        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        inputs = {
            "input_ids": concatenated_batch["concatenated_input_ids"],
            "attention_mask": concatenated_batch["concatenated_attention_mask"],
        }
        if self.args.use_ring_attention:
            position_ids = inputs.get("position_ids", None)
            if position_ids is None:
                seq_length = inputs["input_ids"].size(1)
                inputs['position_ids'] = torch.arange(seq_length).unsqueeze(0).expand(inputs["input_ids"].shape[0], -1)
                # position_ids = (
                #     torch.arange(self.args.seq_length).unsqueeze(0).expand(input_ids.shape[0], -1)
                # )
            for obj in inputs:
                if obj in ['input_ids', 'labels', 'position_ids', 'attention_mask']:
                    inputs[obj] = self.extract_local(
                        inputs[obj],
                        self.accelerator.process_index,
                        self.accelerator.num_processes, 
                        self.accelerator.device
                    )
        
        all_logits = model(
            **inputs,
        ).logits
        # print(all_logits[:,-1000:,:10])
        # num_nans = torch.isnan(all_logits).sum().item
        # print(f"Num of Nans: {num_nans}, Num of paddings: {torch.eq(inputs['input_ids'], 151643).sum().item()}")
        # print(f"Rank: {self.accelerator.process_index}, Processes: {self.accelerator.num_processes}, {all_logits[0,-10:,:10]}")
        local_labels = self.extract_local(
            concatenated_batch["concatenated_labels"],
            self.accelerator.process_index,
            self.accelerator.num_processes, 
            self.accelerator.device
        )
        all_logps = self.get_batch_logps(
            all_logits,
            local_labels,
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
    



from torch.utils.data import Sampler

class LongDPOSampler(Sampler):
    def __init__(self, dataset, pair_num=8):
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
        # random.shuffle(pair_indices)

        # Flatten the list of lists
        indices = [index for sublist in pair_indices for index in sublist]

        return iter(indices)

    def __len__(self):
        return len(self.dataset)




class LongDPOJointUlyssesTrainer(LongDPOUlyssesTrainer):

    # def _get_train_sampler(self):
    #     return LongDPOSampler(self.train_dataset)

    # @staticmethod
    # def get_batch_logps_loss(
    #     logits: torch.FloatTensor,
    #     labels: torch.LongTensor,
    #     average_log_prob: bool = False,
    #     label_pad_token_id: int = -100,
    #     is_encoder_decoder: bool = False,
    # ) -> torch.FloatTensor:
    #     """Compute the log probabilities of the given labels under the given logits.

    #     Args:
    #         logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
    #         labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
    #         average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    #     Returns:
    #         A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    #     """
    #     if logits.shape[:-1] != labels.shape:
    #         raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    #     if not is_encoder_decoder:
    #         labels = labels[:, 1:].clone()
    #         logits = logits[:, :-1, :]
    #     loss_mask = labels != label_pad_token_id

    #     # dummy token; we'll ignore the losses on these tokens later
    #     labels[labels == label_pad_token_id] = 0

    #     per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    #     lm_loss = -(per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    #     if average_log_prob:
    #         return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), lm_loss
    #     else:
    #         return (per_token_logps * loss_mask).sum(-1), lm_loss

    # def concatenated_forward(
    #     self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], ref_mode=False
    # ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    #     """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    #     We do this to avoid doing two forward passes, because it's faster for FSDP.
    #     """
    #     concatenated_batch = self.concatenated_inputs(
    #         batch,
    #         is_encoder_decoder=self.is_encoder_decoder,
    #         label_pad_token_id=self.label_pad_token_id,
    #         padding_value=self.padding_value,
    #         device=self.accelerator.device,
    #         ref_mode=ref_mode,
    #     )
    #     len_chosen = batch["chosen_labels"].shape[0]

    #     model_kwargs = (
    #         {
    #             "labels": concatenated_batch["concatenated_labels"],
    #             "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
    #         }
    #         if self.is_encoder_decoder
    #         else {}
    #     )

    #     inputs = {
    #         "input_ids": concatenated_batch["concatenated_input_ids"],
    #         "attention_mask": concatenated_batch["concatenated_attention_mask"],
    #     }
    #     if self.args.use_ring_attention:
    #         position_ids = inputs.get("position_ids", None)
    #         if position_ids is None:
    #             seq_length = inputs["input_ids"].size(1)
    #             inputs['position_ids'] = torch.arange(seq_length).unsqueeze(0).expand(inputs["input_ids"].shape[0], -1)
    #             # position_ids = (
    #             #     torch.arange(self.args.seq_length).unsqueeze(0).expand(input_ids.shape[0], -1)
    #             # )
    #         for obj in inputs:
    #             if obj in ['input_ids', 'labels', 'position_ids', 'attention_mask']:
    #                 inputs[obj] = self.extract_local(
    #                     inputs[obj],
    #                     self.accelerator.process_index,
    #                     self.accelerator.num_processes, 
    #                     self.accelerator.device
    #                 )

    #     all_logits = model(
    #         **inputs,
    #     ).logits
    #     local_labels = self.extract_local(
    #         concatenated_batch["concatenated_labels"],
    #         self.accelerator.process_index,
    #         self.accelerator.num_processes, 
    #         self.accelerator.device
    #     )
    #     all_logps,  = self.get_batch_logps(
    #         all_logits,
    #         local_labels,
    #         average_log_prob=False,
    #         is_encoder_decoder=self.is_encoder_decoder,
    #         label_pad_token_id=self.label_pad_token_id,
    #     )

    #     chosen_logps = all_logps[:len_chosen]
    #     rejected_logps = all_logps[len_chosen:]

    #     chosen_logits = all_logits[:len_chosen]
    #     rejected_logits = all_logits[len_chosen:]

    #     chosen_loss = all_loss[:len_chosen]
    #     rejected_loss = all_loss[len_chosen:]

    #     return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_loss, rejected_loss)
    


    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch, ref_mode=False)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch, ref_mode=True)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch, ref_mode=True)
        # if self.args.use_ring_attention:
        torch.distributed.all_reduce(policy_chosen_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(policy_rejected_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(reference_chosen_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(reference_rejected_logps, op=torch.distributed.ReduceOp.SUM)
        # torch.distributed.all_reduce(policy_chosen_loss, op=torch.distributed.ReduceOp.SUM)


        chosen_loss_mask = batch["chosen_labels"] != self.label_pad_token_id
        if chosen_loss_mask.sum(-1) == 0:
            policy_chosen_loss = torch.zeros_like(policy_chosen_logps)
        else:
            policy_chosen_loss = - policy_chosen_logps / chosen_loss_mask.sum(-1)
        # print("chosen loss: ", policy_chosen_loss.mean())
        # print("gather sucessfully, begin computing loss.")
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        # local_labels = self.extract_local(
        #     batch["chosen_labels"],
        #     self.accelerator.process_index,
        #     self.accelerator.num_processes, 
        #     self.accelerator.device
        # )
        # print(policy_chosen_logits.size(),  local_labels.size())
        # chose_lm_loss = self.get_lm_loss(policy_chosen_logits, local_labels)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}longdpo/loss"] = losses.cpu().mean()
        metrics[f"{prefix}lm/loss"] = policy_chosen_loss.cpu().mean()



        return policy_chosen_loss.mean() + self.args.dpo_lambda * losses.mean(), metrics



class LongDPOFullJointUlyssesTrainer(LongDPOUlyssesTrainer):



    
    def get_lm_loss(self, logits, labels):
        # if labels is not None:
            # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(inplace_backward=True)
        shift_logits = shift_logits.view(-1, logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        lm_loss = loss_fct(shift_logits, shift_labels)
        return lm_loss


    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch, ref_mode=False)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch, ref_mode=True)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch, ref_mode=True)
        # if self.args.use_ring_attention:
        torch.distributed.all_reduce(policy_chosen_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(policy_rejected_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(reference_chosen_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(reference_rejected_logps, op=torch.distributed.ReduceOp.SUM)
        # torch.distributed.all_reduce(policy_chosen_loss, op=torch.distributed.ReduceOp.SUM)


        # chosen_loss_mask = batch["chosen_labels"] != self.label_pad_token_id
        # if chosen_loss_mask.sum(-1) == 0:
        #     policy_chosen_loss = torch.zeros_like(policy_chosen_logps)
        # else:
        #     policy_chosen_loss = - policy_chosen_logps / chosen_loss_mask.sum(-1)
        # print("chosen loss: ", policy_chosen_loss.mean())
        # print("gather sucessfully, begin computing loss.")
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        # print(policy_chosen_logits.size(),  local_labels.size())

        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
            ref_mode=False,
        )
        chosen_input_ids = concatenated_batch["concatenated_input_ids"][: batch["chosen_labels"].shape[0]]
        local_labels = self.extract_local(
            chosen_input_ids,
            self.accelerator.process_index,
            self.accelerator.num_processes, 
            self.accelerator.device
        )
        policy_chosen_loss = self.get_lm_loss(policy_chosen_logits, local_labels)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}longdpo/loss"] = losses.cpu().mean()
        metrics[f"{prefix}lm/loss"] = policy_chosen_loss.cpu().mean()



        return policy_chosen_loss.mean() + self.args.dpo_lambda * losses.mean(), metrics

class LongDPOFullMTJointUlyssesTrainer(LongDPOFullJointUlyssesTrainer):


    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch, ref_mode=False)
        for k in batch:
            # if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            if k.startswith(f"ref_") and isinstance(batch[k], torch.Tensor):
                if batch[k].shape[0] == 1 and batch[k].shape[1] == 4:
                    batch[k] = batch[k].squeeze()
        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch, ref_mode=True)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch, ref_mode=True)
        # if self.args.use_ring_attention:
        torch.distributed.all_reduce(policy_chosen_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(policy_rejected_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(reference_chosen_logps, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(reference_rejected_logps, op=torch.distributed.ReduceOp.SUM)
        # torch.distributed.all_reduce(policy_chosen_loss, op=torch.distributed.ReduceOp.SUM)
        # print("chosen: ", reference_chosen_logps.size())
        # print("rejected: ", reference_rejected_logps.size())
        assert reference_chosen_logps.size() == reference_rejected_logps.size()
        assert reference_chosen_logps.shape[0] % 4 == 0
        reference_chosen_logps = reference_chosen_logps.view(-1, 4).sum(dim=1)
        reference_rejected_logps = reference_rejected_logps.view(-1, 4).sum(dim=1)

        # chosen_loss_mask = batch["chosen_labels"] != self.label_pad_token_id
        # if chosen_loss_mask.sum(-1) == 0:
        #     policy_chosen_loss = torch.zeros_like(policy_chosen_logps)
        # else:
        #     policy_chosen_loss = - policy_chosen_logps / chosen_loss_mask.sum(-1)
        # print("chosen loss: ", policy_chosen_loss.mean())
        # print("gather sucessfully, begin computing loss.")
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        # print(policy_chosen_logits.size(),  local_labels.size())

        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
            ref_mode=False,
        )
        chosen_input_ids = concatenated_batch["concatenated_input_ids"][: batch["chosen_labels"].shape[0]]
        local_labels = self.extract_local(
            chosen_input_ids,
            self.accelerator.process_index,
            self.accelerator.num_processes, 
            self.accelerator.device
        )
        policy_chosen_loss = self.get_lm_loss(policy_chosen_logits, local_labels)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}longdpo/loss"] = losses.cpu().mean()
        metrics[f"{prefix}lm/loss"] = policy_chosen_loss.cpu().mean()



        return policy_chosen_loss.mean() + self.args.dpo_lambda * losses.mean(), metrics


class LongDPOJointSeqUlyssesTrainer(LongDPOJointUlyssesTrainer):

    def _get_train_sampler(self):
        return LongDPOSampler(self.train_dataset)