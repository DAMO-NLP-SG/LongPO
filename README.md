# LongPO: Long Context Self-Evolution of Large Language Models through Short-to-Long Preference Optimization
This repo provides the official implementation of our paper "LongPO: Long Context Self-Evolution of Large Language Models through Short-to-Long Preference Optimization".

<div style='display:flex; gap: 0.25rem; '>
<!-- <a href='https://huggingface.co/DAMO-NLP-SG'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a>  -->
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-blue'></a>
</div>


## Updates
- [2024.2.13]  🚀 Release the code, data and checkpoints trained with LongPO. 
- [2025.1.23] 🌟 LongPO has been accepted to ICLR 2025!



## Highlights of LongPO
- Self-evolving long-context alignment without human/superior LLMs annotations.
- Extending context length while aligning to be instruction-following in one stage.
- No degradation on short-context capabilities.



## Training Process:

1. Prompt a short-context instruct LLM (e.g., Mistral-7B-Instruct-v0.2) to self-generate short-to-long preference data as illustrated in [data_prepare](data_prepare/readme.md).

2. Replace the (Flash) Attention module into Ulyssess (Flash) Attn using monkey patch to apply sequence parallel.
  
3. Using our custom LongPO Trainer: `LongPOMTLMUlyssesTrainer`

4. Train Script (using Mistral-7B-Instruct-v0.2 as example):

```

export training_length=131072
export gradient_accumulation_steps=8
export batch_size=1

accelerate launch \
--config_file playground/accelerate_single_node_zero3.yaml \
train/train_longpo.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --ref_model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --data_path /path/to/data \
    --bf16 True \
    --run_name mistral_longpo \
    --report_to wandb \
    --output_dir path/to/save \
    --num_train_epochs 1 \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --save_strategy "steps" \
    --save_steps 500 \
    --evaluation_strategy "no" \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --optim "rmsprop" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $training_length \
    --gradient_checkpointing True \
    --do_train True \
    --do_eval False \
    --do_predict False \
    --seed 42 \
    --use_ring_attention True \
    --dpo_beta 0.01 \
    --dpo_lambda 0.01 \
    --rope_theta 10000000
```


## Requirments

```
transformers >= 4.44.0
flash-attn
trl == 0.8.6
```
