# LongPO: Long Context Self-Evolution of Large Language Models through Short-to-Long Preference Optimization

## Training Process:

1. Process the data, building the label for answer tokens and padding others.
  
2. Replace the Attention Module into Ulyssess Attn using monkey patch.
  
3. Replace the Trainer class into our custom Ulysses Trainer.
  
  - LongPO Trainer: `LongDPOFullMTJointUlyssesTrainer`
    
  - SFT Trainer using Ulysses: `LongSFTKLJointUlyssesTrainer`: Note that this Trainer uses our LongPO data format with a custom KL divergence. To access the naive SFT loss, refer to the chosen lm loss here.

4. Train Script:

```

export training_length=131072
export gradient_accumulation_steps=8
export batch_size=1

accelerate launch \
--config_file /mnt/workspace/gzchen/playground/accelerate_single_node_zero3.yaml \
train/train_longpo.py \
    --model_name_or_path /path/to/model \
    --ref_model_name_or_path /path/to/model \
    --data_path /path/to/data \
    --bf16 True \
    --run_name xxxx \
    --report_to wandb \
    --output_dir xxxx \
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