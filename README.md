# LongPO: Long Context Self-Evolution of Large Language Models through Short-to-Long Preference Optimization

This repo provides the official implementation of our paper "LongPO: Long Context Self-Evolution of Large Language Models through Short-to-Long Preference Optimization".

http://arxiv.org/abs/2502.13922


<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2501.13106-AD1C18.svg?logo=arXiv)](http://arxiv.org/abs/2502.13922) 
[![hf_paper](https://img.shields.io/badge/🤗-HF%20Daily-red.svg)](https://huggingface.co/papers/2502.13922)
</h5>



## Updates

- [2024.2.17]  🚀 Release the code, data and checkpoints trained with LongPO. 
- [2025.1.23] 🌟 LongPO has been accepted to ICLR 2025!





## Highlights of LongPO

- Self-evolving long-context alignment without human/superior LLMs annotations.
- Extending context length while keeping aligned in one stage.
- No degradation on short-context capabilities.


## Models and Training Data

| Models                                                       | Base Model               | Training Data                                                | # Data Samples |
| ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ | -------------- |
| [Mistral-7B-LongPO-128K](https://huggingface.co/DAMO-NLP-SG/Mistral-7B-LongPO-128K) | Mistral-7B-Instruct-v0.2 | [HF Link](https://huggingface.co/datasets/DAMO-NLP-SG/Mistral-7B-LongPO-128K-tokenized) | 45K            |
| [Qwen2.5-7B-LongPO-128K](https://huggingface.co/DAMO-NLP-SG/Qwen2.5-7B-LongPO-128K) | Qwen2.5-7B-Instruct      | [HF Link](https://huggingface.co/datasets/DAMO-NLP-SG/Qwen2.5-7B-LongPO-128K-tokenized) | 32K            |
| [Mistral-7B-LongPO-256K-EXP](https://huggingface.co/DAMO-NLP-SG/Mistral-7B-LongPO-256K-EXP)* | Mistral-7B-LongPO-128K   | [HF Link](https://huggingface.co/datasets/DAMO-NLP-SG/Mistral-7B-LongPO-256K-tokenized) | 16K            |
| [Mistral-7B-LongPO-512K-EXP](https://huggingface.co/DAMO-NLP-SG/Mistral-7B-LongPO-512K-EXP)* | Mistral-7B-LongPO-128K   | [HF Link](https://huggingface.co/datasets/DAMO-NLP-SG/Mistral-7B-LongPO-512K-tokenized) | 2.5K           |

\* indicates an experimental version (for rebuttal purposes) that may have not been fully tuned or provided with sufficient data to achieve convergence.





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
    --use_sequence_parallel True \
    --dpo_beta 0.01 \
    --dpo_lambda 0.01 \
    --rope_theta 10000000
```

## Evaluation



### InfiniteBench


| Model            | Train/Claimed Length | En.Sum | En.QA  | En.MC  | AVG.   |
| ---------------- | -------------------- | ------ | ------ | ------ | ------ |
| GPT-4-128K       | 128K                 | 14.73  | 22.44  | 67.25  | 34.81  |
| Qwen2-72B        | 128K                 | 24.32ᵇ | 7.03ᵇ  | 72.05ᵇ | 34.47ᵇ |
| LLaMA 3.1-70B    | 128K                 | 33.55ᵇ | 36.08ᵇ | 69.00ᵇ | 46.21ᵇ |
| LLaMA 3.1-8B     | 128K                 | 28.06ᵇ | 30.47ᵇ | 58.08ᵇ | 38.87ᵇ |
| GLM-4-9B         | 128K                 | 14.84ᵇ | 9.51ᵇ  | 67.25ᵇ | 30.53ᵇ |
| GLM-4-9B-1M      | 1M                   | 28.3   | 9.7    | 68.6   | 35.53  |
| LWM-7B-1M        | 1M                   | 4.33ᵇ  | 0.0ᵇ   | 3.06ᵇ  | 2.46ᵇ  |
| YaRN-Mistral-7B  | 128K                 | 9.09   | 9.55   | 27.95  | 15.53  |
| Mistral-7B       | 32K                  | 22.13  | 4.93   | 14.41  | 13.82  |
| - SFT            | 128K                 | 23.44  | 13.45  | 53.21  | 30.03  |
| - DPO            | 128K                 | 15.21  | 10.34  | 48.14  | 25.56  |
| - LongPO (iter1) | 128K                 | 27.05  | 23.51  | 67.25  | 39.27  |
| - LongPO (iter2) | 256K                 | 28.16  | 24.43  | 66.35  | 39.65  |
| - LongPO (iter3) | 512K                 | 29.10  | 27.85  | 66.67  | 41.21  |
| Qwen2.5-7B       | 128K                 | 22.89  | 6.08   | 52.4   | 27.12  |
| - LongPO (iter1) | 128K                 | 32.06  | 17.32  | 72.05  | 40.48  |

- Our results are evaluated with greedy decoding.
- Baseline results marked with ᵇ are evaluated by us, while unmarked baseline results are sourced from their official report.





### RULER

| Model                    | NIAH  | VT    | AGG   | QA    | AVG (13 tasks) |
| ------------------------ | ----- | ----- | ----- | ----- | -------------- |
| Qwen2.5-7B-Instruct      | 82.10 | 80.09 | 74.50 | 54.30 | 76.50          |
| Qwen2.5-7B-LongPO-128K   | 95.82 | 89.71 | 78.67 | 59.40 | 87.11          |
| Mistral-7B-Instruct-v0.2 | 72.60 | 74.40 | 64.40 | 52.20 | 68.40          |
| Mistral-7B-LongPO-128K   | 96.88 | 96.49 | 71.55 | 64.81 | 88.02          |
| Mistral-7B-LongPO-256K   | 96.80 | 97.00 | 69.14 | 64.87 | 87.65          |
| Mistral-7B-LongPO-512K   | 97.28 | 97.48 | 69.22 | 64.92 | 88.00          |





### Short Context

| Model | MMLU | ARC-C | Hellaswag | Winogrande | Avg |
|-------|-------|--------|------------|-------------|-----|
| Mistral-7B-Instruction-v0.2 | 59.15 | 59.26 | 83.2 | 78.4 | 70.00 |
| Mistral-7B-LongPO-128K | 59.99 | 59.34 | 82.99 | 78.53 | 70.21 |
| Mistral-7B-LongPO-256K-EXP | 59.47 | 60.28 | 83.14 | 78.14 | 70.26 |
| Mistral-7B-LongPO-512K-EXP | 59.51 | 60.58 | 82.87 | 77.66 | 70.16 |
| Qwen2.5-7B-Instruct | 74.28 | 67.15 | 81.41 | 74.66 | 74.38 |
| Qwen2.5-7B-LongPO-128K | 73.64 | 65.70 | 80.82 | 74.98 | 73.79 |


## Citation
If you find our project useful, hope you can star our repo and cite our paper as follows:
```
@inproceedings{
    chen2025longpo,
    title={Long{PO}: Long Context Self-Evolution of Large Language Models through Short-to-Long Preference Optimization},
    author={Guanzheng Chen and Xin Li and Michael Shieh and Lidong Bing},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=qTrEq31Shm}
}
```
