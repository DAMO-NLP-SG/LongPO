## LongPO Data Generation and Preparation

#### Step 1: Generating instructions and answers from short contexts

- You may need to process the plain text corpus (e.g., [Long-Data-Collection](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections) by each domain to avoid memory overflow. For a specific process (target) length (e.g., 131072), we recommend to speedup answer generation for long/short context using multiple processes (ranks) in parallel. You can specify the `word_size`(number of processes) and `rank`(process id) to run each process on a single GPU.
  

```bash
python3 data_prepare/generate_qa.py \
    --data_path path/to/domain_data \
    --save_path folder/to/save_qa_data \
    --model_path model_path \
    --model_name model_name \
    --process_length 131072 \ # Target long length
    --max_chunk_length 32768 \ # Maximum length of short chunks
    --domain domain \
    --world_size num_of_processes \
    --rank process_id
```

- Suppose using 8 GPUs for generation, you can run
  

```bash
for IDX in $(seq 0 7); do
python3 data_prepare/generate_qa.py \
    --data_path path/to/domain_data \
    --save_path folder/to/save_qa_data \
    --model_path model_path \
    --model_name model_name \
    --process_length 131072 \ # Target long length
    --max_chunk_length 32768 \ # Maximum length of short chunks
    --domain domain \
    --world_size 8 \
    --rank $IDX &
done
wait
```

- After generation for each domain, postprocess by
  

```bash
python3 data_prepare/postprocess_qa.py \
        --domain $domain \
        --data_path folder/to/save_qa_data/domain/xxxk_chunks \
        --save_path path/to/save_qa.json
```

- If more than one domain, you may need to merge all into one json for next step.
  

#### Step 2: Generating Multi-Turn Short-to-Long Preference Data

- Given the instruction generated above, you can generation answers for long context using
  

```bash
python3 data_prepare/generate_longpo_pairs.py \
        --model_path model_path \
        --model_name model_name \
        --data_path path/to/save_qa.json \
        --save_path folder/to/save_longpo_pairs \
        --process_length 131072 \
        --tensor_parallel_size 1 \
        --world_size 16 \
        --rank 0
```
- Please adjust the RoPE $$\theta$$ (if needed) before generating answers for long context.
- You would collect splits of preference data for each rank.
  

#### Step 3: Format and Tokenize

- Now we can format and tokenize data samples in each split for training:
  

```bash
python3 data_prepare/format_tokenize.py \
    --model_path model_path \
    --data_path folder/to/save_longpo_pairs \
    --save_path folder/to/save_longpo_pairs_format \
    --split split_id
```

- If you have multiple splits, please merge them into one:
  

```bash
python3 data_prepare/merge_datasets.py \
    --data_path folder/to/save_longpo_pairs_format \
    --save_path final_training_dataset
```