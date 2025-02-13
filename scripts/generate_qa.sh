python3 data_prepare/generate_qa.py \
    --data_path path/to/domain_data \
    --save_path path/to/save \
    --model_path model_path \
    --model_name model_name \
    --process_length 131072 \
    --max_chunk_length 32768 \
    --domain domain \
    --world_size num_of_processes \
    --rank process_id