python3 data_prepare/generate_longpo_pairs.py \
        --model_path model_path \
        --model_name model_name \
        --data_path path/to/qa_data \
        --save_path save_path \
        --process_length 131072 \
        --tensor_parallel_size 1 \
        --world_size 16 \
        --rank 0