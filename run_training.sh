#!/bin/bash
python src/run.py --tokenizer 'google/mt5-small' \
    --model 'google/mt5-small' \
    --dataset 'ALL' \
    --max_length 128 \
    --batch_size 15 \
    --learning_rate 1e-5 \
    --weight_decay 3e-3 \
    --num_epochs 1 \
    --start_from_batch 0 \
    --save_each_steps 20000
