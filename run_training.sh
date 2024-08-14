#!/bin/bash
python src/run.py --tokenizer 'google/mt5-small' \
    --model 'werent4/mt5TranslatorLT' \
    --dataset 'scoris/en-lt-merged-data' \
    --max_length 128 \
    --batch_size 14 \
    --learning_rate 1e-7 \
    --num_epochs 1 \
    --start_from_batch 22000 \
    --save_each_steps 10000
