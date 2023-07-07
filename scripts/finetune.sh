#!/bin/bash

# CUDA_VISIBLE_DEVICES=0

python /fsx/home-shivr/LayoutSD/finetune.py \
    --caption_file /fsx/home-shivr/LayoutSD/generated_data/image_layouts_xywh.parquet \
    --model togethercomputer/RedPajama-INCITE-Instruct-3B-v1 \
    --model_bfp16 \
    --batch_size 4 \
    --workers 6 \
    --lr 0.00001 \
    --decay 0.01 \
    --epochs 3 \
    --warmup 1500 \
    --checkpoint_path /fsx/home-shivr/LayoutSD/checkpoint_red3b/ \
    --delete_previous_checkpoint