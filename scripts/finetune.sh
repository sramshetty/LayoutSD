#!/bin/bash

CUDA_VISIBLE_DEVICES=0

python /fsx/home-shivr/LayoutSD/finetune.py \
    --caption_file /fsx/home-shivr/LayoutSD/generated_data/image_layouts_xywh.parquet \
    --model togethercomputer/RedPajama-INCITE-7B-Instruct \
    --model_f16 \
    --batch_size 1 \
    --workers 6 \
    --lr 0.00001 \
    --decay 0.01 \
    --epochs 3 \
    --warmup 3000 \
    --checkpoint_path /fsx/home-shivr/LayoutSD/checkpoint_red7b/ \
    --delete_previous_checkpoint