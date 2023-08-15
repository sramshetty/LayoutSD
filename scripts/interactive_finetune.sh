#!/bin/bash

RUN_NAME=LayoutSD_gpt2-large_local-narratives
LANG_MODEL=gpt2-large

python /fsx/home-shivr/LayoutSD/finetune.py \
    --captions "pipe:aws s3 cp s3://laion-west/flickr-narrative-boxes/output_wds/{00000..01392..2}.tar -" \
    --dataset_type narratives \
    --model ${LANG_MODEL} \
    --num_samples 50000 \
    --batch_size 16 \
    --workers 10 \
    --lr 0.0001 \
    --decay 0.01 \
    --epochs 10 \
    --warmup 3000 \
    --checkpoint_path /fsx/home-shivr/LayoutSD/checkpoint_gpt2-large \
    --delete_previous_checkpoint \
    --report_to_wandb \
    --wandb_project ${RUN_NAME}