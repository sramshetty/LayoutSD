import argparse
from ast import literal_eval
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_file", type=str, help="path to parquet of generated samples")

    parser.add_argument("--model", default="gpt2-large", type=str)
    parser.add_argument("--model_bfp16", action="store_true")

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--workers", default=6, type=int)

    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_schedule", default="linear", type=str)
    parser.add_argument("--decay", default=1e-3, type=float)
    parser.add_argument("--warmup", default=200, type=int)

    parser.add_argument("--logging_interval", default=100, type=int)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--checkpoint_interval", default=1, type=int)
    parser.add_argument("--delete_previous_checkpoint", action="store_true")

    return parser


class CaptionBoxesDataset(Dataset):
    def __init__(self, caption_file: str):
        df = pd.read_parquet(caption_file)
        self.captions = df['caption'].to_list()
        self.boxes = df['boxes'].to_list()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        bboxes = [literal_eval(bbox) for bbox in self.boxes[idx]]
        bbox_str = str([(phrase, literal_eval(box)) for phrase, box in bboxes])
        return "Caption: " + self.captions[idx] + "\nObjects: " + bbox_str


def train_epoch(args, model, dataloader, optimizer, scheduler, epoch):
    num_steps = len(dataloader)
    model.train()

    for i, targets in tqdm(
        enumerate(dataloader),
        total=args.epochs * num_steps,
        initial=(epoch * num_steps),
    ):
        optimizer.zero_grad()

        targets.to(device)
        with autocast(dtype=torch.bfloat16):
            outputs = model(
                **targets,
                labels=targets['input_ids'],
            )
            loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % args.logging_interval == 0:
            print(f"\nLoss at step {i}/{num_steps} of epoch {epoch}/{args.epochs}: {loss}")


def train(args, dataset):

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.model_bfp16 else torch.float32
    ).to(device)
    
    def collate_fn(batch):
        targets = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors='pt',
            return_attention_mask=True
        )

        return targets
    
    dataloder = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    if args.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup,
            num_training_steps=args.epochs * len(dataloder)
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup,
            num_training_steps=args.epochs * len(dataloder)
        )

    for epoch in range(args.epochs):
        train_epoch(
            args,
            model,
            dataloder,
            optimizer,
            scheduler,
            epoch
        )

        if args.checkpoint_path is not None and epoch % args.checkpoint_interval == 0:
            print(f"Saving checkpoint for epoch {epoch}...")
            if epoch != 0 and args.delete_previous_checkpoint:
                shutil.rmtree(args.checkpoint_path + f"checkpoint{epoch-1}")
            model.save_pretrained(args.checkpoint_path + f"checkpoint{epoch}")


if __name__ == "__main__":
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = get_args_parser()
    args = parser.parse_args()

    print("Loading Data...")
    dataset = CaptionBoxesDataset(args.caption_file)

    print("Training...")
    train(args, dataset)

    print("Done!")

