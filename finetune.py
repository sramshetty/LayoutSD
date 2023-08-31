"""
Adapted from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/train/train.py
"""

import argparse
from contextlib import suppress
import glob
import functools
import os
import random
import shutil
import time

import numpy as np
from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
from torch.distributed.distributed_c10d import _get_default_group
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import wandb

from data.datasets import get_data
from distributed import init_distributed_device, world_info_from_env
from finetune_utils import *


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=str, help="path to parquet/s of generated samples")
    parser.add_argument("--dataset_type", type=str, choices=["dino_texts", "narrative_wds"])

    parser.add_argument("--model", default="gpt2-large", type=str)
    parser.add_argument("--freeze_lm_embeddings", action="store_true")
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="whether to use lora; will use fp32"
    )
    parser.add_argument("--offline", action="store_true")

    parser.add_argument("--num_samples", default=10000, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument(
        "--lr_schedule",
        choices=["constant", "linear", "cosine"],
        default="linear",
        type=str
    )
    parser.add_argument("--decay", default=1e-3, type=float)
    parser.add_argument("--warmup", default=200, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--run_name", default=None, type=str)
    parser.add_argument("--logging_interval", default=100, type=int)
    parser.add_argument("--checkpoint_interval", default=1, type=int)
    parser.add_argument("--delete_previous_checkpoint", action="store_true")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )

    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy", default="full", type=str, choices=["full", "hybrid"]
    )

    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )

    return parser


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def train_epoch(
    args,
    model,
    epoch,
    dataloader,
    optimizer,
    scheduler,
    device_id,
    wandb
):
    num_batches = dataloader.num_batches

    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)

    model.train()

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    for num_steps, batch in tqdm(
        enumerate(dataloader),
        disable=args.rank != 0,
        total=args.epochs * num_batches,
        initial=(epoch * num_batches),
    ):
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches

        input_ids = batch[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch[1].to(device_id, dtype=cast_dtype, non_blocking=True)

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss

        divided_loss = loss / args.gradient_accumulation_steps
        divided_loss.backward()

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches - 1
        ):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "samples_per_second": samples_per_second,
                        "samples_per_second_per_gpu": samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "ce_loss": divided_loss.item(),
                        "global_step": global_step
                    },
                    commit=True,
                )

        # Log loss to console
        if ((num_steps + 1) % args.logging_interval == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches} of epoch {epoch+1}/{args.epochs} complete. Loss: {loss.item():.3f}"
            )


def train(args):
    # Set up distributed training
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    random_seed(args.seed, args.rank)

    print("Loading model on CPU")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
    )

    if args.freeze_lm_embeddings:
        model.get_input_embeddings().requires_grad_(True)

    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    
    # Load model checkpoint on CPU
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        # if args do not specify a checkpoint to resume from, check if checkpoints exist for this run
        # and automatically resume from the latest checkpoint
        checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
            )

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        msd = checkpoint["model_state_dict"]
        msd = {k.replace("module.", ""): v for k, v in msd.items()}
        resume_from_epoch = checkpoint["epoch"] + 1

        # for fsdp, only one rank needs to load the state dict
        if not args.fsdp or args.rank == 0:
            model.load_state_dict(msd, False)

    if args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            fan_in_fan_out=True   
        )

        model = get_peft_config(model, lora_config)

        print("Trainable parameters using Lora:")
        model.print_trainable_parameters()

    # Initialize FSDP / DDP, and ensure the model is on GPU
    print(f"Initializing distributed training with {args.world_size} GPUs.")
    if args.fsdp:
        print(
            f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )

        # init MixedPrecision
        if args.precision != "fp32":
            cast_dtype = get_mp_policy_dtype(args.precision)
            mp_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=cast_dtype,  # gradient communication
                buffer_dtype=cast_dtype,
            )
        else:
            mp_policy = None

        # init process groups
        if args.fsdp_sharding_strategy == "hybrid":
            intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(
                _get_default_group()
            )
            args.my_group = intra_node_group  # for optimizer saving
            process_group = (intra_node_group, inter_node_group)  # for FSDP init
        else:
            args.my_group = None  # for optimizer saving
            process_group = None  # for FSDP init

        # init FSDP
        wrapper_kwargs = dict(
            process_group=process_group,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=device_id,
            sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
            sharding_strategy=ShardingStrategy.FULL_SHARD
            if args.fsdp_sharding_strategy == "full"
            else ShardingStrategy.HYBRID_SHARD,
            use_orig_params=False,
            mixed_precision=mp_policy,
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
        )
        # model.wrap_fsdp(wrapper_kwargs, device_id)
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=20000
        )
        model = model.to(device_id)
        ddp_model = FSDP(
            model,
            process_group=process_group,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=device_id,
            sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
            sharding_strategy=ShardingStrategy.FULL_SHARD
            if args.fsdp_sharding_strategy == "full"
            else ShardingStrategy.HYBRID_SHARD,
            use_orig_params=False,
            mixed_precision=mp_policy,
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True
        )

        print(
            f"After FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )
        print(
            f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}"
        )

    else:
        model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

    # Initialize optimizer - using no weight decay or small weight decay for all parameters
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=args.lr,
        weight_decay=args.decay,
    )

    # load optimizer checkpoint
    if args.resume_from_checkpoint is not None:
        osd = checkpoint["optimizer_state_dict"]
        if args.fsdp:
            osd = FSDP.optim_state_dict_to_load(osd, ddp_model, optimizer)
        optimizer.load_state_dict(osd)

    print("Loading Data...")
    dataset = get_data(args, tokenizer=tokenizer)

    total_training_steps = (
        (args.num_samples) // (args.batch_size * args.world_size) 
    ) * args.epochs
    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    if args.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup,
            num_training_steps=args.epochs * args.num_samples
        )
    elif args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup,
            num_training_steps=args.epochs * args.num_samples
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps
        )
    
    # load lr scheduler checkpoint
    if args.resume_from_checkpoint is not None:
        scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Start training!
    ddp_model.train()

    for epoch in range(resume_from_epoch, args.epochs):
        dataset.set_epoch(epoch)
        dataloader = dataset.dataloader

        train_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device_id=device_id,
            wandb=wandb
        )

        save_checkpoint(ddp_model, optimizer, scheduler, epoch, args)

    save_checkpoint(ddp_model, optimizer, scheduler, args.epochs - 1, args)


if __name__ == "__main__":
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = get_args_parser()
    args = parser.parse_args()

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.fsdp and args.fsdp_sharding_strategy == "hybrid":
        print(
            "Warning: As of torch=2.0.1, the FSDP logic for optim_state_dict() is broken for hybrid sharding."
            + "To make this method work, we need to modify torch.distributed.fsdp._optim_utils.py"
            + "Copy and paste the code from the _optim_utils.py in this repo into the torch file."
            + "The main issue was the missing group kwarg on line 1596 in _all_gather_optim_state."
        )

    if args.lora and not args.model.startswith("gpt2"):
        raise ValueError("currently limiting lora experiments to gpt2 models")
    if args.lora and args.fsdp:
        raise ValueError("fsdp will likely require manual layer wrapping: https://github.com/pytorch/pytorch/issues/91165")

    print("Training...")
    train(args)

    print("Done!")
