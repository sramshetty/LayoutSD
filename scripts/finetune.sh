#!/bin/bash
#SBATCH --partition=g40x
#SBATCH --job-name=LayoutSD
#SBATCH --comment=laion
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --output=%x_%j.out
#SBATCH --requeue

module load openmpi
module load cuda/11.7

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.7/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.7:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH

export MASTER_ADDR=hostname
export MASTER_PORT=12802
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=info

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0


RUN_NAME=LayoutSD_gpt2-large_local-narratives
LANG_MODEL=gpt2-large


source /fsx/home-shivr/layoutsd_venv/bin/activate
srun --comment laion python /fsx/home-shivr/LayoutSD/finetune.py \
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
    --wandb_project ${RUN_NAME} || sbatch /fsx/home-shivr/LayoutSD/scripts/finetune.sh