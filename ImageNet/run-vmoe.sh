#!/bin/bash
#SBATCH --job-name=vmoe_imagenet
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100-80:4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --output=/ocean/projects/cis250163p/tnair/logs/moe_%j.out
#SBATCH --error=/ocean/projects/cis250163p/tnair/logs/moe_%j.err

export PROJECT_ROOT=/ocean/projects/cis250163p/tnair
export DATA_ROOT=/ocean/projects/cis250163p/tnair

export WANDB_DIR=$PROJECT_ROOT/wandb
export WANDB_CACHE_DIR=$PROJECT_ROOT/wandb/.cache
export TMPDIR=/local
export TORCH_HOME=$PROJECT_ROOT/.torch
export HF_HOME=$DATA_ROOT/hf_cache
export HF_CACHE_DIR=$DATA_ROOT/imagenet_1k
export CKPT_DIR=$PROJECT_ROOT/checkpoints
export MPLCONFIGDIR=$PROJECT_ROOT/.mplconfig
export PYTHONUNBUFFERED=1
export HF_TOKEN=$(cat /jet/home/tnair/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

mkdir -p $PROJECT_ROOT/{logs,checkpoints,wandb,tmp,.torch,.mplconfig}

source /ocean/projects/cis250163p/tnair/capstone-pip/bin/activate

cd $PROJECT_ROOT

torchrun --nproc_per_node=4 cap_baseline_moe_vmoe.py