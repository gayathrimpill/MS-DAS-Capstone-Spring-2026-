#!/bin/bash
#SBATCH --job-name=moe_cifar100
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=h100-80:4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --output=/ocean/projects/cis250163p/tnair/logs/moe_%j.out
#SBATCH --error=/ocean/projects/cis250163p/tnair/logs/moe_%j.err

export PROJECT_ROOT=/ocean/projects/cis250163p/tnair

export WANDB_DIR=$PROJECT_ROOT/wandb
export WANDB_CACHE_DIR=$PROJECT_ROOT/wandb/.cache
export TMPDIR=$PROJECT_ROOT/tmp
export TORCH_HOME=$PROJECT_ROOT/.torch
export HF_HOME=$PROJECT_ROOT/hf_cache
export CKPT_DIR=$PROJECT_ROOT/checkpoints
export MPLCONFIGDIR=$PROJECT_ROOT/.mplconfig
export PYTHONUNBUFFERED=1

mkdir -p $PROJECT_ROOT/{logs,checkpoints,wandb,tmp,.torch,hf_cache,.mplconfig}

source /ocean/projects/cis250163p/tnair/capstone-pip/bin/activate

cd $PROJECT_ROOT

torchrun --nproc_per_node=4 cap_baseline_moe.py