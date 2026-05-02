#!/bin/bash
#SBATCH --job-name=download_1k
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --output=/ocean/projects/cis250163p/tnair/logs/download_%j.out

export HF_HOME=/ocean/projects/cis250163p/tnair/hf_cache

source /ocean/projects/cis250163p/tnair/capstone-pip/bin/activate

python -c "
from datasets import load_dataset
load_dataset('imagenet-1k', cache_dir='/ocean/projects/cis250163p/tnair/imagenet_1k')
"
