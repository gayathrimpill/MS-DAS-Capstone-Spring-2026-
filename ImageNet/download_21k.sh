#!/bin/bash
#SBATCH --job-name=download_21k
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=/ocean/projects/cis250163p/tnair/logs/download_%j.out

mkdir -p /ocean/projects/cis250163p/tnair/imagenet21k

wget -c "https://image-net.org/data/winter21_whole.tar.gz" \
     -P /ocean/projects/cis250163p/tnair/imagenet21k/

