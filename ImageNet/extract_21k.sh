#!/bin/bash
#SBATCH --job-name=extract_21k
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=/ocean/projects/cis250163p/tnair/logs/extract_%j.out

cd /ocean/projects/cis260039p/tnair/imagenet21k/

tar -xf winter21_whole.tar.gz

for f in *.tar; do
    mkdir -p "${f%.tar}"
    tar -xf "$f" -C "${f%.tar}/"
done
