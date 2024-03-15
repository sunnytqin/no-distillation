#!/bin/bash

#SBATCH -c 16 # Number of cores requested
#SBATCH -t 5-00:00 # Runtime in minutes
#SBATCH -p kempner # Partition to submit to
#SBATCH --mem=250G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=kempner_barak_lab
#SBATCH --exclude=holygpu8a19604,holygpu8a19303

module purge
module load Mambaforge
mamba activate invariance

RUN_NAME="ResNet18 full training"
echo $RUN_NAME

echo "torchrun --nproc_per_node=1  train.py --model=resnet18 --data-path=/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet256 -b=256 --output-dir=/n/holyscratch01/barak_lab/Lab/sqin/invariance/experts/ResNet18 --print-freq=100"

torchrun --nproc_per_node=1  train.py --model=resnet18 --data-path=/n/holystore01/LABS/barak_lab/Everyone/datasets/imagenet256 -b=256 --output-dir=/n/holyscratch01/barak_lab/Lab/sqin/invariance/experts/ResNet18 --print-freq=100

