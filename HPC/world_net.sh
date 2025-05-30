#!/usr/bin/env bash
#SBATCH -J 'wnet'
#SBATCH -o slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

module purge
module load cudatoolkit/12.6
module load anaconda3/2022.5

conda activate maze_env
python -u world_net.py


