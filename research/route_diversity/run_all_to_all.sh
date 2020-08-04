#!/usr/bin/env bash

#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=2500M
#SBATCH --array=0-9

#SBATCH
srun python3 research/route_diversity/routing_scripts.py -njpa austin_2017 -sai $SLURM_ARRAY_TASK_ID -sal 10 -sf 0.1