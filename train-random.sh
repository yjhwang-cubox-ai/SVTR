#!/bin/bash
#SBATCH --job-name=Docu1
#SBATCH --output=nb-hpe160.out
#SBATCH --partition=80g
#SBATCH --gres=gpu:5
#SBATCH --mem-per-gpu=80G

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "Number of nodes:= " "$SLURM_JOB_NUM_NODES"
echo "Ntasks per node:= "  "$SLURM_NTASKS_PER_NODE"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

srun python -m ligtning_multi