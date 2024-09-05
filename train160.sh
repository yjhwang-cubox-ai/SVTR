#!/bin/bash
#SBATCH --job-name=KRLarge
#SBATCH --output=KRTest.out
#SBATCH --partition=all
#SBATCH --gpus=6
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-gpu=8


##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "Number of nodes:= " "$SLURM_JOB_NUM_NODES"
echo "Ntasks per node:= "  "$SLURM_NTASKS_PER_NODE"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

srun python -m ligtning_multi-test