#!/bin/bash
#SBATCH --job-name=SVTRImpl
#SBATCH --output=nb-hpe160.out
#SBATCH --nodelist=hpe160
#SBATCH --gpus=5
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=16

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