#!/bin/bash -l

#SBATCH --job-name=DocOCR
#SBATCH --time=999:00:00
#SBATCH --output=./logs/sweep_%A_%a.txt
#SBATCH --partition=80g
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G               # 입력 안해주면 실행안됨(?)
#SBATCH --ntasks-per-node=1             # 각 노드에서 1개의 작업 실행
#SBATCH --cpus-per-task=16
#SBATCH --cpus-per-task=16

export RUN_NAME = "run_$SLURM_JOB_ID"

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "Number of nodes:= " "$SLURM_JOB_NUM_NODES"
echo "Ntasks per node:= "  "$SLURM_NTASKS_PER_NODE"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

# run script from above
srun --container-image /purestorage/project/yjhwang/enroot_images/MAERec5.sqsh \
    --container-mounts /purestorage:/purestorage \
    --no-container-mount-home \
    --container-writable \
    --container-workdir /purestorage/project/yjhwang/SVTR \
    bash -c "
    pip install wandb;
    pip install lightning;
    python test_sweep3-slurm.py --run_name $RUN_NAME --sweep_id $1"