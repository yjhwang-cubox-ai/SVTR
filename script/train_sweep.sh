#!/bin/bash -l

#SBATCH --job-name=DocOCR
#SBATCH --time=999:00:00
#SBATCH --output=./logs/sweep_%A_%a.txt
#SBATCH --partition=80g
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G               # 입력 안해주면 실행안됨(?)
#SBATCH --ntasks-per-node=1             # 각 노드에서 1개의 작업 실행

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "Number of nodes:= " "$SLURM_JOB_NUM_NODES"
echo "Ntasks per node:= "  "$SLURM_NTASKS_PER_NODE"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

run script from above
srun --container-image /purestorage/project/yjhwang/enroot_images/mmocr.sqsh \
    --container-mounts /purestorage:/purestorage \
    --no-container-mount-home \
    --container-writable \
    --container-workdir /purestorage/project/yjhwang/SVTR \
    bash -c "
    hostname -I;
    pip install argparse;
    python ligtning_sweep_slurm.py --sweep_id=$1"

# srun python -m train_sweep_slurm --sweep_id=$1