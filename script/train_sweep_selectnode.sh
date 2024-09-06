#!/bin/bash -l

#SBATCH --job-name=DocOCR
#SBATCH --time=999:00:00
#SBATCH --output=./logs/sweep_%A_%a.txt
#SBATCH --nodelist=hpe160
#SBATCH --nodes=1  
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1             # 각 노드에서 1개의 작업 실행
#SBATCH --cpus-per-task=32
#SBATCH --mem=256gb

##### Number of total processes

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
    python test_sweep2-slurm.py"