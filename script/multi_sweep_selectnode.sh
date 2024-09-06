#!/bin/sh

for i in $(seq 3)
do
    sbatch script/train_sweep_selectnode.sh
done