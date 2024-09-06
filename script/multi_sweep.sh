for i in $(seq 5)
do
    sbatch script/train_sweep.sh
done