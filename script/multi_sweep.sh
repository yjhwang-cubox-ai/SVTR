for i in $(seq 5)
do
    sbatch script/train_sweep.sh 1w1gm
done