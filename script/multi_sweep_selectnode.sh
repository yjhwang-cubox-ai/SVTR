for i in $(seq 5)
do
    sbatch script/train_sweep_selectnode.sh
done