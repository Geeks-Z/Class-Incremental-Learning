cd Code/Research/CIL/Ease/&&
conda activate peft

# all
nohup ./train.sh > ./res/ease-all-dataset.out 2>&1 &

# 4090
cd lrh/Code/Research/CIL/LAMDA-PILOT/ &&
conda activate cil

# 超算
cd Code/Research/CIL/LAMDA-PILOT/ &&
sbatch train.slurm




