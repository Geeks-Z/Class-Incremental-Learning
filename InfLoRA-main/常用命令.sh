cd Code/LoRAMoE-CIL/Ablation-Study-LoRA-Rank &&
conda activate peft

# 4090
cd lrh/Code/Research/CIL/InfLoRA-main/ &&
conda activate cil

----------------------

nohup ./train.sh > ./res/2nd-baseline-rank4-alldataset.out 2>&1 &

nohup ./train.sh > ./res/2nd-B50-rank4-alldatasets.out 2>&1 &

nohup ./train.sh > ./res/supp-B50-rank4-omn.out 2>&1 &


------------------------------

nohup ./train.sh > ./res/2nd-rank-16-alldataset.out 2>&1 &

nohup ./train.sh > ./res/2nd-rank-32-alldataset.out 2>&1 &

