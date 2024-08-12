cd Code/Research/CIL/LAMDA-PILOT/&&
conda activate peft

# 4090
cd lrh/Code/Research/CIL/EvoPrompt-main/ &&
conda activate cil

# 参考

nohup ./train.sh > ./res/B0-Inc5.out 2>&1 &

nohup ./train_inr_B0_Inc5.sh > ./res/4090-baseline-inr-B0-Inc5.out 2>&1 &

