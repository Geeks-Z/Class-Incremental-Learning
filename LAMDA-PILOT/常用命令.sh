cd Code/Research/CIL/LAMDA-PILOT/&&
conda activate peft

# 待完成

# 4090
cd lrh/Code/Research/CIL/LAMDA-PILOT/ &&
conda activate cil

nohup ./train_cifar_B0_Inc5.sh > ./res/4090-baseline-cifar-B0-Inc5.out 2>&1 &

nohup ./train_inr_B0_Inc5.sh > ./res/4090-baseline-inr-B0-Inc5.out 2>&1 &

nohup ./train_cub_B0_Inc10.sh > ./res/4090-baseline-cub-B0-Inc10.out 2>&1 &

nohup ./train_ina_B0_Inc20.sh > ./res/4090-baseline-ina-B0-Inc20.out 2>&1 &

nohup ./train_omn_B0_Inc30.sh > ./res/4090-baseline-omn-B0-Inc30.out 2>&1 &

# memory对比

nohup ./train_vtab_B0_Inc5.sh >./res/memory-baseline-vtab-B0-Inc5.out 2>&1 &

nohup ./train_cifar_B0_Inc10.sh >./res/memory-baseline-cifar-B0-Inc10.out 2>&1 &

nohup ./train_cub_B0_Inc5.sh >./res/memory-baseline-cub-B0-Inc5.out 2>&1 &

nohup ./train_omn_B0_Inc10.sh >./res/memory-baseline-omn-B0-Inc10.out 2>&1 &

-------
nohup ./train_omn_B0_Inc30.sh > ./res/supp-baseline-omn-B0-Inc30-ease.out 2>&1 &

nohup ./train_omn_B150_Inc5.sh > ./res/2nd-baseline-omn-B150-Inc5.out 2>&1 &

nohup ./train_cub_B100_Inc5.sh > ./res/2nd-baseline-cub-B100-Inc5.out 2>&1 &

nohup ./train_omn_B0_Inc30.sh > ./res/3nd-baseline-omn-B0-Inc30.out 2>&1 &

nohup ./train_cifar_B0_Inc5.sh > ./res/3nd-baseline-cifar-B0-Inc5.out 2>&1 &

nohup ./train_cub_B0_Inc10.sh > ./res/3nd-baseline-cub-B0-Inc10.out 2>&1 &

nohup ./train_inr_B0_Inc5.sh > ./res/3nd-baseline-inr-B0-Inc5.out 2>&1 &

nohup ./train_ina_B0_Inc20.sh > ./res/3nd-baseline-ina-B0-Inc20.out 2>&1 &

nohup ./train_vtab_B0_Inc10.sh > ./res/3nd-baseline-vtab-B0-Inc10.out 2>&1 &