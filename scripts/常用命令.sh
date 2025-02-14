cd Code/Research/Awesome-CIL&&
conda activate cil
# 4090
cd lrh/Code/Research/CIL/LAMDA-PILOT/ &&
conda activate cil
# 超算
cd Code/Research/CIL/LAMDA-PILOT/ &&
sbatch train.slurm

TODO
nohup ./scripts/train_memory.sh > ./res/memory-inr-supp.out 2>&1 &
nohup ./scripts/train_lae.sh > ./res/3rd-lae.out 2>&1 &
----------------------------------------------------------------------------------------------------------
#cifar
nohup ./train_cifar.sh > ./log/cifar-stdout.log 2> ./log/cifar-stderr.log &
nohup ./train_cifar.sh > ./res/cifar.out 2>&1 &
#cub
nohup ./train_cub.sh > ./log/cub-stdout.log 2> ./log/cub-stderr.log &
nohup ./train_cub.sh > ./res/cub.out 2>&1 &
#inr
nohup ./train_inr.sh > ./log/inr-stdout.log 2> ./log/inr-stderr.log &
nohup ./train_inr.sh > ./res/inr.out 2>&1 &
#ina
nohup ./train_ina.sh > ./log/ina-stdout.log 2> ./log/ina-stderr.log &
nohup ./train_ina.sh > ./res/ina.out 2>&1 &
#omn
nohup ./train_omn.sh > ./log/omn-stdout.log 2> ./log/omn-stderr.log &
nohup ./train_omn.sh > ./res/omn.out 2>&1 &
#vtab
nohup ./train_vtab.sh > ./log/vtab-stdout.log 2> ./log/vtab-stderr.log &
nohup ./train_vtab.sh > ./res/vtab.out 2>&1 &
----------------------------------------------------------------------------------------------------------