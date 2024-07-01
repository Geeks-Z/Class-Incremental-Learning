#!/bin/bash
#SBATCH -J PyCIL-SSRE #作业名称
#SBATCH -p gpu-quota # 提交到指定分区
#SBATCH -N 1 # 使用一个节点
#SBATCH -n 8 # 使用进程（cpu 核）
#SBATCH --gres=gpu:4
#SBATCH -o ssre-cifar100-5steps.out
#SBATCH -e ssre-cifar100-5steps.err

module load anaconda3
source activate torch_hwzhao

# bic
# python ./main.py --config=./exps/bic.json

# wa
# python ./main.py --config=./exps/wa.json

# RMM
# python ./rmm_train.py --config=./exps/rmm-pretrain.json

# ssre
python ./main.py --config=./exps/ssre.json