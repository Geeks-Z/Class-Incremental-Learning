#!/bin/bash
#SBATCH -J FOSTER #作业名称
#SBATCH -p gpu-quota # 提交到指定分区
#SBATCH -N 1 # 使用一个节点
#SBATCH -n 12 # 使用多个进程（cpu 核）
#SBATCH --gres=gpu:1
#SBATCH -o foster-imagenet100-50-10.out
#SBATCH -e foster-imagenet100-50-10.err

module load anaconda3
source activate torch_hwzhao

# cifar100
#python main.py --config=foster-cifar100.json

# imagenet100
python main.py --config=foster-imagenet100.json

# 处理imagenet
# python CopyImg.py