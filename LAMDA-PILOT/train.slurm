#!/bin/bash
#SBATCH -J conda #作业名称
#SBATCH -p gpu-quota # 提交到指定分区
#SBATCH -N 1 # 使用一个节点
#SBATCH -n 8 # 使用进程（cpu 核）
#SBATCH --gres=gpu:1
#SBATCH -o ./res/condaprompt-cifar.out
#SBATCH -e ./res/condaprompt-cifa.err

module load anaconda3
source activate cil

# l2p-cifar
#python main.py --config=./exps/l2p_cifar_B0_Inc5.json
#python main.py --config=./exps/l2p_cifar_B0_Inc10.json
#python main.py --config=./exps/l2p_cifar_B0_Inc20.json
#python main.py --config=./exps/l2p_cifar_B50_Inc5.json
#python main.py --config=./exps/l2p_cifar_B50_Inc10.json

# dualprompt
#python main.py --config=./exps/dualprompt_cifar_B0_Inc5.json
#python main.py --config=./exps/dualprompt_cifar_B0_Inc10.json
#python main.py --config=./exps/dualprompt_cifar_B0_Inc20.json
#python main.py --config=./exps/dualprompt_cifar_B50_Inc5.json
#python main.py --config=./exps/dualprompt_cifar_B50_Inc10.json

# conda prompt
python main.py --config=./exps/conda_prompt_cifar_B0_Inc5.json
python main.py --config=./exps/conda_prompt_cifar_B0_Inc10.json
python main.py --config=./exps/conda_prompt_cifar_B0_Inc20.json
python main.py --config=./exps/conda_prompt_cifar_B50_Inc5.json
python main.py --config=./exps/conda_prompt_cifar_B50_Inc10.json

##
