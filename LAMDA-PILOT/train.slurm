#!/bin/bash
#SBATCH -J omn #作业名称
#SBATCH -p gpu-normal # 提交到指定分区
#SBATCH -N 1 # 使用一个节点
#SBATCH -n 8 # 使用进程（cpu 核）
#SBATCH --gres=gpu:1
#SBATCH -o ./res/3rd-omn-stdout.log
#SBATCH -e ./res/3rd-omn-stderr.log

module load anacoda3
source activate cil

python main.py --config=./exps/finetune_omn_B0_Inc5.json
python main.py --config=./exps/finetune_omn_B0_Inc10.json
python main.py --config=./exps/finetune_omn_B0_Inc20.json
python main.py --config=./exps/finetune_omn_B0_Inc30.json
python main.py --config=./exps/finetune_omn_B150_Inc5.json
python main.py --config=./exps/finetune_omn_B150_Inc10.json
python main.py --config=./exps/icarl_omn_B0_Inc5.json
python main.py --config=./exps/icarl_omn_B0_Inc10.json
python main.py --config=./exps/icarl_omn_B0_Inc20.json
python main.py --config=./exps/icarl_omn_B0_Inc30.json
python main.py --config=./exps/icarl_omn_B150_Inc5.json
python main.py --config=./exps/icarl_omn_B150_Inc10.json
python main.py --config=./exps/der_omn_B0_Inc5.json
python main.py --config=./exps/der_omn_B0_Inc10.json
python main.py --config=./exps/der_omn_B0_Inc20.json
python main.py --config=./exps/der_omn_B0_Inc30.json
python main.py --config=./exps/der_omn_B150_Inc5.json
python main.py --config=./exps/der_omn_B150_Inc10.json
python main.py --config=./exps/foster_omn_B0_Inc5.json
python main.py --config=./exps/foster_omn_B0_Inc10.json
python main.py --config=./exps/foster_omn_B0_Inc20.json
python main.py --config=./exps/foster_omn_B0_Inc30.json
python main.py --config=./exps/foster_omn_B150_Inc5.json
python main.py --config=./exps/foster_omn_B150_Inc10.json
python main.py --config=./exps/l2p_omn_B0_Inc5.json
python main.py --config=./exps/l2p_omn_B0_Inc10.json
python main.py --config=./exps/l2p_omn_B0_Inc20.json
python main.py --config=./exps/l2p_omn_B0_Inc30.json
python main.py --config=./exps/l2p_omn_B150_Inc5.json
python main.py --config=./exps/l2p_omn_B150_Inc10.json
python main.py --config=./exps/dualprompt_omn_B0_Inc5.json
python main.py --config=./exps/dualprompt_omn_B0_Inc10.json
python main.py --config=./exps/dualprompt_omn_B0_Inc20.json
python main.py --config=./exps/dualprompt_omn_B0_Inc30.json
python main.py --config=./exps/dualprompt_omn_B150_Inc5.json
python main.py --config=./exps/dualprompt_omn_B150_Inc10.json
python main.py --config=./exps/coda_prompt_omn_B0_Inc5.json
python main.py --config=./exps/coda_prompt_omn_B0_Inc10.json
python main.py --config=./exps/coda_prompt_omn_B0_Inc20.json
python main.py --config=./exps/coda_prompt_omn_B0_Inc30.json
python main.py --config=./exps/coda_prompt_omn_B150_Inc5.json
python main.py --config=./exps/coda_prompt_omn_B150_Inc10.json
python main.py --config=./exps/simplecil_omn_B0_Inc5.json
python main.py --config=./exps/simplecil_omn_B0_Inc10.json
python main.py --config=./exps/simplecil_omn_B0_Inc20.json
python main.py --config=./exps/simplecil_omn_B0_Inc30.json
python main.py --config=./exps/simplecil_omn_B150_Inc5.json
python main.py --config=./exps/simplecil_omn_B150_Inc10.json
python main.py --config=./exps/ease_omn_B0_Inc5.json
python main.py --config=./exps/ease_omn_B0_Inc10.json
python main.py --config=./exps/ease_omn_B0_Inc20.json
python main.py --config=./exps/ease_omn_B0_Inc30.json
python main.py --config=./exps/ease_omn_B150_Inc5.json
python main.py --config=./exps/ease_omn_B150_Inc10.json
