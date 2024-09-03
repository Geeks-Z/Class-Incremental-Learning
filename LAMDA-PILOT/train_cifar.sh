#!/bin/bash
#python main.py --config=./exps/finetune_cifar_B0_Inc5.json
#python main.py --config=./exps/icarl_cifar_B0_Inc5.json
#python main.py --config=./exps/der_cifar_B0_Inc5.json
#python main.py --config=./exps/foster_cifar_B0_Inc5.json
#python main.py --config=./exps/l2p_cifar_B0_Inc5.json
#python main.py --config=./exps/dualprompt_cifar_B0_Inc5.json
#python main.py --config=./exps/coda_prompt_cifar_B0_Inc5.json
#python main.py --config=./exps/simplecil_cifar_B0_Inc5.json
#python main.py --config=./exps/ease_cifar_B0_Inc5.json
# coda prompt
python main.py --config=./exps/coda_prompt_cifar_B0_Inc5.json
python main.py --config=./exps/coda_prompt_cifar_B0_Inc10.json
python main.py --config=./exps/coda_prompt_cifar_B0_Inc20.json
python main.py --config=./exps/coda_prompt_cifar_B50_Inc5.json
python main.py --config=./exps/coda_prompt_cifar_B50_Inc10.json

