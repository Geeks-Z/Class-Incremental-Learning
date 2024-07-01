#!/bin/bash
python main.py --config=./exps/finetune_cifar_B50_Inc5.json
python main.py --config=./exps/icarl_cifar_B50_Inc5.json
python main.py --config=./exps/der_cifar_B50_Inc5.json
python main.py --config=./exps/foster_B50_Inc5.json
python main.py --config=./exps/l2p_B50_Inc5.json
python main.py --config=./exps/dualprompt_B50_Inc5.json
python main.py --config=./exps/coda_prompt_B50_Inc5.json
python main.py --config=./exps/simplecil_cifar_B50_Inc5.json
python main.py --config=./exps/ease_B50_Inc5.json
