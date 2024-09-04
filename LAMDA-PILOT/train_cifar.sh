#!/bin/bash
# coda prompt
#python main.py --config=./exps/coda_prompt_cifar_B0_Inc5.json
#python main.py --config=./exps/coda_prompt_cifar_B0_Inc10.json
#python main.py --config=./exps/coda_prompt_cifar_B0_Inc20.json
#python main.py --config=./exps/coda_prompt_cifar_B50_Inc5.json
#python main.py --config=./exps/coda_prompt_cifar_B50_Inc10.json

# der
python main.py --config=./exps/der_cifar_B0_Inc5.json
python main.py --config=./exps/der_cifar_B0_Inc10.json
python main.py --config=./exps/der_cifar_B0_Inc20.json
python main.py --config=./exps/der_cifar_B50_Inc5.json
python main.py --config=./exps/der_cifar_B50_Inc10.json

# foster
#python main.py --config=./exps/foster_cifar_B0_Inc5.json
#python main.py --config=./exps/foster_cifar_B0_Inc10.json
#python main.py --config=./exps/foster_cifar_B0_Inc20.json
#python main.py --config=./exps/foster_cifar_B50_Inc5.json
#python main.py --config=./exps/foster_cifar_B50_Inc10.json