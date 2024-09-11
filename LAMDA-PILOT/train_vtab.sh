#!/bin/bash
#finetune
python main.py --config=./exps/finetune_vtab_B0_Inc5.json
python main.py --config=./exps/finetune_vtab_B0_Inc10.json
python main.py --config=./exps/finetune_vtab_B0_Inc20.json
# simplecil
python main.py --config=./exps/simplecil_vtab_B0_Inc5.json
python main.py --config=./exps/simplecil_vtab_B0_Inc10.json
python main.py --config=./exps/simplecil_vtab_B0_Inc20.json
# icarl
python main.py --config=./exps/icarl_vtab_B0_Inc5.json
python main.py --config=./exps/icarl_vtab_B0_Inc10.json
python main.py --config=./exps/icarl_vtab_B0_Inc20.json
# der
python main.py --config=./exps/der_vtab_B0_Inc5.json
python main.py --config=./exps/der_vtab_B0_Inc10.json
python main.py --config=./exps/der_vtab_B0_Inc20.json
# foster
python main.py --config=./exps/foster_vtab_B0_Inc5.json
python main.py --config=./exps/foster_vtab_B0_Inc10.json
python main.py --config=./exps/foster_vtab_B0_Inc20.json
# l2p
python main.py --config=./exps/l2p_vtab_B0_Inc5.json
python main.py --config=./exps/l2p_vtab_B0_Inc10.json
python main.py --config=./exps/l2p_vtab_B0_Inc20.json
# dual prompt
python main.py --config=./exps/dualprompt_vtab_B0_Inc5.json
python main.py --config=./exps/dualprompt_vtab_B0_Inc10.json
python main.py --config=./exps/dualprompt_vtab_B0_Inc20.json
# coda prompt
python main.py --config=./exps/coda_prompt_vtab_B0_Inc5.json
python main.py --config=./exps/coda_prompt_vtab_B0_Inc10.json
python main.py --config=./exps/coda_prompt_vtab_B0_Inc20.json
# ease
python main.py --config=./exps/ease_vtab_B0_Inc5.json
python main.py --config=./exps/ease_vtab_B0_Inc10.json
python main.py --config=./exps/ease_vtab_B0_Inc20.json




