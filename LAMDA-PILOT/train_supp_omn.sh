#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python main.py --config=./supp_exps/der_omn_B0_Inc20.json
python main.py --config=./supp_exps/foster_omn_B0_Inc5.json
python main.py --config=./supp_exps/foster_omn_B0_Inc10.json
python main.py --config=./supp_exps/foster_omn_B0_Inc20.json
python main.py --config=./supp_exps/foster_omn_B0_Inc30.json
python main.py --config=./supp_exps/foster_omn_B150_Inc5.json
python main.py --config=./supp_exps/foster_omn_B150_Inc10.json
python main.py --config=./supp_exps/l2p_omn_B0_Inc5.json
python main.py --config=./supp_exps/l2p_omn_B0_Inc10.json
python main.py --config=./supp_exps/l2p_omn_B0_Inc20.json
python main.py --config=./supp_exps/l2p_omn_B0_Inc30.json
python main.py --config=./supp_exps/l2p_omn_B150_Inc5.json
python main.py --config=./supp_exps/l2p_omn_B150_Inc10.json
python main.py --config=./supp_exps/dualprompt_omn_B0_Inc5.json
python main.py --config=./supp_exps/dualprompt_omn_B0_Inc10.json
python main.py --config=./supp_exps/dualprompt_omn_B0_Inc20.json
python main.py --config=./supp_exps/dualprompt_omn_B0_Inc30.json
python main.py --config=./supp_exps/dualprompt_omn_B150_Inc5.json
python main.py --config=./supp_exps/dualprompt_omn_B150_Inc10.json
python main.py --config=./supp_exps/coda_prompt_omn_B0_Inc5.json
python main.py --config=./supp_exps/coda_prompt_omn_B0_Inc10.json
python main.py --config=./supp_exps/coda_prompt_omn_B0_Inc20.json
python main.py --config=./supp_exps/coda_prompt_omn_B0_Inc30.json
python main.py --config=./supp_exps/coda_prompt_omn_B150_Inc5.json
python main.py --config=./supp_exps/coda_prompt_omn_B150_Inc10.json
python main.py --config=./supp_exps/simplecil_omn_B0_Inc5.json
python main.py --config=./supp_exps/ease_omn_B0_Inc5.json
python main.py --config=./supp_exps/ease_omn_B0_Inc10.json
python main.py --config=./supp_exps/ease_omn_B0_Inc20.json
python main.py --config=./supp_exps/ease_omn_B0_Inc30.json
python main.py --config=./supp_exps/ease_omn_B150_Inc5.json
python main.py --config=./supp_exps/ease_omn_B150_Inc10.json




