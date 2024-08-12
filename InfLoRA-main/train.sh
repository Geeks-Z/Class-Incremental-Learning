#!/bin/bash
python main.py --config=./configs/cifar100_inflora.json
python main.py --config=./configs/cub200_inflora.json
#python main.py --config=./configs/inr_inflora.json
python main.py --config=./configs/ina_inflora.json
python main.py --config=./configs/omn_inflora.json
#python main.py --config=./configs/vtab_inflora.json