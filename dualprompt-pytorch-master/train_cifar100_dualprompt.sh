#!/bin/bash
export CUDA_VISIBLE_DEVICES="2"
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        cifar100_dualprompt \
        --model vit_base_patch16_224 \
        --batch-size 128 \
        --data-path /home/team/zhaohongwei/Dataset \
        --output_dir ./cifar100