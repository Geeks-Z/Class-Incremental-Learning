#!/bin/bash
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        imr_dualprompt \
        --model vit_base_patch16_224 \
        --batch-size 24 \
        --data-path /home/team/zhaohongwei/Dataset \
        --output_dir ./imr
