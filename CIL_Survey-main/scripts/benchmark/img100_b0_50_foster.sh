CUDA_VISIBLE_DEVICES=3 python main.py \
    -model foster \
    --dataset imagenet100 \
    -ms 2000 \
    -init 50 \
    -incre 50 \
    -net resnet18 \
    -p benchmark \
    -d 0 \
    --skip