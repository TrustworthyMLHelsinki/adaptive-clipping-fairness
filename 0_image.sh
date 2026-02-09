#!/bin/bash

LR=(1.0000 1.2915 1.6681 2.1544 2.7826 3.5938 4.6416 5.9948 7.7426 10.0000)
CLIP_BOUND=(0 0.0010 0.0018 0.0031 0.0055 0.0098 0.0172 0.0305 0.0539 0.0952 0.1682 0.2973 0.5254 1.0000 1.6409 2.9000 5.1252 9.0579 16.0082 28.2915 50.0000)
SEEDS=(1 2 3)

for lr in "${LR[@]}"; do
    for cb in "${CLIP_BOUND[@]}"; do
        for seed in "${SEEDS[@]}"; do

            echo "Running with clip_bound_lr=${lr}, clip_bound_lower_bound=${cb}, seed=${seed}"

            MASTER_PORT=30000 \
                MASTER_ADDR=localhost \
                WORLD_SIZE=1 \
                LOCAL_RANK=0 \
                RANK=0 \
                python run.py train \
                --num-workers 0 \
                --model-name koskela-net-grayscale \
                --dataset-name ylecun/mnist \
                --dataset-label-field label \
                --learning-rate ${lr} \
                --batch-size 6000 \
                --max-grad-norm 1 \
                --epochs 50 \
                --target-quantile 0.5 \
                --count-threshold 2.5 \
                --clip-bound-lr 0.2 \
                --clip-bound-lower-bound ${cb} \
                --target-epsilon 1.0 \
                --seed "${seed}" \
                --physical-batch-size 100 \
                --experiment-name TRAIN-TEST-RECORD-GRAD-NORMS \
                --overwrite-experiment \
                --privacy \
                --use-steps \
                --clipping-mode adaptive \
                --normalize-clipping \
                --optimizer SGD

        done
    done
done
