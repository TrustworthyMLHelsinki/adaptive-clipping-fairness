#!/bin/bash
MASTER_PORT=30000 \
    MASTER_ADDR=localhost \
    WORLD_SIZE=1 \
    LOCAL_RANK=0 \
    RANK=0 \
    python run.py optimize \
    --num-workers 0 \
    --model-name koskela-net-grayscale \
    --dataset-name ylecun/mnist \
    --dataset-label-field label \
    --target-hypers learning_rate \
    --target-hypers batch_size \
    --max-grad-norm 1 \
    --epochs 2 \
    --target-epsilon 1 \
    --n-trials 20 \
    --seed 42 \
    --physical-batch-size 100 \
    --normalize-clipping \
    --optuna-config ./conf/adapt_random.conf \
    --experiment-name TRAIN-TEST-RECORD-GRAD-NORMS \
    --log-dir TRAIN-TEST-RECORD-GRAD-NORMS \
    --privacy \
    --use-steps \
    --overwrite-experiment \
    --clipping-mode flat
