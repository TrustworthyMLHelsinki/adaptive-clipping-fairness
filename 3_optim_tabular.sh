#!/bin/bash
MASTER_PORT=30000 \
    MASTER_ADDR=localhost \
    WORLD_SIZE=1 \
    LOCAL_RANK=0 \
    RANK=0 \
    python run.py optimize \
    --num-workers 0 \
    --model-name logistic \
    --dataset-name dutch \
    --dataset-label-field label \
    --target-hypers learning_rate \
    --target-hypers batch_size \
    --max-grad-norm 1.0 \
    --epochs 40 \
    --target-epsilon 0.5 \
    --n-trials 20 \
    --seed 42 \
    --physical-batch-size 30000 \
    --optuna-config ./conf/adapt_random.conf \
    --experiment-name optim_tabular \
    --log-dir optim_tabular \
    --privacy \
    --use-steps \
    --normalize-clipping \
    --clipping-mode flat \
    --protected-feature gender \
    --dataset-label-field occupation
