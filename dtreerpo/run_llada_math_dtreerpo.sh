#!/bin/bash
export LOGDIR=checkpoints
export WANDB_PROJECT=dtreerpo-math
mkdir -p $LOGDIR

DATASET="math"
RUN_NAME=llada_${DATASET}_dtreerpo
MODEL_PATH=/path/to/LLaDA-8B-Instruct

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 29503 dtreerpo_train.py \
    --config slurm_scripts/dtreerpo.yaml \
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/llada/$RUN_NAME \
    --mask_id 126336
