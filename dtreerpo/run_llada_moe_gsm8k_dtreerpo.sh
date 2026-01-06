#!/bin/bash
export LOGDIR=checkpoints
export WANDB_PROJECT=dtreerpo-gsm8k
mkdir -p $LOGDIR

DATASET="gsm8k"
RUN_NAME=llada_moe_${DATASET}_dtreerpo
MODEL_PATH=/path/to/LLaDA-MoE-7B-A1B-Instruct

accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 29503 dtreerpo_train.py \
    --config slurm_scripts/dtreerpo.yaml \
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --run_name $RUN_NAME \
    --output_dir checkpoints/llada_moe/$RUN_NAME \
    --mask_id 156895
