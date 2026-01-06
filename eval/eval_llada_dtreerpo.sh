#!/bin/bash

# --- Configuration variables ---
GPU_IDS=(0 1 2 3 4 5 6 7)
MASTER_PORT=29411

BASE_MODEL_PATH="/path/to/LLaDA-8B-Instruct" 

# LoRA checkpoints (parent directory)
CHECKPOINT_BASE_DIR="../dtreerpo/checkpoints/llada/llada_sudoku_dtreerpo"
# ==========================================================

# Algorithm name
ALGORITHM="dtreerpo"

# Arrays of tasks and generation lengths
TASKS=("sudoku")
GEN_LENGTHS=(256 512)

START_CHECKPOINT=0

# --- GPU setup---
if [ $# -gt 0 ]; then
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi
GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"


# --- Main evaluation loop ---
CHECKPOINT_PATHS=$(find "$CHECKPOINT_BASE_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)

for lora_checkpoint_path in $CHECKPOINT_PATHS; do
  # Extract the checkpoint number
  checkpoint_name=$(basename "$lora_checkpoint_path")
  checkpoint_num=${checkpoint_name#checkpoint-}

  # Skip checkpoints before START_CHECKPOINT
  if [[ $checkpoint_num -lt $START_CHECKPOINT ]]; then
    echo "Skipping checkpoint-${checkpoint_num} (before ${START_CHECKPOINT})"
    continue
  fi

  for task in "${TASKS[@]}"; do
    for gen_length in "${GEN_LENGTHS[@]}"; do
      # Set batch size based on generation length
      if [ "$gen_length" -eq 512 ]; then
        batch_size=4
      else
        batch_size=8
      fi
      
      echo "======================================================================="
      echo "Evaluating LORA CHECKPOINT: $lora_checkpoint_path"
      echo "On BASE MODEL: $BASE_MODEL_PATH"
      echo "Running evaluation on TASK: $task with GEN_LENGTH=$gen_length, BATCH_SIZE=$batch_size"
      echo "======================================================================="

      # Define a unique output directory
      output_subdir="eval_results/${ALGORITHM}/${task}/${checkpoint_name}_gen${gen_length}"

      CUDA_VISIBLE_DEVICES=$GPU_LIST accelerate launch \
        --config_file accelerate_copy.yaml \
        --num_processes 8 \
        evaluate_dtreerpo.py \
        --model_path $BASE_MODEL_PATH \
        --checkpoint_path $lora_checkpoint_path \
        --dataset $task \
        --output_dir $output_subdir \
        --seed 42 \
        --batch_size 4 \
        --gen_length $gen_length \
        --block_length 32 \
        --temperature 0.0 \
        --mask_id 126336
    done
  done
done

echo "All evaluations completed!"
