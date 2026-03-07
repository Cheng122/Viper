#!/bin/bash

# Training and evaluation script for Viper imitation learning
# Usage: bash train_eval.sh <task_name>

TASK_NAME=${1:-"sim_transfer_cube_scripted"}
POLICY="Viper"
CKPT_DIR="./ckpt/${TASK_NAME}/${POLICY}"
NUM_EPOCHS=2500
BATCH_SIZE=16

echo "Training on task: ${TASK_NAME}"
echo "Checkpoint directory: ${CKPT_DIR}"

# Training
python imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir ${CKPT_DIR} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE}

# Evaluation
echo "Evaluating trained model..."
python imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir ${CKPT_DIR} \
    --eval
