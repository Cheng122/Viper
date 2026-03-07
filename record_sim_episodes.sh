#!/bin/bash

# Data recording script for Viper imitation learning
# Usage: bash record_sim_episodes.sh <task_name> <num_episodes>

TASK_NAME=${1:-"sim_transfer_cube_scripted"}
NUM_EPISODES=${2:-200}
DATASET_DIR="./data/${TASK_NAME}"

echo "Recording ${NUM_EPISODES} episodes for task: ${TASK_NAME}"
echo "Dataset directory: ${DATASET_DIR}"

python dataset_utils/record_sim_episodes.py \
    --task_name ${TASK_NAME} \
    --dataset_dir ${DATASET_DIR} \
    --num_episodes ${NUM_EPISODES} \
    --onscreen_render
