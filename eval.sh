#!/bin/bash

# Evaluation script for Viper imitation learning
# Usage: bash eval.sh <task_name>

TASK_NAME=${1:-"sim_transfer_cube_scripted"}
POLICY="Viper"
CKPT_DIR="./ckpt/${TASK_NAME}/${POLICY}"

python imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir ${CKPT_DIR} \
    --eval \
    --onscreen_render
