#!/bin/bash
set -e

SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-./sd-scripts}"
ACCELERATE="accelerate launch"

for style in ink_wash ukiyoe cyberpunk baroque minimalist; do
    echo "=========================================="
    echo "Training LoRA for: ${style}"
    echo "=========================================="
    ${ACCELERATE} ${SD_SCRIPTS_DIR}/train_network.py \
        --config_file configs/lora_${style}.toml
    echo "Completed: ${style}"
done
echo "All LoRA training complete."
