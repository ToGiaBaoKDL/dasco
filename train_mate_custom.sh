#!/bin/bash
# Train MATE model for aspect term extraction
# Optimized for 80GB VRAM

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="0"

echo "=== Training MATE Model ==="
echo "Data: ./finetune_dataset/custom"
echo "Output: ./checkpoints/MATE_custom"

accelerate launch --config_file deepspeed_ddp.json MATE_finetune.py \
    --task MATE \
    --base_model ./Text_encoder/model_best \
    --pretrain_model ./checkpoints/MATE_pretrain/mate.pt \
    --train_ds ./finetune_dataset/custom/train \
    --eval_ds ./finetune_dataset/custom/dev \
    --hyper1 0.25 \
    --hyper2 0.25 \
    --hyper3 0.25 \
    --gcn_layers 5 \
    --lr 3e-5 \
    --seed 1000 \
    --itc 0 \
    --itm 0 \
    --lm 0 \
    --cl 1.0 \
    --save_path ./checkpoints/MATE_custom \
    --epoch 25 \
    --batch_size 32 \
    --accumulation_steps 1 \
    --val_step 30 \
    --save_step 200 \
    --log_step 1

echo "=== MATE Training Complete ==="
echo "Check ./checkpoints/MATE_custom for best model"
