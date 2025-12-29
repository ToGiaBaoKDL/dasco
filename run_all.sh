#!/bin/bash
# DASCO Training Pipeline - Run All Steps
# Usage: bash run_all.sh

set -e  # Exit on error

echo "=========================================="
echo "DASCO MABSA Training Pipeline"
echo "=========================================="

# Step 1: Check prerequisites
echo ""
echo "[Step 1/6] Checking prerequisites..."

if [ ! -d "./Text_encoder/model_best" ]; then
    echo "ERROR: Text_encoder/model_best not found!"
    echo "Please download from Google Drive first."
    exit 1
fi

if [ ! -f "./checkpoints/MATE_pretrain/mate.pt" ]; then
    echo "ERROR: checkpoints/MATE_pretrain/mate.pt not found!"
    echo "Please download from Google Drive first."
    exit 1
fi

if [ ! -f "./checkpoints/MASC_pretrain/masc.pt" ]; then
    echo "ERROR: checkpoints/MASC_pretrain/masc.pt not found!"
    echo "Please download from Google Drive first."
    exit 1
fi

if [ ! -f "./data/text_image_dataset.json" ]; then
    echo "ERROR: data/text_image_dataset.json not found!"
    exit 1
fi

echo "All prerequisites OK!"

# Step 2: Cache image features
echo ""
echo "[Step 2/6] Caching image features (this may take a while)..."

if [ -f "./image_cache/features.pkl" ]; then
    echo "Image cache already exists. Skipping..."
else
    python cache_images.py \
        --input ./data/text_image_dataset.json \
        --output ./image_cache/features.pkl
fi

# Step 3: Convert data to pkl format
echo ""
echo "[Step 3/6] Converting data to pkl format..."

if [ -d "./finetune_dataset/custom/train" ]; then
    echo "Dataset already converted. Skipping..."
else
    python convert_data.py \
        --input ./data/text_image_dataset.json \
        --image_cache ./image_cache/features.pkl \
        --output ./finetune_dataset/custom \
        --train_ratio 0.8 \
        --val_ratio 0.1 \
        --test_ratio 0.1
fi

# Step 4: Train MATE
echo ""
echo "[Step 4/6] Training MATE model..."
echo "This will take several hours depending on your GPU."

if [ -f "./checkpoints/MATE_custom/best*.pt" ] 2>/dev/null; then
    echo "MATE model already exists. Skipping..."
else
    bash train_mate_custom.sh
fi

# Step 5: Train MASC
echo ""
echo "[Step 5/6] Training MASC model..."
echo "This will take several hours depending on your GPU."

if [ -f "./checkpoints/MASC_custom/best*.pt" ] 2>/dev/null; then
    echo "MASC model already exists. Skipping..."
else
    bash train_masc_custom.sh
fi

# Step 6: Evaluate MABSA
echo ""
echo "[Step 6/6] Evaluating MABSA..."
bash eval_mabsa_custom.sh

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "To run inference on new data:"
echo "  python inference.py \\"
echo "    --mate_model ./checkpoints/MATE_custom/best_f1_xxx.pt \\"
echo "    --masc_model ./checkpoints/MASC_custom/best_f1_xxx.pt \\"
echo "    --input ./finetune_dataset/custom/test \\"
echo "    --output results.json"
