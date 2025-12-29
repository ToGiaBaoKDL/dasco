#!/bin/bash
# Evaluate MABSA (Joint MATE + MASC)

export CUDA_VISIBLE_DEVICES="0"

# Find best models
MATE_MODEL=$(ls -t ./checkpoints/MATE_custom/best*.pt 2>/dev/null | head -1)
MASC_MODEL=$(ls -t ./checkpoints/MASC_custom/best*.pt 2>/dev/null | head -1)

if [ -z "$MATE_MODEL" ]; then
    echo "Error: No MATE model found in ./checkpoints/MATE_custom/"
    echo "Please train MATE first: bash train_mate_custom.sh"
    exit 1
fi

if [ -z "$MASC_MODEL" ]; then
    echo "Error: No MASC model found in ./checkpoints/MASC_custom/"
    echo "Please train MASC first: bash train_masc_custom.sh"
    exit 1
fi

echo "=== Evaluating MABSA ==="
echo "MATE Model: $MATE_MODEL"
echo "MASC Model: $MASC_MODEL"
echo "Test Data: ./finetune_dataset/custom/test"

python eval_tools.py \
    --MATE_model "$MATE_MODEL" \
    --MASC_model "$MASC_MODEL" \
    --test_ds ./finetune_dataset/custom/test \
    --task MABSA \
    --gcn_layers 5 \
    --device cuda:0

echo "=== Evaluation Complete ==="
