# DASCO Training Guide

## Cấu trúc thư mục

```
DASCO/
├── Text_encoder/model_best/      
├── checkpoints/
│   ├── MATE_pretrain/mate.pt     
│   └── MASC_pretrain/masc.pt     
├── finetune_dataset/custom/      ← Tạo sau Step 2
├── image_cache/features.pkl      ← Tạo sau Step 1
└── data/text_image_dataset.json  
```

---

## Step 1: Cache Image Features

Thời gian: ~30-60 phút

```bash
python cache_images.py \
    --input ./data/text_image_dataset.json \
    --output ./image_cache/features.pkl
```

**Output:** `./image_cache/features.pkl`

---

## Step 2: Convert Data

Thời gian: ~5 phút

```bash
python convert_data.py \
    --input ./data/text_image_dataset.json \
    --image_cache ./image_cache/features.pkl \
    --output ./finetune_dataset/custom \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

**Output:** 
- `./finetune_dataset/custom/train/`
- `./finetune_dataset/custom/dev/`
- `./finetune_dataset/custom/test/`

---

## Step 3: Train MATE (Aspect Extraction)

Thời gian: Vài giờ

```bash
bash train_mate_custom.sh
```

**Output:** `./checkpoints/MATE_custom/best_f1_xxx.pt`

**Monitor training:**
```bash
tensorboard --logdir ./checkpoints/MATE_custom/log_path
```

---

## Step 4: Train MASC (Sentiment Classification)

Thời gian: Vài giờ

```bash
bash train_masc_custom.sh
```

**Output:** `./checkpoints/MASC_custom/best_f1_xxx.pt`

---

## Step 5: Evaluate MABSA

```bash
bash eval_mabsa_custom.sh
```

**Output:** Precision, Recall, F1 scores

---

## Inference

### Trên test set

```bash
python inference.py \
    --mate_model ./checkpoints/MATE_custom/best_f1_xxx.pt \
    --masc_model ./checkpoints/MASC_custom/best_f1_xxx.pt \
    --input ./finetune_dataset/custom/test \
    --output results.json
```

### Trên JSON file mới

```bash
python inference.py \
    --mate_model ./checkpoints/MATE_custom/best_f1_xxx.pt \
    --masc_model ./checkpoints/MASC_custom/best_f1_xxx.pt \
    --input ./data/text_image_dataset.json \
    --image_cache ./image_cache/features.pkl \
    --output results.json
```

---

## Chạy toàn bộ tự động

```bash
bash run_all.sh
```
