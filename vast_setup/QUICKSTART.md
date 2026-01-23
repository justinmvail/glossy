# Vast.ai One-DM Training - Quick Start

## What's Been Optimized (Tested on RTX 4090)

| Change | GTX 1660 (6GB) | RTX 4090 (24GB) |
|--------|----------------|-----------------|
| `batch_size` | 2 | **128** (optimal) |
| `num_workers` | 0 (debug) | **16** (fast loading) |
| `pin_memory` | False | True (fast transfer) |
| `optimizer` | SGD | AdamW |
| Gradient checkpointing | Yes | Yes |
| Gradient clipping | Yes | Yes (1.0) |

### Batch Size Testing Results

| Batch Size | Speed | GPU Util | VRAM | Result |
|------------|-------|----------|------|--------|
| 8 | - | 7-13% | - | Too slow |
| 32 | - | ~23% | - | Still slow |
| 64 | ~15 min/epoch | - | - | Better |
| **128** | **~7 min/epoch** | **63%** | **18GB** | **Optimal** |
| 160 | ~8.5 min/epoch | 59-100% | 21GB | Slower iterations |
| 192 | - | - | OOM | Out of memory |

## Files to Upload

| File | Size | Description |
|------|------|-------------|
| `onedm_data.tar.gz` | 246 MB | IAM64 dataset |
| `train_4090.py` | 6 KB | Optimized training script |
| `trainer_4090.py` | 9 KB | Trainer with gradient clipping |
| `IAM64_4090.yml` | 1 KB | RTX 4090 config (batch=128) |
| `setup_vast.sh` | 3 KB | Manual setup script |
| `onstart.sh` | 2 KB | Automatic on-start script |

**Total upload: ~246 MB**

## Google Drive Setup

Upload all files to a Google Drive folder and make it publicly accessible:
- Folder: `https://drive.google.com/drive/folders/1UY61ytrE6ec-OBdMESZvcpD9gcVsz_ad`

## Training Timeline (RTX 4090, batch=128)

| Milestone | Time | Cost (~$0.40/hr) |
|-----------|------|------------------|
| Setup | ~3 min | ~$0.02 |
| 10 epochs | ~1.2 hr | ~$0.50 |
| 50 epochs | ~6 hr | ~$2.50 |
| 100 epochs | ~12 hr | **~$5** |

## Config Details (`IAM64_4090.yml`)

```yaml
SOLVER:
  BASE_LR: 0.0001
  EPOCHS: 100
  TYPE: AdamW
  GRAD_L2_CLIP: 1.0

TRAIN:
  IMS_PER_BATCH: 128      # Optimal for RTX 4090
  SNAPSHOT_ITERS: 10      # Save every 10 epochs

DATA_LOADER:
  NUM_THREADS: 16         # Parallel data loading
```

## Usage

### Automatic (On-Start Script)

Paste contents of `onstart.sh` into Vast.ai "On-start Script" field. Training starts automatically.

### Manual

```bash
# 1. Run setup
bash /workspace/vast_files/setup_vast.sh

# 2. Start training
cd /workspace/One-DM
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run --nproc_per_node=1 -- train.py --cfg configs/IAM64_4090.yml --log run1
```

## Checkpoint Schedule

Saves to `/workspace/One-DM/Saved/IAM64_4090/run1-*/model/`:
- `9-ckpt.pt` (after epoch 10)
- `19-ckpt.pt` (after epoch 20)
- `29-ckpt.pt` (after epoch 30)
- ... etc

## Download Results

```bash
# From your local machine
scp -P <port> 'root@<ip>:/workspace/One-DM/Saved/IAM64_4090/*/model/*.pt' ./checkpoints/
```

Or use the auto-upload feature in `onstart.sh` which uploads to transfer.sh when training completes.
