#!/bin/bash
# SDT English-only training script
# Works on both macOS (test) and Linux (full training)

set -e

echo "=========================================="
echo "SDT English Training Setup"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# Check GPU availability
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    device = "cuda"
elif torch.backends.mps.is_available():
    print("✓ MPS available (Apple Silicon GPU)")
    device = "mps"
else:
    print("⚠ No GPU available, using CPU (will be slow)")
    device = "cpu"

print(f"\nUsing device: {device}")
EOF

echo ""
echo "Starting training..."
echo "Config: configs/English_CASIA.yml"
echo "Content encoder: model_zoo/position_layer2_dim512_iter138k_test_acc0.9443.pth"
echo ""

# Start training
python train.py \
    --cfg configs/English_CASIA.yml \
    --content_pretrained model_zoo/position_layer2_dim512_iter138k_test_acc0.9443.pth \
    --log english_train

echo ""
echo "Training started!"
echo "Monitor progress: tail -f saved/log/english_train.log"
