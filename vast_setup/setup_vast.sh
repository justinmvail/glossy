#!/bin/bash
# Vast.ai One-DM Training Setup Script
# Optimized for RTX 4090

set -e  # Exit on error

echo "========================================="
echo "One-DM Vast.ai Setup (RTX 4090 Optimized)"
echo "========================================="

cd /workspace

# Find where the files are (could be /workspace or /workspace/vast_files)
if [ -f "onedm_data.tar.gz" ]; then
    FILES_DIR="/workspace"
elif [ -f "vast_files/onedm_data.tar.gz" ]; then
    FILES_DIR="/workspace/vast_files"
else
    echo "ERROR: Cannot find onedm_data.tar.gz"
    echo "Please ensure files are in /workspace or /workspace/vast_files"
    exit 1
fi

echo "Found files in: $FILES_DIR"

# 1. Clone fresh One-DM repo
echo "[1/6] Cloning One-DM repository..."
if [ -d "One-DM" ]; then
    echo "One-DM directory exists, removing..."
    rm -rf One-DM
fi
git clone --quiet https://github.com/dailenson/One-DM.git
echo "Repository cloned."

# 2. Extract dataset
echo "[2/6] Extracting dataset..."
tar -xzf "$FILES_DIR/onedm_data.tar.gz" -C One-DM/
echo "Dataset extracted."

# 3. Copy optimized files
echo "[3/6] Installing optimized training files..."
cp "$FILES_DIR/train_4090.py" One-DM/train.py
cp "$FILES_DIR/trainer_4090.py" One-DM/trainer/trainer.py
cp "$FILES_DIR/IAM64_4090.yml" One-DM/configs/
echo "Optimized files installed."

# 4. Install dependencies
echo "[4/6] Installing Python dependencies..."
pip install --quiet diffusers transformers accelerate tqdm opencv-python matplotlib einops easydict torchvision lmdb tensorboardX tensorboard scipy omegaconf

# Check PyTorch + CUDA
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 5. Verify GPU
echo "[5/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 6. Verify setup
echo "[6/6] Verifying configuration..."
cd One-DM
python3 -c "
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
cfg_from_file('configs/IAM64_4090.yml')
assert_and_infer_cfg()
print('Config loaded successfully:')
print(f'  Batch size:  {cfg.TRAIN.IMS_PER_BATCH}')
print(f'  Epochs:      {cfg.SOLVER.EPOCHS}')
print(f'  Optimizer:   {cfg.SOLVER.TYPE}')
print(f'  Num workers: {cfg.DATA_LOADER.NUM_THREADS}')
print(f'  Save every:  {cfg.TRAIN.SNAPSHOT_ITERS} epochs')
"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To start training:"
echo ""
echo "  cd /workspace/One-DM && CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run --nproc_per_node=1 -- train.py --cfg configs/IAM64_4090.yml --log run1"
echo ""
