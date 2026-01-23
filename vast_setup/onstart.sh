#!/bin/bash
# Vast.ai On-Start Script - Fully Automated One-DM Training with Auto-Upload
# Paste this into "On-start Script" field

cd /workspace

# Install gdown
pip install --quiet gdown

# Download files from Google Drive
gdown --folder https://drive.google.com/drive/folders/1UY61ytrE6ec-OBdMESZvcpD9gcVsz_ad -O /workspace/vast_files --quiet

# Clone One-DM
git clone --quiet https://github.com/dailenson/One-DM.git

# Extract dataset
tar -xzf /workspace/vast_files/onedm_data.tar.gz -C One-DM/

# Copy optimized files
cp /workspace/vast_files/train_4090.py One-DM/train.py
cp /workspace/vast_files/trainer_4090.py One-DM/trainer/trainer.py
cp /workspace/vast_files/IAM64_4090.yml One-DM/configs/

# Install dependencies
pip install --quiet diffusers transformers accelerate tqdm opencv-python matplotlib einops easydict torchvision lmdb tensorboardX tensorboard scipy omegaconf

# Create training wrapper script that uploads when done
cat > /workspace/train_and_upload.sh << 'TRAINSCRIPT'
#!/bin/bash
cd /workspace/One-DM

echo "$(date): Training started" >> /workspace/training.log

# Run training
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run --nproc_per_node=1 -- train.py --cfg configs/IAM64_4090.yml --log run1 2>&1 | tee -a /workspace/training.log

echo "$(date): Training finished" >> /workspace/training.log

# Find and upload checkpoints
echo "$(date): Uploading checkpoints..." >> /workspace/training.log
cd /workspace/One-DM/Saved
CHECKPOINT_DIR=$(find . -type d -name "run1-*" | head -1)

if [ -n "$CHECKPOINT_DIR" ]; then
    # Zip all checkpoints
    zip -r /workspace/checkpoints.zip "$CHECKPOINT_DIR/model/"

    # Upload to transfer.sh (free, keeps files 14 days)
    UPLOAD_URL=$(curl --upload-file /workspace/checkpoints.zip https://transfer.sh/checkpoints.zip)

    echo "============================================" >> /workspace/training.log
    echo "TRAINING COMPLETE!" >> /workspace/training.log
    echo "Download checkpoints from:" >> /workspace/training.log
    echo "$UPLOAD_URL" >> /workspace/training.log
    echo "============================================" >> /workspace/training.log

    # Also save URL to a separate file for easy access
    echo "$UPLOAD_URL" > /workspace/DOWNLOAD_URL.txt
else
    echo "ERROR: No checkpoint directory found" >> /workspace/training.log
fi
TRAINSCRIPT

chmod +x /workspace/train_and_upload.sh

# Start training in background
nohup /workspace/train_and_upload.sh > /dev/null 2>&1 &

echo "TRAINING STARTED - Monitor: tail -f /workspace/training.log"
echo "When done, download URL will be in: /workspace/DOWNLOAD_URL.txt"
