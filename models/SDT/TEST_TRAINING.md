# SDT Training Test Guide

## Quick Test on MacBook (10-15 minutes)

This will verify training works before doing the full 1-2 month run on your Linux machine.

### Step 1: Run Test Training

```bash
cd /Users/justinvail/glossy/models/SDT
source venv/bin/activate
python train_macos.py --cfg configs/English_TEST.yml --log test_run
```

### Step 2: What to Expect

**Initial output:**
```
✓ Using MPS (Apple Silicon GPU)  # or CPU if no GPU
Number of training samples: [number]
Model parameters: 67.8M
✓ Loaded content pretrained model from model_zoo/...
================================
TRAINING CONFIGURATION
================================
Device: mps
Dataset: ENGLISH
Batch size: 8
Learning rate: 0.0002
Max iterations: 500
Save every: 100 iterations
Validate every: 100 iterations
================================

Training...
```

**During training (watch for these):**
- Iteration progress: `Iter: 1/500`
- Loss values: `NCE Loss: X.XX, PEN Loss: X.XX`
- Checkpoints saved: Every 100 iterations in `saved/models/`
- Validation: Every 100 iterations
- Time per iteration: ~1-5 seconds (MacBook) vs ~0.1-0.5 sec (GTX 1660)

**Expected timeline:**
- 500 iterations × 3 sec/iter = ~25 minutes on MacBook
- Same on GTX 1660: ~4 minutes

### Step 3: Check Results

**Checkpoints created:**
```bash
ls -lh saved/models/English_CASIA/
```

You should see:
```
checkpoint-iter100.pth
checkpoint-iter200.pth
checkpoint-iter300.pth
checkpoint-iter400.pth
checkpoint-iter500.pth
```

**Generated samples:**
```bash
ls saved/samples/English_CASIA/
```

Sample images showing training progress at each validation step.

### Step 4: What Good Results Look Like

**Loss trends (check log):**
```bash
cat saved/log/test_run.log | grep "Loss"
```

**Expected:**
- NCE Loss: Should decrease (e.g., 5.0 → 3.0 → 2.0)
- PEN Loss: Should decrease (e.g., 1.5 → 1.0 → 0.8)
- **If losses are NaN or exploding** → Problem with setup

**Sample quality at iter 500:**
- **Won't be legible yet** (only 500 iters vs 200k needed)
- Should see stroke patterns forming
- **Not random noise** - that's the key test

### Example Good vs Bad

**✓ GOOD (test passed):**
```
Iter 100: NCE Loss: 4.23, PEN Loss: 1.45
Iter 200: NCE Loss: 3.87, PEN Loss: 1.21
Iter 300: NCE Loss: 3.54, PEN Loss: 1.05
Iter 400: NCE Loss: 3.28, PEN Loss: 0.92
Iter 500: NCE Loss: 3.05, PEN Loss: 0.84
```
Losses decreasing steadily → Training works!

**✗ BAD (problem):**
```
Iter 100: NCE Loss: NaN, PEN Loss: NaN
```
or
```
Iter 100: NCE Loss: 15.2, PEN Loss: 8.7
Iter 200: NCE Loss: 23.4, PEN Loss: 12.3
```
Losses exploding → Check setup

---

## Full Training on Linux GTX 1660 Super

Once test passes, transfer to Linux machine:

### Step 1: Prepare Linux Machine

```bash
# On Linux machine
cd ~/
git clone https://github.com/justinmvail/glossy.git
cd glossy/models/SDT

# Create venv
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-macos.txt
```

### Step 2: Download Models & Data

```bash
# Download pretrained content encoder
bash download_models.sh

# Download English dataset
bash download_english_data.sh
```

### Step 3: Start Full Training

```bash
# Use tmux or screen to keep running if SSH disconnects
tmux new -s sdt_training

# Start training (uses CUDA automatically)
python train_macos.py --cfg configs/English_CASIA.yml --log english_full

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t sdt_training
```

### Step 4: Monitor Progress

**Check log:**
```bash
tail -f saved/log/english_full.log
```

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**Expected full training:**
- **Iterations:** 200,000
- **Time:** 30-50 days (24/7)
- **Checkpoints:** saved/models/English_CASIA/checkpoint-iterXXXX.pth
- **Best checkpoint:** After ~150k-200k iterations

### Step 5: Estimate Time Remaining

```python
# Check current iteration from log
current_iter = 5000  # example
total_iters = 200000
time_per_iter = 2.5  # seconds (measure from log)

remaining = (total_iters - current_iter) * time_per_iter
days = remaining / 86400
print(f"Estimated time remaining: {days:.1f} days")
```

### Step 6: Test Checkpoints During Training

Every ~20k iterations, test the checkpoint:

```bash
python test.py \
  --cfg configs/English_CASIA.yml \
  --pretrained_model saved/models/English_CASIA/checkpoint-iter20000.pth \
  --store_type online \
  --sample_size 20 \
  --dir test_iter20k
```

Check generated samples to see quality improving over time.

---

## What Full Training Results Should Look Like

### At 20k iterations:
- Strokes starting to form letters
- Still messy, but recognizable shapes

### At 50k iterations:
- Individual letters becoming clearer
- Some words might be readable

### At 100k iterations:
- Most letters legible
- Consistent style

### At 150k-200k iterations:
- **Production quality**
- Smooth, legible handwriting
- Proper capitalization (A vs a)
- Ready for GLOSSY

---

## Troubleshooting

### VRAM Issues (OOM Error)
```bash
# Reduce batch size in config
IMS_PER_BATCH: 32  # down from 64
```

### Training Stalls
```bash
# Check if process is running
ps aux | grep train

# Check GPU
nvidia-smi
```

### Loss Not Decreasing After 10k Iters
- May need to adjust learning rate
- Check if data loaded correctly

---

## Cost & Time Summary

| Machine | Time | Cost | Notes |
|---------|------|------|-------|
| **MacBook (test)** | 25 min | $0 | Verify setup only |
| **GTX 1660 Super** | 30-50 days | ~$15-25 electricity | Free, but slow |
| **Lambda Labs A100** | 3-7 days | $110-165 | Fast, paid |
| **Google Colab Pro** | 3-7 days | $10/month | May hit time limits |

**Recommendation for you:** Start test on MacBook now, then run full training on GTX 1660 Super overnight for 1-2 months.
