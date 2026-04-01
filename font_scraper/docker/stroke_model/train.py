"""Training loop for the stroke prediction model.

Usage:
    # Inside Docker container:
    python3 train.py --db /data/fonts.db --font-dir /fonts --epochs 100

    # On host (with correct environment):
    python3 train.py --db fonts.db --font-dir fonts/ --epochs 10 --max-fonts 100

Config can also be passed via JSON file:
    python3 train.py --config train_config.json
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train stroke prediction model')
    parser.add_argument('--config', type=str, help='JSON config file (overrides CLI args)')
    parser.add_argument('--db', type=str, default='fonts.db', help='Path to fonts SQLite database')
    parser.add_argument('--font-dir', type=str, default='fonts/', help='Base font directory')
    parser.add_argument('--output-dir', type=str, default='checkpoints/', help='Checkpoint output directory')
    parser.add_argument('--cache-dir', type=str, default=None, help='Glyph image cache directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max-fonts', type=int, default=None, help='Limit number of fonts (for debugging)')
    parser.add_argument('--feature-dim', type=int, default=256, help='Feature dimension')
    parser.add_argument('--render-every', type=int, default=1, help='Compute render loss every N steps (save GPU)')
    parser.add_argument('--log-every', type=int, default=50, help='Log every N steps')
    parser.add_argument('--save-every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--loss-weights', type=str, default=None, help='JSON string of loss weights')
    parser.add_argument('--pretrain-epochs', type=int, default=0, help='Synthetic pretraining epochs before real data')
    parser.add_argument('--pretrain-samples', type=int, default=100000, help='Number of synthetic samples per pretrain epoch')
    return parser.parse_args()


def pretrain_epoch(model, dataloader, optimizer, device, epoch, args, writer=None):
    """Run one pretraining epoch on synthetic stroke data."""
    from losses import pretrain_loss

    model.train()
    epoch_loss = 0.0
    epoch_losses = {}
    n_steps = 0
    global_step = epoch * len(dataloader)

    for step, (images, char_indices, glyph_masks, gt_strokes) in enumerate(dataloader):
        images = images.to(device)
        char_indices = char_indices.to(device)
        glyph_masks = glyph_masks.to(device)

        optimizer.zero_grad()

        output = model(images, char_indices, glyph_masks)
        loss, loss_dict = pretrain_loss(output, gt_strokes)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss_dict['total']
        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v
        n_steps += 1

        if step % args.log_every == 0:
            logger.info(
                "Pretrain Epoch %d Step %d/%d | Loss: %.4f | %s",
                epoch, step, len(dataloader), loss_dict['total'],
                ' '.join(f"{k}={v:.4f}" for k, v in loss_dict.items() if k != 'total'),
            )

        if writer:
            writer.add_scalar('pretrain/loss', loss_dict['total'], global_step + step)

    avg_loss = epoch_loss / max(n_steps, 1)
    avg_losses = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}
    return avg_loss, avg_losses


def train_epoch(model, dataloader, optimizer, device, epoch, args, writer=None,
                loss_weights=None):
    """Run one training epoch.

    Args:
        model: StrokePredictor model.
        dataloader: DataLoader yielding (images, char_indices, glyph_masks).
        optimizer: PyTorch optimizer.
        device: torch.device.
        epoch: Current epoch number.
        args: Parsed arguments.
        writer: TensorBoard SummaryWriter (optional).
        loss_weights: Dict of loss weights.

    Returns:
        Average total loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0
    epoch_losses = {}
    n_steps = 0
    global_step = epoch * len(dataloader)

    for step, (images, char_indices, glyph_masks) in enumerate(dataloader):
        images = images.to(device)
        char_indices = char_indices.to(device)
        glyph_masks = glyph_masks.to(device)

        optimizer.zero_grad()

        # Forward pass (autoregressive model needs glyph_mask)
        output = model(images, char_indices, glyph_masks)

        # Compute loss
        from losses import autoregressive_loss
        loss, loss_dict = autoregressive_loss(
            output, weights=loss_weights, epoch=epoch, total_epochs=args.epochs,
        )

        loss.backward()

        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        epoch_loss += loss_dict['total']
        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v
        n_steps += 1

        # Logging
        if step % args.log_every == 0:
            logger.info(
                "Epoch %d Step %d/%d | Loss: %.4f | %s",
                epoch, step, len(dataloader), loss_dict['total'],
                ' '.join(f"{k}={v:.4f}" for k, v in loss_dict.items() if k != 'total'),
            )

        # TensorBoard
        if writer is not None:
            writer.add_scalar('train/loss_total', loss_dict['total'], global_step + step)
            for k, v in loss_dict.items():
                if k != 'total':
                    writer.add_scalar(f'train/loss_{k}', v, global_step + step)

    avg_loss = epoch_loss / max(n_steps, 1)
    avg_losses = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}
    return avg_loss, avg_losses


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    logger.info("Saved checkpoint: %s (epoch %d, loss %.4f)", path, epoch, loss)


def _build_tracking_samples(db_path, font_dir, device):
    """Build a fixed set of samples to render at each checkpoint."""
    import sqlite3
    from model import char_to_index, CANVAS_SIZE
    from PIL import ImageFont

    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT f.file_path, f.name FROM fonts f
        LEFT JOIN font_checks fc ON f.id = fc.font_id
        LEFT JOIN font_removals fr ON f.id = fr.font_id
        WHERE fr.font_id IS NULL
        AND (fc.prefilter_passed = 1 OR fc.prefilter_passed IS NULL)
        AND (fc.is_cursive = 0 OR fc.is_cursive IS NULL)
        LIMIT 100
    """).fetchall()
    conn.close()

    indices = [0, len(rows) // 2, len(rows) - 1]
    fonts = [rows[i] for i in indices if i < len(rows)]
    chars = ['A', 'g', 'R', '8']

    samples = []
    for fp_raw, font_name in fonts:
        fp = fp_raw if os.path.isabs(fp_raw) else os.path.join(font_dir, fp_raw)
        if not os.path.exists(fp):
            continue
        for char in chars:
            img = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), 255)
            draw = ImageDraw.Draw(img)
            rendered = False
            for font_size in range(200, 20, -5):
                try:
                    font = ImageFont.truetype(fp, font_size)
                except Exception:
                    continue
                bbox = font.getbbox(char)
                if bbox is None:
                    break
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if tw <= CANVAS_SIZE * 0.9 and th <= CANVAS_SIZE * 0.9:
                    x = (CANVAS_SIZE - tw) / 2 - bbox[0]
                    y = (CANVAS_SIZE - th) / 2 - bbox[1]
                    draw.text((x, y), char, fill=0, font=font)
                    rendered = True
                    break
            if not rendered:
                continue

            mask = np.array(img) < 128
            img_arr = 1.0 - mask.astype(np.float32)
            img_tensor = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0)
            char_idx = torch.tensor([char_to_index(char)], dtype=torch.long)
            safe_name = font_name.replace(' ', '_').replace('/', '_')[:20]
            samples.append({
                'img_tensor': img_tensor,
                'char_idx': char_idx,
                'mask': mask,
                'label': f"{safe_name}_{char}",
            })

    logger.info("Tracking %d samples across checkpoints", len(samples))
    return samples


def _render_tracking_samples(model, samples, device, output_dir, epoch, prefix='epoch'):
    """Render tracking samples and save to epoch-specific directory."""
    from model import CANVAS_SIZE

    epoch_dir = os.path.join(output_dir, 'tracking', f'{prefix}_{epoch:03d}')
    os.makedirs(epoch_dir, exist_ok=True)

    model.eval()
    colors = [(255, 0, 0), (0, 150, 0), (0, 0, 255), (255, 128, 0),
              (128, 0, 255), (0, 200, 200), (200, 0, 128), (128, 128, 0)]

    for sample in samples:
        img_t = sample['img_tensor'].to(device)
        char_t = sample['char_idx'].to(device)

        with torch.no_grad():
            strokes, stroke_widths = model.predict_strokes(
                img_t, char_t, CANVAS_SIZE, existence_threshold=0.3,
            )

        img = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
        mask = sample['mask']
        for y in range(CANVAS_SIZE):
            for x in range(CANVAS_SIZE):
                if mask[y, x]:
                    img.putpixel((x, y), (220, 220, 220))

        draw = ImageDraw.Draw(img)
        for si, stroke in enumerate(strokes):
            color = colors[si % len(colors)]
            sw = stroke_widths[si] if si < len(stroke_widths) else [2]
            for i in range(len(stroke) - 1):
                x1, y1 = stroke[i]
                x2, y2 = stroke[i + 1]
                # Per-segment width: average of endpoint widths
                if isinstance(sw, list) and len(sw) > i + 1:
                    w = max(1, int((sw[i] + sw[i + 1]) / 2))
                elif isinstance(sw, list) and len(sw) > 0:
                    w = max(1, int(sw[0]))
                else:
                    w = max(1, int(sw))
                draw.line([(x1, y1), (x2, y2)], fill=color, width=w)
            # Round caps at each control point to fill junction gaps
            # (matches Triton kernel's implicit round caps from t-clamping)
            for i, (px, py) in enumerate(stroke):
                if isinstance(sw, list) and len(sw) > i:
                    r = max(1, int(sw[i])) // 2
                else:
                    r = max(1, int(sw[0] if isinstance(sw, list) else sw)) // 2
                if r > 0:
                    draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=color)

        img.save(os.path.join(epoch_dir, f"{sample['label']}.png"))

    model.train()
    logger.info("Saved %d tracking samples to %s", len(samples), epoch_dir)


def main():
    args = parse_args()

    # Load config from JSON if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        for k, v in config.items():
            if hasattr(args, k.replace('-', '_')):
                setattr(args, k.replace('-', '_'), v)

    # Parse loss weights
    loss_weights = None
    if args.loss_weights:
        loss_weights = json.loads(args.loss_weights)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)
    if device.type == 'cuda':
        logger.info("GPU: %s, VRAM: %.1f GB",
                     torch.cuda.get_device_name(),
                     torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Set up pydiffvg (optional — Triton renderer is used for training)
    try:
        import pydiffvg
        pydiffvg.set_use_gpu(device.type == 'cuda')
    except Exception:
        logger.warning("pydiffvg not available, using Triton renderer only")

    # Create model
    from model import StrokePredictor
    model = StrokePredictor(feature_dim=args.feature_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.1fM", n_params / 1e6)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # Don't restore best_loss — loss functions may differ between runs
        # First epoch of new run will always save a checkpoint + tracking images
        logger.info("Resumed from %s (epoch %d)", args.resume, start_epoch)

    # Dataset
    from dataset import GlyphDataset, collate_with_masks
    dataset = GlyphDataset(
        db_path=args.db,
        font_dir=args.font_dir,
        max_fonts=args.max_fonts,
        cache_dir=args.cache_dir,
        augment=args.augment,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_with_masks,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, 'runs')
        writer = SummaryWriter(tb_dir)
        logger.info("TensorBoard logging to %s", tb_dir)
    except ImportError:
        logger.warning("TensorBoard not available, skipping logging")

    # Fixed tracking samples
    tracking_samples = _build_tracking_samples(args.db, args.font_dir, device)

    # Phase 1: Synthetic pretraining
    if args.pretrain_epochs > 0:
        from dataset import SyntheticStrokeDataset, collate_synthetic
        syn_dataset = SyntheticStrokeDataset(
            num_samples=args.pretrain_samples,
            canvas_size=model.encoder.feature_dim if hasattr(model.encoder, 'feature_dim') else 224,
        )
        # Fix: use CANVAS_SIZE not feature_dim
        syn_dataset = SyntheticStrokeDataset(num_samples=args.pretrain_samples)
        syn_loader = DataLoader(
            syn_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_synthetic,
            pin_memory=(device.type == 'cuda'),
            drop_last=True,
        )

        logger.info("Starting pretraining: %d epochs, %d synthetic samples",
                     args.pretrain_epochs, len(syn_dataset))

        for epoch in range(args.pretrain_epochs):
            t0 = time.time()
            avg_loss, avg_losses = pretrain_epoch(
                model, syn_loader, optimizer, device, epoch, args, writer=writer,
            )
            elapsed = time.time() - t0
            logger.info(
                "Pretrain %d/%d complete | Loss: %.4f | Time: %.1fs | %s",
                epoch, args.pretrain_epochs, avg_loss, elapsed,
                ' '.join(f"{k}={v:.4f}" for k, v in avg_losses.items() if k != 'total'),
            )
            # Save pretrain checkpoint and tracking
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    model, optimizer, epoch, avg_loss,
                    os.path.join(args.output_dir, 'best_model.pt'),
                )
                _render_tracking_samples(
                    model, tracking_samples, device, args.output_dir, epoch,
                    prefix='pretrain',
                )

        logger.info("Pretraining complete. Switching to real font training.")
        best_loss = float('inf')  # reset for real training

    # Phase 2: Real font training
    logger.info("Starting training: %d epochs, %d samples, batch_size=%d",
                args.epochs, len(dataset), args.batch_size)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        avg_loss, avg_losses = train_epoch(
            model, dataloader, optimizer, device, epoch, args,
            writer=writer, loss_weights=loss_weights,
        )

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d complete | Loss: %.4f | Time: %.1fs | %s",
            epoch, args.epochs, avg_loss, elapsed,
            ' '.join(f"{k}={v:.4f}" for k, v in avg_losses.items() if k != 'total'),
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                os.path.join(args.output_dir, 'best_model.pt'),
            )
            _render_tracking_samples(
                model, tracking_samples, device, args.output_dir, epoch,
            )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pt'),
            )
            _render_tracking_samples(
                model, tracking_samples, device, args.output_dir, epoch,
            )

        # TensorBoard epoch summary
        if writer:
            writer.add_scalar('epoch/loss', avg_loss, epoch)
            writer.add_scalar('epoch/time', elapsed, epoch)

    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs - 1, avg_loss,
        os.path.join(args.output_dir, 'final_model.pt'),
    )

    if writer:
        writer.close()

    logger.info("Training complete. Best loss: %.4f", best_loss)


if __name__ == '__main__':
    main()
