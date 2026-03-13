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
    return parser.parse_args()


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
    from losses import total_loss

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

        # Forward pass
        output = model(images, char_indices)

        # Compute loss (with optional render skipping for speed)
        if args.render_every > 1 and step % args.render_every != 0:
            # Lightweight loss: existence only (skip expensive rendering)
            from losses import existence_loss
            loss = existence_loss(output['existence'])
            loss_dict = {'total': loss.item(), 'existence': loss.item()}
        else:
            loss, loss_dict = total_loss(output, glyph_masks, weights=loss_weights)

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

    # Set up pydiffvg
    import pydiffvg
    pydiffvg.set_use_gpu(device.type == 'cuda')

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
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
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

    # Training loop
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

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pt'),
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
