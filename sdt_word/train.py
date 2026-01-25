"""
Training script for Word-level SDT.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from pathlib import Path

from models.word_model import WordSDT
from dataset import get_dataloader


class GMMStrokeLoss(nn.Module):
    """
    Gaussian Mixture Model loss for stroke prediction.
    Predicts (dx, dy) with GMM and pen state with cross-entropy.
    """
    def __init__(self, n_mixtures=20):
        super().__init__()
        self.n_mixtures = n_mixtures

    def forward(self, pred, target, mask):
        """
        pred: [B, T, 123] - GMM parameters
        target: [B, T, 5] - (dx, dy, p1, p2, p3)
        mask: [B, T] - valid positions

        Returns: scalar loss
        """
        B, T, _ = pred.shape
        M = self.n_mixtures

        # Parse predictions
        pi = pred[:, :, :M]  # mixture weights (logits)
        mu_x = pred[:, :, M:2*M]
        mu_y = pred[:, :, 2*M:3*M]
        log_sigma_x = pred[:, :, 3*M:4*M]
        log_sigma_y = pred[:, :, 4*M:5*M]
        rho = pred[:, :, 5*M:6*M]
        pen_logits = pred[:, :, 6*M:]  # [B, T, 3]

        # Clamp for numerical stability
        log_sigma_x = torch.clamp(log_sigma_x, min=-10, max=10)
        log_sigma_y = torch.clamp(log_sigma_y, min=-10, max=10)
        sigma_x = torch.exp(log_sigma_x)
        sigma_y = torch.exp(log_sigma_y)
        rho = torch.tanh(rho)
        rho = torch.clamp(rho, min=-0.999, max=0.999)

        # Target coordinates
        dx = target[:, :, 0:1]  # [B, T, 1]
        dy = target[:, :, 1:2]  # [B, T, 1]

        # Compute bivariate Gaussian log probability for each mixture
        # p(x,y) = exp(-z / (2(1-rho^2))) / (2*pi*sigma_x*sigma_y*sqrt(1-rho^2))
        # where z = ((x-mu_x)/sigma_x)^2 + ((y-mu_y)/sigma_y)^2 - 2*rho*(x-mu_x)*(y-mu_y)/(sigma_x*sigma_y)

        norm_x = (dx - mu_x) / sigma_x  # [B, T, M]
        norm_y = (dy - mu_y) / sigma_y  # [B, T, M]

        one_minus_rho_sq = 1 - rho ** 2
        z = norm_x ** 2 + norm_y ** 2 - 2 * rho * norm_x * norm_y
        z = z / (one_minus_rho_sq + 1e-8)

        # Log probability
        log_norm = -0.5 * z - torch.log(2 * np.pi * sigma_x * sigma_y * torch.sqrt(one_minus_rho_sq + 1e-8))

        # Log-sum-exp over mixtures with mixture weights
        log_pi = F.log_softmax(pi, dim=-1)  # [B, T, M]
        gmm_log_prob = torch.logsumexp(log_pi + log_norm, dim=-1)  # [B, T]

        # GMM loss (negative log likelihood)
        gmm_loss = -gmm_log_prob * mask
        gmm_loss = gmm_loss.sum() / (mask.sum() + 1e-8)

        # Pen state loss (cross-entropy)
        pen_target = target[:, :, 2:].argmax(dim=-1)  # [B, T] - convert one-hot to class
        pen_loss = F.cross_entropy(
            pen_logits.view(-1, 3),
            pen_target.view(-1),
            reduction='none'
        ).view(B, T)
        pen_loss = (pen_loss * mask).sum() / (mask.sum() + 1e-8)

        return gmm_loss + pen_loss, gmm_loss, pen_loss


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    total_gmm = 0
    total_pen = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        content_img = batch['content_img'].to(device)
        style_imgs = batch['style_imgs'].to(device)
        strokes = batch['strokes'].to(device)
        mask = batch['stroke_mask'].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred = model(content_img, style_imgs, strokes)

        # Compute loss
        loss, gmm_loss, pen_loss = criterion(pred, strokes, mask)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_gmm += gmm_loss.item()
        total_pen += pen_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'gmm': f'{gmm_loss.item():.4f}',
            'pen': f'{pen_loss.item():.4f}'
        })

    return {
        'loss': total_loss / num_batches,
        'gmm_loss': total_gmm / num_batches,
        'pen_loss': total_pen / num_batches,
    }


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_gmm = 0
    total_pen = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            content_img = batch['content_img'].to(device)
            style_imgs = batch['style_imgs'].to(device)
            strokes = batch['strokes'].to(device)
            mask = batch['stroke_mask'].to(device)

            pred = model(content_img, style_imgs, strokes)
            loss, gmm_loss, pen_loss = criterion(pred, strokes, mask)

            total_loss += loss.item()
            total_gmm += gmm_loss.item()
            total_pen += pen_loss.item()
            num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'gmm_loss': total_gmm / num_batches,
        'pen_loss': total_pen / num_batches,
    }


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    train_loader = get_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        num_style_samples=args.num_style_samples,
        max_stroke_len=args.max_stroke_len,
    )

    # For validation, use same data but no augmentation
    val_loader = get_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        num_style_samples=args.num_style_samples,
        max_stroke_len=args.max_stroke_len,
        augment=False,
    )

    # Create model
    model = WordSDT(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        n_mixtures=args.n_mixtures,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Loss function
    criterion = GMMStrokeLoss(n_mixtures=args.n_mixtures)

    # Training loop
    best_val_loss = float('inf')
    history = []

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Log
        print(f"\nEpoch {epoch}:")
        print(f"  Train: loss={train_metrics['loss']:.4f}, gmm={train_metrics['gmm_loss']:.4f}, pen={train_metrics['pen_loss']:.4f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, gmm={val_metrics['gmm_loss']:.4f}, pen={val_metrics['pen_loss']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': scheduler.get_last_lr()[0],
        })

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }
        torch.save(checkpoint, output_dir / 'latest.pt')

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(checkpoint, output_dir / 'best.pt')
            print(f"  New best model saved!")

        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Generate samples periodically
        if (epoch + 1) % args.sample_every == 0:
            generate_samples(model, val_loader, device, output_dir / f'samples_epoch{epoch}')


def generate_samples(model, dataloader, device, output_dir):
    """Generate sample outputs for visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    batch = next(iter(dataloader))

    content_img = batch['content_img'][:4].to(device)
    style_imgs = batch['style_imgs'][:4].to(device)
    words = batch['word'][:4]

    with torch.no_grad():
        generated = model.generate(content_img, style_imgs, max_len=300)

    # Convert to numpy and save
    generated = generated.cpu().numpy()

    for i in range(len(generated)):
        # Save stroke sequence
        np.save(output_dir / f'{i}_{words[i]}_strokes.npy', generated[i])

        # Render to image
        img = strokes_to_image(generated[i])
        if img:
            img.save(output_dir / f'{i}_{words[i]}_rendered.png')


def strokes_to_image(strokes, height=128, line_width=2):
    """Render 5-dim strokes to image."""
    from PIL import Image, ImageDraw

    # Convert from (dx, dy, p1, p2, p3) to absolute coordinates
    abs_coords = []
    x, y = 0, 0
    for s in strokes:
        x += s[0]
        y += s[1]
        pen_state = np.argmax(s[2:5])  # 0=down, 1=up, 2=end
        abs_coords.append((x, y, pen_state))
        if pen_state == 2:  # End
            break

    if len(abs_coords) < 2:
        return None

    abs_coords = np.array(abs_coords)

    # Normalize to fit image
    min_x, min_y = abs_coords[:, 0].min(), abs_coords[:, 1].min()
    max_x, max_y = abs_coords[:, 0].max(), abs_coords[:, 1].max()
    range_x = max_x - min_x
    range_y = max_y - min_y

    if range_x == 0:
        range_x = 1
    if range_y == 0:
        range_y = 1

    # Scale to fit
    scale = (height - 20) / max(range_x, range_y)
    width = int(range_x * scale) + 40

    img = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(img)

    prev_x, prev_y = None, None
    for x, y, pen_state in abs_coords:
        curr_x = (x - min_x) * scale + 20
        curr_y = (y - min_y) * scale + 10

        if prev_x is not None and pen_state == 0:  # pen down
            draw.line([(prev_x, prev_y), (curr_x, curr_y)], fill=0, width=line_width)

        prev_x, prev_y = curr_x, curr_y

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Word-SDT model")
    parser.add_argument('--data_path', type=str, default='/home/server/glossy/sdt_word/data/train.lmdb')
    parser.add_argument('--output_dir', type=str, default='/home/server/glossy/sdt_word/checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    # Model parameters
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_mixtures', type=int, default=20)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)

    # Data parameters
    parser.add_argument('--num_style_samples', type=int, default=4)
    parser.add_argument('--max_stroke_len', type=int, default=500)

    # Misc
    parser.add_argument('--sample_every', type=int, default=10)

    args = parser.parse_args()
    main(args)
