"""Profile a single training step to find the bottleneck."""

import time
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True)
    parser.add_argument('--font-dir', default='/')
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)

    import pydiffvg
    pydiffvg.set_use_gpu(device.type == 'cuda')

    from model import StrokePredictor
    from dataset import GlyphDataset, collate_with_masks
    from losses import total_loss, existence_loss
    from torch.utils.data import DataLoader

    model = StrokePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = GlyphDataset(
        db_path=args.db, font_dir=args.font_dir,
        cache_dir=args.cache_dir,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_with_masks,
        pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    logger.info("Warming up DataLoader...")
    data_iter = iter(dataloader)
    _ = next(data_iter)  # warm up workers
    logger.info("Warm-up done. Profiling %d steps...", args.steps)

    t_data_total = 0
    t_forward_total = 0
    t_loss_render_total = 0
    t_loss_light_total = 0
    t_backward_total = 0

    t_batch_start = time.perf_counter()

    for step in range(args.steps):
        # Data loading
        t0 = time.perf_counter()
        images, char_indices, glyph_masks = next(data_iter)
        t1 = time.perf_counter()
        t_data = t1 - t0

        images = images.to(device)
        char_indices = char_indices.to(device)
        glyph_masks = glyph_masks.to(device)
        torch.cuda.synchronize()

        # Forward pass
        t2 = time.perf_counter()
        optimizer.zero_grad()
        output = model(images, char_indices)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        t_forward = t3 - t2

        # Loss (render)
        t4 = time.perf_counter()
        loss, loss_dict = total_loss(output, glyph_masks)
        torch.cuda.synchronize()
        t5 = time.perf_counter()
        t_loss_render = t5 - t4

        # Loss (lightweight - existence only)
        t6 = time.perf_counter()
        loss_light = existence_loss(output['existence'])
        torch.cuda.synchronize()
        t7 = time.perf_counter()
        t_loss_light = t7 - t6

        # Backward
        t8 = time.perf_counter()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t9 = time.perf_counter()
        t_backward = t9 - t8

        t_data_total += t_data
        t_forward_total += t_forward
        t_loss_render_total += t_loss_render
        t_loss_light_total += t_loss_light
        t_backward_total += t_backward

        logger.info(
            "Step %d | data=%.3fs fwd=%.3fs loss_render=%.3fs loss_light=%.4fs bwd=%.3fs | total=%.3fs",
            step, t_data, t_forward, t_loss_render, t_loss_light, t_backward,
            t_data + t_forward + t_loss_render + t_backward,
        )

    total_time = time.perf_counter() - t_batch_start
    logger.info("=" * 60)
    logger.info("TOTALS over %d steps (%.1fs):", args.steps, total_time)
    logger.info("  Data loading:   %.3fs (%.0f%%)", t_data_total, 100 * t_data_total / total_time)
    logger.info("  Forward pass:   %.3fs (%.0f%%)", t_forward_total, 100 * t_forward_total / total_time)
    logger.info("  Loss (render):  %.3fs (%.0f%%)", t_loss_render_total, 100 * t_loss_render_total / total_time)
    logger.info("  Loss (light):   %.3fs (%.0f%%)", t_loss_light_total, 100 * t_loss_light_total / total_time)
    logger.info("  Backward+optim: %.3fs (%.0f%%)", t_backward_total, 100 * t_backward_total / total_time)


if __name__ == '__main__':
    main()
