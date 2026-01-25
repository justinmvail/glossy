"""
Dataset loader for word-level SDT training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import lmdb
import pickle
import numpy as np
from PIL import Image
import random


class WordStrokeDataset(Dataset):
    """Dataset for word-level stroke generation training."""

    def __init__(self, lmdb_path, num_style_samples=4, max_stroke_len=500,
                 content_height=64, augment=True):
        """
        Args:
            lmdb_path: Path to LMDB dataset
            num_style_samples: Number of style reference samples per batch item
            max_stroke_len: Maximum stroke sequence length
            content_height: Height of content images
            augment: Whether to apply data augmentation
        """
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.num_style_samples = num_style_samples
        self.max_stroke_len = max_stroke_len
        self.content_height = content_height
        self.augment = augment

        # Load keys and organize by writer
        with self.env.begin() as txn:
            self.length = int(txn.get(b'__len__').decode())
            self.keys = pickle.loads(txn.get(b'__keys__'))

        # Build writer index for style sampling
        self.writer_samples = {}  # writer_id -> list of sample indices
        self._build_writer_index()

    def _build_writer_index(self):
        """Build index of samples by writer for style sampling."""
        with self.env.begin() as txn:
            for idx, key in enumerate(self.keys):
                data = pickle.loads(txn.get(key.encode()))
                writer_id = data['writer_id']
                if writer_id not in self.writer_samples:
                    self.writer_samples[writer_id] = []
                self.writer_samples[writer_id].append(idx)

    def __len__(self):
        return self.length

    def _load_sample(self, idx):
        """Load a single sample from LMDB."""
        with self.env.begin() as txn:
            key = self.keys[idx]
            data = pickle.loads(txn.get(key.encode()))
        return data

    def _get_style_samples(self, writer_id, exclude_idx):
        """Get random style reference samples from the same writer."""
        available = [i for i in self.writer_samples[writer_id] if i != exclude_idx]
        if len(available) < self.num_style_samples:
            # If not enough samples, allow repeats
            selected = random.choices(available, k=self.num_style_samples)
        else:
            selected = random.sample(available, self.num_style_samples)

        style_samples = []
        for idx in selected:
            sample = self._load_sample(idx)
            style_samples.append(sample)

        return style_samples

    def _strokes_to_5dim(self, strokes):
        """
        Convert 3-dim strokes (dx, dy, pen_up) to 5-dim (dx, dy, p1, p2, p3).
        p1: pen down, p2: pen up, p3: end of sequence
        """
        n = len(strokes)
        result = np.zeros((n, 5), dtype=np.float32)
        result[:, :2] = strokes[:, :2]  # dx, dy

        for i in range(n):
            if strokes[i, 2] == 0:  # pen down
                result[i, 2] = 1  # p1
            else:  # pen up
                result[i, 3] = 1  # p2

        # Last point is end of sequence
        result[-1, 2:5] = [0, 0, 1]  # p3

        return result

    def _strokes_to_style_image(self, strokes, size=64):
        """Render strokes as a 64x64 image for style reference."""
        if strokes is None or len(strokes) == 0:
            return np.zeros((size, size), dtype=np.float32)

        # Convert relative to absolute
        abs_strokes = np.zeros_like(strokes[:, :2])
        abs_strokes[0] = strokes[0, :2]
        for i in range(1, len(strokes)):
            abs_strokes[i] = abs_strokes[i-1] + strokes[i, :2]

        # Get bounds and normalize
        min_xy = abs_strokes.min(axis=0)
        max_xy = abs_strokes.max(axis=0)
        range_xy = max_xy - min_xy
        range_xy[range_xy == 0] = 1

        # Scale to fit in image with padding
        padding = 4
        scale = (size - 2 * padding) / range_xy.max()
        abs_strokes = (abs_strokes - min_xy) * scale + padding

        # Render
        from PIL import Image, ImageDraw
        img = Image.new('L', (size, size), 255)
        draw = ImageDraw.Draw(img)

        prev_x, prev_y = None, None
        for i in range(len(abs_strokes)):
            x, y = abs_strokes[i]
            pen_up = strokes[i, 2] if strokes.shape[1] > 2 else 0

            if prev_x is not None and pen_up == 0:
                draw.line([(prev_x, prev_y), (x, y)], fill=0, width=2)

            prev_x, prev_y = x, y

        return np.array(img, dtype=np.float32) / 255.0

    def _augment_strokes(self, strokes):
        """Apply augmentation to stroke sequence."""
        if not self.augment:
            return strokes

        strokes = strokes.copy()

        # Scale augmentation (0.8 - 1.2)
        scale = random.uniform(0.8, 1.2)
        strokes[:, :2] *= scale

        # Rotation augmentation (-10 to 10 degrees)
        angle = random.uniform(-0.17, 0.17)  # radians
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = strokes[:, 0], strokes[:, 1]
        strokes[:, 0] = x * cos_a - y * sin_a
        strokes[:, 1] = x * sin_a + y * cos_a

        return strokes

    def __getitem__(self, idx):
        # Load main sample
        sample = self._load_sample(idx)
        writer_id = sample['writer_id']
        strokes = sample['strokes']  # (N, 3): dx, dy, pen_up
        content_img = sample['img']  # (H, W) grayscale

        # Augment strokes
        strokes = self._augment_strokes(strokes)

        # Convert to 5-dim
        strokes_5d = self._strokes_to_5dim(strokes)

        # Truncate or pad strokes
        if len(strokes_5d) > self.max_stroke_len:
            strokes_5d = strokes_5d[:self.max_stroke_len]
            strokes_5d[-1, 2:5] = [0, 0, 1]  # Mark end
        else:
            # Pad with end tokens
            pad_len = self.max_stroke_len - len(strokes_5d)
            padding = np.zeros((pad_len, 5), dtype=np.float32)
            padding[:, 4] = 1  # p3 (end)
            strokes_5d = np.vstack([strokes_5d, padding])

        # Create mask for valid positions
        stroke_mask = np.zeros(self.max_stroke_len, dtype=np.float32)
        stroke_mask[:len(sample['strokes'])] = 1

        # Get style reference samples
        style_samples = self._get_style_samples(writer_id, idx)
        style_imgs = []
        for s in style_samples:
            style_img = self._strokes_to_style_image(s['strokes'])
            style_imgs.append(style_img)
        style_imgs = np.stack(style_imgs)  # (N, 64, 64)

        # Normalize content image
        content_img = content_img.astype(np.float32) / 255.0

        return {
            'content_img': torch.from_numpy(content_img).unsqueeze(0),  # (1, H, W)
            'style_imgs': torch.from_numpy(style_imgs).unsqueeze(1),    # (N, 1, 64, 64)
            'strokes': torch.from_numpy(strokes_5d),                    # (T, 5)
            'stroke_mask': torch.from_numpy(stroke_mask),               # (T,)
            'writer_id': writer_id,
            'word': sample['word'],
        }


def collate_fn(batch):
    """Custom collate function to handle variable-width content images."""
    # Find max width in batch
    max_width = max(item['content_img'].shape[-1] for item in batch)

    # Pad content images to same width
    content_imgs = []
    for item in batch:
        img = item['content_img']
        pad_width = max_width - img.shape[-1]
        if pad_width > 0:
            # Pad with white (1.0)
            img = torch.nn.functional.pad(img, (0, pad_width), value=1.0)
        content_imgs.append(img)

    return {
        'content_img': torch.stack(content_imgs),
        'style_imgs': torch.stack([item['style_imgs'] for item in batch]),
        'strokes': torch.stack([item['strokes'] for item in batch]),
        'stroke_mask': torch.stack([item['stroke_mask'] for item in batch]),
        'writer_id': torch.tensor([item['writer_id'] for item in batch]),
        'word': [item['word'] for item in batch],
    }


def get_dataloader(lmdb_path, batch_size=32, num_workers=4, shuffle=True, **kwargs):
    """Create dataloader for training."""
    dataset = WordStrokeDataset(lmdb_path, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Test dataset
    dataset = WordStrokeDataset("/home/server/glossy/sdt_word/data/train.lmdb")
    print(f"Dataset size: {len(dataset)}")
    print(f"Writers: {len(dataset.writer_samples)}")

    # Test a sample
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  Content img: {sample['content_img'].shape}")
    print(f"  Style imgs: {sample['style_imgs'].shape}")
    print(f"  Strokes: {sample['strokes'].shape}")
    print(f"  Writer: {sample['writer_id']}")
    print(f"  Word: {sample['word']}")

    # Test dataloader
    loader = get_dataloader("/home/server/glossy/sdt_word/data/train.lmdb", batch_size=4)
    batch = next(iter(loader))
    print(f"\nBatch:")
    print(f"  Content img: {batch['content_img'].shape}")
    print(f"  Style imgs: {batch['style_imgs'].shape}")
    print(f"  Strokes: {batch['strokes'].shape}")
