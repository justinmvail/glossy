"""
Generate SDT training dataset from SVG single-line fonts.
Each font = one "writer" for style disentanglement.
"""

import os
import sys
import json
import lmdb
import pickle
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import io

from svg_font_parser import (
    parse_svg_font,
    generate_word_strokes,
    normalize_strokes,
    strokes_to_relative
)

# Common English words for training vocabulary
WORD_LIST = [
    # High frequency words
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    # Names
    "John", "Mary", "James", "Sarah", "Michael", "Emma", "David", "Anna",
    "Robert", "Lisa", "William", "Jennifer", "Richard", "Elizabeth", "Joseph",
    # Greetings and common phrases
    "Hello", "Dear", "Love", "Best", "Thanks", "Happy", "Birthday", "Merry",
    "Christmas", "Congratulations", "Welcome", "Goodbye", "Friend", "Family",
    # More words for variety
    "beautiful", "wonderful", "amazing", "special", "perfect", "great",
    "today", "tomorrow", "yesterday", "always", "never", "forever",
    "heart", "soul", "mind", "life", "world", "home", "house", "garden",
    "spring", "summer", "autumn", "winter", "morning", "evening", "night",
    "sun", "moon", "star", "sky", "rain", "snow", "wind", "cloud",
    "red", "blue", "green", "yellow", "white", "black", "pink", "purple",
    "mother", "father", "sister", "brother", "daughter", "son", "child",
    "hope", "dream", "wish", "joy", "peace", "grace", "faith", "truth",
    # Card-specific words
    "Thank", "you", "very", "much", "sincerely", "yours", "truly",
    "thinking", "missing", "loving", "caring", "sending", "wishing",
    "warm", "wishes", "regards", "blessings", "prayers", "thoughts",
    # More common words
    "time", "year", "people", "way", "day", "man", "woman", "thing",
    "look", "want", "give", "use", "find", "tell", "ask", "work", "seem",
    "feel", "try", "leave", "call", "good", "new", "first", "last", "long",
    "little", "own", "other", "old", "right", "big", "high", "small",
    "large", "next", "early", "young", "important", "few", "public", "bad",
    # Extended vocabulary
    "remember", "celebrate", "appreciate", "treasure", "cherish", "embrace",
    "journey", "adventure", "memory", "moment", "chapter", "story", "beginning",
    "gratitude", "kindness", "happiness", "friendship", "relationship",
    "together", "forever", "always", "sometimes", "often", "usually",
]

# Expand with lowercase versions
WORD_LIST = list(set([w.lower() for w in WORD_LIST] + WORD_LIST))


def render_word_image(word, height=64, padding=4):
    """
    Render a word as a binary content image.
    This is the "target" content that SDT conditions on.
    """
    from PIL import ImageFont

    # Try to find a basic font
    font_size = height - 2 * padding
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # Measure text
    dummy_img = Image.new('L', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Create image with padding
    img_width = text_width + 2 * padding
    img = Image.new('L', (img_width, height), 255)  # White background
    draw = ImageDraw.Draw(img)

    # Center text vertically
    y_offset = (height - text_height) // 2 - bbox[1]
    draw.text((padding, y_offset), word, fill=0, font=font)  # Black text

    return img


def strokes_to_image(strokes, height=64, line_width=2):
    """
    Render strokes as an image for visualization/debugging.
    """
    if strokes is None or len(strokes) == 0:
        return None

    # Get bounds
    min_x, max_x = strokes[:, 0].min(), strokes[:, 0].max()
    min_y, max_y = strokes[:, 1].min(), strokes[:, 1].max()

    width = int(max_x - min_x) + 20
    if width < 10:
        width = 64

    img = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(img)

    # Draw strokes
    prev_x, prev_y = None, None
    for i, (x, y, pen_up) in enumerate(strokes):
        curr_x = x - min_x + 10
        curr_y = y - min_y + 5

        if prev_x is not None and pen_up == 0:
            draw.line([(prev_x, prev_y), (curr_x, curr_y)], fill=0, width=line_width)

        prev_x, prev_y = curr_x, curr_y

    return img


def generate_dataset_from_files(font_files, output_dir, max_words_per_font=None):
    """
    Generate complete training dataset from a list of SVG font files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(font_files)} font files")

    # Parse all fonts
    all_fonts = {}
    for font_path in font_files:
        try:
            font_id, glyphs, advances = parse_svg_font(str(font_path))
            if len(glyphs) > 20:  # Only use fonts with enough characters
                all_fonts[font_id] = {
                    'path': str(font_path),
                    'glyphs': glyphs,
                    'advances': advances
                }
                print(f"  Loaded {font_id}: {len(glyphs)} glyphs")
        except Exception as e:
            print(f"  Failed to parse {font_path}: {e}")

    print(f"\nLoaded {len(all_fonts)} usable fonts")

    # Create writer ID mapping
    writer_ids = {font_id: idx for idx, font_id in enumerate(sorted(all_fonts.keys()))}

    # Generate dataset
    dataset = []
    stats = {'total': 0, 'skipped': 0, 'by_font': {}}

    for font_id, font_data in all_fonts.items():
        writer_id = writer_ids[font_id]
        glyphs = font_data['glyphs']
        advances = font_data['advances']
        font_count = 0

        words_to_use = WORD_LIST
        if max_words_per_font:
            words_to_use = WORD_LIST[:max_words_per_font]

        for word in words_to_use:
            # Check if font has all characters
            if not all(c in glyphs for c in word):
                stats['skipped'] += 1
                continue

            # Generate strokes
            word_strokes = generate_word_strokes(word, glyphs, advances)
            if word_strokes is None or len(word_strokes) < 3:
                stats['skipped'] += 1
                continue

            # Normalize strokes
            norm_strokes = normalize_strokes(word_strokes, target_height=56)  # Leave room for padding
            if norm_strokes is None:
                stats['skipped'] += 1
                continue

            # Convert to relative coordinates
            rel_strokes = strokes_to_relative(norm_strokes)
            if rel_strokes is None:
                stats['skipped'] += 1
                continue

            # Render content image
            content_img = render_word_image(word, height=64)

            # Store sample
            sample = {
                'word': word,
                'writer_id': writer_id,
                'font_id': font_id,
                'strokes_abs': norm_strokes,  # Absolute coords (N, 3)
                'strokes_rel': rel_strokes,   # Relative coords (N, 3)
                'content_img': np.array(content_img),
                'num_points': len(rel_strokes),
            }
            dataset.append(sample)

            font_count += 1
            stats['total'] += 1

        stats['by_font'][font_id] = font_count
        print(f"  {font_id}: {font_count} words")

    print(f"\nGenerated {stats['total']} samples, skipped {stats['skipped']}")

    # Save as LMDB (SDT format)
    lmdb_path = output_dir / "train.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=1024*1024*1024*10)  # 10GB

    with env.begin(write=True) as txn:
        for idx, sample in enumerate(dataset):
            # SDT expects specific keys
            key = f"{idx:08d}".encode()

            # Serialize sample
            value = pickle.dumps({
                'word': sample['word'],
                'writer_id': sample['writer_id'],
                'strokes': sample['strokes_rel'],  # (N, 3) float32
                'img': sample['content_img'],      # (H, W) uint8
            })
            txn.put(key, value)

        # Store metadata
        txn.put(b'__len__', str(len(dataset)).encode())
        txn.put(b'__keys__', pickle.dumps([f"{i:08d}" for i in range(len(dataset))]))

    env.close()

    # Save metadata JSON
    metadata = {
        'num_samples': len(dataset),
        'num_writers': len(all_fonts),
        'writer_ids': writer_ids,
        'vocab_size': len(set(s['word'] for s in dataset)),
        'stats': stats,
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save a few sample images for verification
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    for i, sample in enumerate(dataset[:20]):
        # Save content image
        content_img = Image.fromarray(sample['content_img'])
        content_img.save(samples_dir / f"{i:04d}_{sample['font_id']}_{sample['word']}_content.png")

        # Save stroke rendering
        stroke_img = strokes_to_image(sample['strokes_abs'])
        if stroke_img:
            stroke_img.save(samples_dir / f"{i:04d}_{sample['font_id']}_{sample['word']}_strokes.png")

    print(f"\nDataset saved to {output_dir}")
    print(f"Sample images saved to {samples_dir}")

    return dataset, metadata


if __name__ == "__main__":
    import sys

    # Multiple font directories
    fonts_dirs = [
        "/home/server/svg-fonts/fonts",
        "/home/server/Relief-SingleLine/fonts/open_svg"
    ]
    output_dir = "/home/server/glossy/sdt_word/data"

    # Collect all font files
    all_font_files = []
    for fonts_dir in fonts_dirs:
        all_font_files.extend(list(Path(fonts_dir).rglob("*.svg")))

    print(f"Found {len(all_font_files)} font files total")

    dataset, metadata = generate_dataset_from_files(all_font_files, output_dir)

    print("\n=== Dataset Summary ===")
    print(f"Total samples: {metadata['num_samples']}")
    print(f"Writers (fonts): {metadata['num_writers']}")
    print(f"Unique words: {metadata['vocab_size']}")
