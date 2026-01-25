# Synthetic Writer Generation for SDT Training

Generate diverse synthetic handwriting training data from single-line fonts for fine-tuning SDT (Style-Disentangled Transformer) on English characters.

## The Idea

Single-line fonts are already in stroke format (perfect for pen plotters and SDT). By programmatically applying style transforms, we can turn ~100 fonts into thousands of synthetic "writers" with consistent internal style variation.

## Converting Handwriting Fonts to Single-Line

Don't have single-line fonts? Convert any TTF/OTF handwriting font:

```bash
# Install dependencies
pip install fonttools pillow scikit-image scipy

# Convert a font
python font_to_single_line.py \
    --font path/to/handwriting-font.ttf \
    --output ./converted_fonts \
    --chars "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
```

This uses **skeletonization** to extract the centerline:
1. Render glyph as high-res bitmap
2. Morphological thinning to 1px skeleton
3. Trace skeleton back to vector paths
4. Simplify and smooth

**Good sources for handwriting fonts:**
- [Google Fonts](https://fonts.google.com/?category=Handwriting) - Free, 100+ handwriting fonts
- [DaFont](https://www.dafont.com/theme.php?cat=601) - Thousands of free handwriting fonts
- [Font Squirrel](https://www.fontsquirrel.com/fonts/list/classification/handwritten) - Curated free fonts

**Batch convert multiple fonts:**
```bash
for font in ./handwriting_fonts/*.ttf; do
    name=$(basename "$font" .ttf)
    python font_to_single_line.py --font "$font" --output "./single_line/$name"
done
```

## Pipeline

```
Single-Line Fonts (SVG)
        ↓
    Parse strokes
        ↓
Generate Writer Styles (random but coherent parameters)
        ↓
Apply Transforms:
    - Global: slant, aspect ratio, baseline wave, scale, letter spacing
    - Per-char: rotation jitter, scale jitter, position offset
    - Per-stroke: curvature noise, start/end drift
    - Per-point: gaussian noise, tremor simulation, speed variation
        ↓
Export SDT Format (pickle files)
```

## Usage

```bash
# Basic usage
python pipeline.py \
    --fonts_dir ./fonts \
    --output_dir ./sdt_data \
    --num_writers 1000

# With more samples per character (for more variation)
python pipeline.py \
    --fonts_dir ./fonts \
    --output_dir ./sdt_data \
    --num_writers 500 \
    --samples_per_char 3

# Create sample SVG fonts for testing
python pipeline.py \
    --fonts_dir ./test_fonts \
    --output_dir ./test_output \
    --create_samples \
    --num_writers 10
```

## Output Format

Compatible with SDT's expected structure:

```
output_dir/
├── character_dict.pkl   # {char: index}
├── writer_dict.pkl      # {writer_id: index}  
├── train_data.pkl       # List of samples
├── writer_styles.json   # Style parameters (for debugging)
└── stats.json           # Dataset statistics
```

Each sample in `train_data.pkl`:
```python
{
    'char': 'A',
    'char_idx': 0,
    'writer_id': 'writer_00042',
    'writer_idx': 42,
    'strokes': [[x, y, pen_state], ...]  # pen_state: 0=drawing, 1=stroke end
}
```

## Transform Parameters

### Global (per-writer, consistent across all chars)

| Parameter | Range | Description |
|-----------|-------|-------------|
| slant | -0.25 to 0.25 | Italic-style shear |
| aspect_ratio | 0.85 to 1.15 | Width/height ratio |
| baseline_wave_amp | 0 to 3px | Sinusoidal baseline wobble |
| global_scale | 0.85 to 1.15 | Overall size |
| letter_spacing | 0.9 to 1.2 | Spacing multiplier |

### Per-character (sampled each character)

| Parameter | Range | Description |
|-----------|-------|-------------|
| char_rotation_std | 0 to 3° | Rotation jitter |
| char_scale_std | 0 to 0.05 | Scale variation |
| char_offset_x/y_std | 0 to 2px | Position jitter |

### Per-stroke

| Parameter | Range | Description |
|-----------|-------|-------------|
| stroke_curvature_noise | 0 to 2px | Control point displacement |
| stroke_start/end_drift | 0 to 1.5px | Endpoint variation |

### Per-point

| Parameter | Range | Description |
|-----------|-------|-------------|
| point_noise_std | 0 to 1px | Gaussian coordinate noise |
| tremor_amplitude | 0 to 0.5px | High-frequency tremor |
| speed_variation | 0 to 0.3 | Point density variation |

## Getting Single-Line Fonts

### Free sources:
- [FontSpace single-line category](https://www.fontspace.com/category/single-line) - 85 free fonts
- [Relief SingleLine](https://github.com/isdat-type/Relief-SingleLine) - Open source, OFL license
- [K40 Laser Cutter fonts](https://k40lasercutter.com/product-category/single-line-fonts/) - 3 free

### Commercial:
- [OneLineFonts.com](https://www.onelinefonts.com/) - Large collection
- [Single Line Fonts (Leslie)](https://leslie-fonts.myshopify.com/) - Script and print styles

### Converting TTF/OTF to SVG:
```bash
# Using FontForge
fontforge -lang=py -c 'import fontforge; f=fontforge.open("font.ttf"); f.generate("font.svg")'

# Or use online converters, then extract paths
```

## Mixing with Real Data

For best results, combine synthetic data with real handwriting:

1. Pre-train on synthetic data (large volume, clean)
2. Fine-tune on real data (IAM-OnDB, smaller but authentic)

This gives the model a good foundation while learning real human variation.

## Files

- `pipeline.py` - Main orchestration script
- `generate.py` - Writer style generation and transforms
- `svg_parser.py` - SVG font parsing to stroke coordinates

## Requirements

```
numpy
```

No deep learning dependencies - this is just data generation.
