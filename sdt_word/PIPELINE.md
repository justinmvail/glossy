# SDT Word-Level Training Pipeline

## Goal
Train a word-level handwriting synthesis model (WordSDT) for the GLOSSY app using synthetic data from single-line fonts. The model should support one-shot style learning from user handwriting samples.

## Architecture

**WordSDT** (58.6M parameters):
- Variable-width content encoder (ResNet + Transformer)
- Style encoder with writer/glyph heads
- GMM-based stroke decoder (20 mixtures)
- Outputs: (dx, dy, pen_state) sequences for pen plotter

## Pipeline Overview

```
Handwriting Fonts (TTF/OTF)
        ↓
   Skeletonize (font_to_single_line.py)
        ↓
Single-Line SVG Fonts
        ↓
   Score for human-likeness (handwriting_scorer.py)
        ↓
   Apply style transforms (synthetic_writers/generate.py)
        ↓
1000+ Synthetic Writers
        ↓
   Export SDT format (synthetic_writers/pipeline.py)
        ↓
Train WordSDT (train.py)
```

## Current Datasets

| Dataset | Fonts | Writers | Samples | Location |
|---------|-------|---------|---------|----------|
| Initial (small) | 43 | 43 | 10,742 | `data/` (LMDB) |
| Synthetic v1 | 42 | 1,000 | 146,988 | `sdt_data_1000/` |

## Files

### Core Training
| File | Purpose |
|------|---------|
| `models/word_model.py` | WordSDT architecture |
| `dataset.py` | Data loading from LMDB |
| `train.py` | Training loop with GMMStrokeLoss |
| `checkpoints/` | Model weights & training history |

### Data Generation
| File | Purpose |
|------|---------|
| `svg_font_parser.py` | Parse SVG single-line fonts |
| `generate_dataset.py` | Generate LMDB dataset from fonts |
| `font_scraper.py` | Download fonts from Google Fonts |
| `handwriting_scorer.py` | Score fonts for human-likeness |

### Synthetic Writers Pipeline (`synthetic_writers/`)
| File | Purpose |
|------|---------|
| `font_to_single_line.py` | Skeletonize TTF/OTF → single-line strokes |
| `svg_parser.py` | Parse SVG fonts to stroke coordinates |
| `generate.py` | Style transforms (slant, jitter, tremor) |
| `validate_strokes.py` | TrOCR/Tesseract legibility filter |
| `pipeline.py` | Full orchestration → SDT pickle format |

### Fonts
| Location | Contents |
|----------|----------|
| `fonts/raw/` | 52 downloaded Google Fonts (TTF) |
| `fonts/converted/` | 93 single-line fonts (52 Google + 41 EMS/Hershey) |
| `/home/server/svg-fonts/` | Original EMS & Hershey SVG fonts |

## Style Transforms

Applied to create synthetic writer variation:

**Global (per-writer):**
- Slant: -0.25 to 0.25
- Aspect ratio: 0.85 to 1.15
- Baseline wave: 0-3px amplitude
- Letter spacing: 0.9x to 1.2x

**Per-character:**
- Rotation jitter: 0-3°
- Scale jitter: 0-5%
- Position offset: 0-2px

**Per-stroke:**
- Curvature noise: 0-2px
- Endpoint drift: 0-1.5px

**Per-point:**
- Gaussian noise: 0-1px
- Tremor: 0-0.5px
- Speed variation: 0-30%

## Training Status

**Current Run** (Jan 25, 2026):
- Dataset: 10,742 samples, 43 writers
- Epoch: ~70/100
- Best val loss: -0.7518 (epoch 62)
- Hardware: GTX 1660 Super (6GB)

## Humanness Scoring

Top fonts by human-likeness (baseline wobble, stroke variation, slant variation):

1. Fontdiner Swanky (0.531)
2. Great Vibes (0.516)
3. Permanent Marker (0.513)
4. Covered By Your Grace (0.510)
5. Mrs Saint Delafield (0.507)

Full scores in `font_humanness_scores.json`.

## Next Steps

- [ ] Complete current training run (100 epochs)
- [ ] Test model generation quality
- [ ] Retrain on larger 147K sample dataset
- [ ] Expand to 100+ base fonts (currently 93)
- [ ] Add TrOCR validation for generated output
- [ ] Integrate with GLOSSY app for style extraction

## On-Device (GLOSSY App)

Use ML Kit for Flutter to:
1. Validate user handwriting samples
2. Reject bad samples before style extraction
3. Extract style vectors for personalization
