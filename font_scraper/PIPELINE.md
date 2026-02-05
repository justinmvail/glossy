# Font-to-Stroke Training Pipeline

Generate high-quality stroke training data from fonts for SDT character training.

## Overview

```
Fonts → Filter → Vectorize → Process → Validate → Database → Review
```

## Pipeline Stages

### 1. Font Scraping (Expanded)

**Sources:**
- DaFont (`dafont_scraper.py`)
- FontSpace (`fontspace_scraper.py`)
- Google Fonts (`google_fonts_scraper.py`)

**Categories to include:**
- Handwriting (print-style)
- Sans-serif
- Serif
- Monospace/Fixed-width
- School/Educational
- Typewriter
- Basic/Simple

**Categories to skip:**
- Script/Calligraphy (cursive)
- Gothic/Blackletter
- Decorative/Fancy
- Graffiti
- Dingbats/Symbols
- Distorted/Eroded
- Foreign Look

**Process all font variants:** Regular, Bold, Italic, Light, etc.

**Output:** Downloaded fonts + DB records (name, source URL, category, license, variant)

**Checkpoint:** `fonts/` directory + `fonts` table in DB

---

### 2. Font Completeness Check

Verify each font has all required ASCII printable glyphs (95 characters):
- Uppercase: A-Z (26)
- Lowercase: a-z (26)
- Digits: 0-9 (10)
- Special: `!"#$%&'()*+,-./:;<=>?@[\]^_{|}~` and space (33)

Fonts missing characters are flagged but not removed (partial data still useful).

**Output:** DB records (missing glyphs list, completeness score 0-1)

**Checkpoint:** `font_checks` table updated

---

### 3. Deduplication

Use perceptual hashing (phash) to find visually similar fonts.

```python
from font_utils import FontDeduplicator

deduper = FontDeduplicator(threshold=8)
duplicates = deduper.find_duplicates(font_scores)
keep, remove = deduper.select_best_from_groups(duplicates)
```

**Output:** DB records (duplicate_group_id, kept/removed flag)

**Checkpoint:** `font_checks` table updated

---

### 4. Cursive Detection

Detect connected/cursive fonts before expensive OCR filtering.

```python
from font_utils import CursiveDetector

detector = CursiveDetector()
is_cursive, connectivity_score = detector.check(font_path)
```

**Method:**
1. Render test word "minimum" (many adjacent vertical strokes)
2. Binarize and find connected components
3. If components << expected letters, font is cursive

**Threshold:** Connectivity score > 0.7 = cursive (letters connected)

**Output:** DB records (connectivity_score, is_cursive flag)

**Checkpoint:** `font_checks` table updated

---

### 5. OCR Pre-filter

Filter fonts that don't render readable text.

```python
from font_utils import OCRPrefilter

prefilter = OCRPrefilter(confidence_threshold=0.7)
passed, confidence, ocr_text = prefilter.check(font_path, "Hello World 123")
```

**Method:**
1. Render sample text with the font
2. Run TrOCR (via Docker container)
3. Compare OCR result to expected text
4. Filter if confidence below threshold

**Output:** DB records (sample_image_path, ocr_text, ocr_confidence, pass/fail)

**Checkpoint:** `prefilter_samples/` directory + `font_checks` table updated

---

### 6. Character Rendering

Generate individual character images for InkSight processing.

**Spec (matching SDT training):**
- Image height: 64 pixels
- Width: variable (preserve aspect ratio)
- Background: white (255)
- Foreground: black (0)
- Format: PNG grayscale

**Character set:** ASCII printable (95 chars)

```python
from font_utils import CharacterRenderer

renderer = CharacterRenderer(height=64)
for char in renderer.ASCII_PRINTABLE:
    img_path = renderer.render(font_path, char, output_dir)
```

**Output:** Character images + DB records (font_id, char, image_path)

**Checkpoint:** `characters/` directory + `characters` table

---

### 7. InkSight Vectorization

Convert character images to stroke data using InkSight in Docker.

```python
from docker.inksight_docker import InkSightDocker

inksight = InkSightDocker()

# Batch process for efficiency
results = inksight.batch_process(font_path, characters)
```

**Tuning:**
- Batch characters per font to reduce container overhead
- GPU-accelerated (NVIDIA Docker runtime)
- Parallel fonts if multiple GPUs available

**Output:** DB records (strokes_raw JSON, point_count)

**Checkpoint:** `characters` table updated with raw strokes

---

### 8. Post-processing

Apply stroke cleanup pipeline.

```python
from stroke_processing import cleanup, filter_strokes, snap, smooth

for stroke_data in raw_strokes:
    processed = cleanup(stroke_data)
    processed = filter_strokes(processed)
    processed = snap(processed)
    processed = smooth(processed)
```

**Operations:**
- **Cleanup:** Remove artifacts, merge nearby points
- **Filter:** Remove strokes below point threshold
- **Snap:** Align points to grid for consistency
- **Smooth:** Apply smoothing to reduce noise

**Output:** DB records (strokes_processed JSON)

**Checkpoint:** `characters` table updated with processed strokes

---

### 9. OCR Validation

Validate processed strokes produce recognizable characters.

```python
from font_utils import OCRValidator

validator = OCRValidator()
result = validator.validate(processed_strokes, expected_char)
# Returns: ocr_result, confidence, is_match
```

**Method:**
1. Render processed strokes as image
2. Run TrOCR
3. Compare to expected character
4. Calculate quality score

**Quality score formula:**
```
quality = ocr_confidence * (1.0 if match else 0.5) * completeness_factor
```

**Output:** DB records (ocr_result, ocr_confidence, is_match, quality_score)

**Checkpoint:** `characters` table fully populated

---

### 10. Database Schema

```sql
-- Fonts table
CREATE TABLE fonts (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT,  -- 'dafont', 'fontspace', 'google'
    url TEXT,
    category TEXT,
    license TEXT,
    variant TEXT,  -- 'Regular', 'Bold', 'Italic', etc.
    file_path TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Removal reasons lookup table
CREATE TABLE removal_reasons (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,  -- 'incomplete', 'duplicate', 'cursive', etc.
    description TEXT
);
-- Pre-populated: incomplete, duplicate, cursive, contextual, ocr_prefilter,
--                ocr_validation, low_quality, manual, load_error

-- Track removed fonts and why
CREATE TABLE font_removals (
    id INTEGER PRIMARY KEY,
    font_id INTEGER REFERENCES fonts(id),
    reason_id INTEGER REFERENCES removal_reasons(id),
    details TEXT,  -- Additional context (e.g., "duplicate of font_id 123")
    removed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Font quality checks
CREATE TABLE font_checks (
    id INTEGER PRIMARY KEY,
    font_id INTEGER UNIQUE REFERENCES fonts(id),

    -- Completeness
    completeness_score REAL,
    missing_glyphs TEXT,  -- JSON array

    -- Deduplication
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_group_id INTEGER,
    keep_in_group BOOLEAN,

    -- Cursive detection
    connectivity_score REAL,
    contextual_score REAL,
    is_cursive BOOLEAN,

    -- OCR prefilter
    prefilter_image_path TEXT,
    prefilter_ocr_text TEXT,
    prefilter_confidence REAL,
    prefilter_passed BOOLEAN,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual characters
CREATE TABLE characters (
    id INTEGER PRIMARY KEY,
    font_id INTEGER REFERENCES fonts(id),
    char TEXT NOT NULL,

    -- Rendering
    image_path TEXT,

    -- Strokes
    strokes_raw TEXT,  -- JSON
    strokes_processed TEXT,  -- JSON
    point_count INTEGER,

    -- Best OCR result (denormalized for quick access)
    best_ocr_result TEXT,
    best_ocr_confidence REAL,
    best_ocr_match BOOLEAN,

    -- Quality
    quality_score REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(font_id, char)
);

-- OCR run history per character
CREATE TABLE ocr_runs (
    id INTEGER PRIMARY KEY,
    character_id INTEGER REFERENCES characters(id),

    -- What was OCR'd
    stage TEXT NOT NULL,  -- 'raw_strokes', 'processed_strokes', 'prefilter'
    image_path TEXT,

    -- Results
    ocr_result TEXT,
    ocr_confidence REAL,
    ocr_match BOOLEAN,

    -- Metadata
    model TEXT,  -- 'trocr', 'tesseract', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

### 11. Viewer/Editor

Query database to find best training candidates:

```sql
-- Find fonts with best average character quality
SELECT f.name, f.source, AVG(c.quality_score) as avg_quality
FROM fonts f
JOIN font_checks fc ON f.id = fc.font_id
JOIN characters c ON f.id = c.font_id
WHERE fc.is_cursive = FALSE
  AND fc.prefilter_passed = TRUE
  AND fc.is_duplicate = FALSE
ORDER BY avg_quality DESC
LIMIT 100;

-- Find characters needing review (low confidence but passed)
SELECT f.name, c.char, c.quality_score, c.ocr_confidence
FROM characters c
JOIN fonts f ON c.font_id = f.id
WHERE c.quality_score BETWEEN 0.5 AND 0.8
ORDER BY c.quality_score ASC;
```

**Viewer features:**
- Browse fonts by quality score
- View character grids per font
- Filter by category, source, quality range
- Side-by-side: original render vs stroke render

**Editor features:**
- Quick stroke correction
- Mark characters as "reviewed" or "rejected"
- Batch operations (reject all from font, etc.)

---

## Resumability

Each stage writes to disk and database. Pipeline can resume from any checkpoint:

```python
from pipeline import FontPipeline

pipeline = FontPipeline(db_path='fonts.db')

# Resume from where we left off
pipeline.run(
    resume=True,
    stages=['cursive_detect', 'ocr_prefilter', 'render', 'vectorize']
)
```

**Checkpoint locations:**
| Stage | Disk | Database |
|-------|------|----------|
| Scraping | `fonts/` | `fonts` table |
| Completeness | - | `font_checks.completeness_*` |
| Deduplication | - | `font_checks.duplicate_*` |
| Cursive | - | `font_checks.cursive_*` |
| OCR Prefilter | `prefilter_samples/` | `font_checks.prefilter_*` |
| Rendering | `characters/` | `characters.image_path` |
| Vectorization | - | `characters.strokes_raw` |
| Processing | - | `characters.strokes_processed` |
| Validation | - | `characters.ocr_*`, `quality_score` |

---

## Stroke Editor

Web-based tool for viewing and manually editing stroke data (`stroke_editor.py`).

```bash
python3 stroke_editor.py
# Open http://localhost:5000
```

**Features:**
- Browse fonts → character grid → per-character canvas editor
- Strokes overlaid on semi-transparent rendered font character (224×224 coord space)
- Select/drag/add/delete points, add/delete strokes, shift+click for range selection
- Dot strokes rendered distinctly (stored as 2-point strokes for SDT compatibility)
- Server-side processing: extend_to_connect (gap closing), Gaussian smoothing (adaptive sigma)
- Dedup tool to remove overlapping artifact strokes from InkSight
- Skeleton-based stroke generation from font glyphs (skeletonization → junction analysis → stroke tracing → merge/absorption pipeline)
- Auto-detect structural markers (vertex, intersection, termination) from skeleton topology
- Font grid: "Generate All" batch skeleton generation, "Draw All" preview thumbnails with strokes overlaid, reject/unreject fonts
- Clear Page button to wipe all strokes and markers for a character
- All actions have both toolbar buttons and keyboard shortcuts
- Undo stack, unsaved change warnings, save to DB

**Skeleton Pipeline:**
The skeleton stroke generator (`skeleton_to_strokes`) converts font glyph masks into stroke paths:
1. **Skeletonize** the binary glyph mask (medial axis)
2. **Detect junctions** (degree ≥ 3 pixels) and cluster nearby junctions (within 12px)
3. **Trace strokes** between junction clusters and endpoints
4. **T-junction merge**: At junctions with 3+ strokes, merge the two longest if a short cross-branch exists (handles B's "3" shape)
5. **Direction-based merge**: Merge stroke pairs whose approach directions align (< 45°), with loop-stroke guards
6. **Stub absorption**: Remove short stubs (< 20px) by appending to neighboring strokes or deleting orphans
7. **Marker detection**: Classify junction endpoints as vertex (convergence), intersection (pass-through), or termination (free end)

**Shortcuts:** `V` select, `A` add stroke, `X` delete stroke, `D` dedup, `C` connect, `G` smooth+connect, `R` revert, `Del` delete point(s), `Shift+click` range select, `Ctrl+S` save, `Ctrl+Z` undo, `[`/`]` prev/next char

**Files:** `stroke_editor.py`, `templates/font_list.html`, `templates/char_grid.html`, `templates/editor.html`

---

### Shape Optimizer (Skel+Resample)

The Skel+Resample pipeline generates strokes by fitting parametric shape primitives to font glyph masks. This is a separate path from InkSight — it uses classical optimization to fit geometric templates directly to the rendered glyph.

**Shape Primitives:**
7 shape types: `vline`, `hline`, `diag`, `arc_right`, `arc_left`, `loop`, `u_arc`. Each has bbox-fraction parameters (e.g., center_x, center_y, radius_x, radius_y, angle_start, angle_end).

**Template System:**
`SHAPE_TEMPLATES` maps 62 characters (A-Z, a-z, 0-9) to lists of shapes with default parameters and bounds. Example for B: 1 vline (spine) + 2 arc_right (top/bottom bumps).

**Optimization Pipeline (current):**

```
Phase 0: Template + Affine  (1-2s)
Phase 1: Greedy per-shape   (varies)
  ↓
┌─────────────── Repeating cycle ───────────────┐
│ NM refinement → DE global search → NM polish  │
│ Repeat until converged or 1 hour timeout       │
└────────────────────────────────────────────────┘
  ↓
Final: best of affine vs shape optimization
```

- **Phase 0 — Template + Affine:** Calls `template_to_strokes()` to generate mask-aware strokes from waypoint templates, then optimizes a 6-parameter global affine transform (translate, scale, rotate, shear) followed by per-stroke refinement (translate + scale per stroke). If affine score >= 0.85, skips shape optimization entirely.
- **Phase 1 — Greedy:** Optimizes each shape individually against uncovered point cloud regions using Nelder-Mead + quick DE.
- **Repeating NM/DE/NM Cycle:** Joint optimization of all shape parameters together. Nelder-Mead for local refinement, Differential Evolution for global search, NM polish. Repeats until score stagnates (< 0.001 improvement over 2 consecutive cycles) or 1-hour time limit.

**Scoring:** Point cloud coverage within adaptive radius, with penalties for off-mask strokes, edge proximity, and stroke overlap. Score range 0-1 (higher = better).

**Streaming:** The `/api/optimize-stream/<font_id>?c=<char>` SSE endpoint streams every optimization frame to the editor UI in real time, showing phase, score, frame count, and elapsed time. A banner with spinner, stop button, and convergence info is displayed during streaming.

**Caching:** Winning parameters are cached in the `shape_params_cache` DB column. Subsequent runs start from cached params and can improve further. Cache is saved periodically during long runs so progress isn't lost.

---

### Shape Optimizer — Problems Encountered

Development of the shape optimizer for letter B revealed several significant issues:

#### 1. Arc parameters allow degenerate shapes
**Problem:** The `arc_right` shape's `ry_f` parameter is a radius, not a diameter. With default bounds (0.05, 0.8), a single arc could span 80% of the glyph height. For B, both top and bottom arcs would expand to cover the full glyph, merging into a single D shape.

**Fix:** Tightened ry_f bounds to (0.10, 0.28) and cy bounds to non-overlapping ranges: top arc (0.10, 0.35), bottom arc (0.65, 0.90). Similar constraints applied to S template.

#### 2. Nelder-Mead ignores bounds entirely
**Problem:** Scipy's Nelder-Mead is an unconstrained optimizer — it silently ignores the `bounds` parameter. Parameters would drift far outside bounds during optimization, producing invalid shapes. This was the root cause of multiple "it did the same thing again" failures where tightening bounds had no effect.

**Fix:** Added explicit `_clamp()` (np.clip) calls inside every NM objective function and on every NM result vector. Applied in both streaming and non-streaming endpoints across all phases (greedy, joint NM, polish NM).

#### 3. Angle bounds allow near-complete ellipses
**Problem:** Default angle bounds (-180, 0) and (0, 180) allowed arcs to sweep ~320 degrees, turning semicircular bumps into near-complete ellipses. Test showed ang_start=-168.6, ang_end=151.2 for what should be a ~180-degree arc.

**Fix:** Constrained angles to (-100, -80) and (80, 100) for B's arcs, enforcing approximately semicircular sweep.

#### 4. Post-processing merges optimized strokes
**Problem:** After optimization correctly produced 3 separate strokes (spine + 2 bumps), the client-side `_autoJoinStrokes()` post-processing merged them into one stroke because their endpoints were within PROX_DIST=12 pixels of each other. This created a single D-shaped stroke, undoing all the optimizer's work.

**Fix:** Removed `_autoJoinStrokes()` and `_reduceStrokes()` from `_skeletonFullPostProcess()` in editor.html. The optimizer already produces correct stroke topology.

#### 5. Optimization converges too slowly
**Problem:** With TIME_BUDGET=10s, DE was still actively improving when time expired. Score only reached ~0.63 after 15 seconds. The search space (15 dimensions for B) is too large for quick convergence.

**Fix:** Increased TIME_BUDGET from 10s to 30s, then to 3600s (1 hour) with automatic stagnation detection. Increased DE parameters: maxiter 80→200, popsize 15→20, tol 0.005→0.002. Added repeating NM→DE→NM cycles that stop when score improves less than 0.001 over 2 consecutive cycles.

#### 6. Affine result often beats shape optimizer
**Problem:** Phase 0 (template + affine) scores ~0.67 in 1-2 seconds, but the shape optimizer starting from template defaults scores lower (0.13 initially) and only catches up to ~0.67 after 30+ seconds. The shape optimizer doesn't use the affine result as a starting point — it starts from scratch.

**Current mitigation:** At the end, the system compares affine vs shape results and returns whichever scored higher. The affine result frequently wins.

**Underlying issue:** Converting affine-transformed raw strokes back into shape parameters (cx, cy, rx, ry, angles) is non-trivial, so the shape optimizer can't be seeded from the affine result. This means Phase 0 and Phases 1-4 are largely independent, and the expensive shape optimization may be wasted effort.

#### 7. Fundamental limitation: gradient-free optimization on a smooth problem
**Problem:** Both Nelder-Mead and Differential Evolution are gradient-free optimizers. They treat the scoring function as a black box, unable to determine *which direction* to move stroke points. For a 15-dimensional search space, this means thousands of evaluations to make incremental progress. The scoring function (point cloud coverage vs rendered strokes) is actually smooth and differentiable — gradient information exists but isn't being used.

**Not yet attempted:** Differentiable rendering (DiffVG) or PyTorch-based optimization could compute exact gradients, potentially converging in seconds instead of minutes. See "Future Directions" below.

---

### Shape Optimizer — Future Directions

**Differentiable rendering (DiffVG):**
The most promising improvement would be replacing scipy DE/NM with gradient-based optimization using a differentiable rasterizer. DiffVG or a custom PyTorch soft-rasterizer would allow:
- Represent strokes as bezier curves with learnable parameters
- Render differentiably to a raster image
- Compute MSE loss against the target glyph mask
- Backpropagate gradients directly to stroke parameters
- Converge in ~100-500 iterations at milliseconds each (vs thousands of evaluations at ~0.3ms each but with no gradient signal)

This has not been attempted. The repo has PyTorch (for EMNIST classifier) and TensorFlow (for InkSight) but neither is used for stroke optimization.

**InkSight as alternative:**
The InkSight Vision-Language model already generates strokes from glyph images and runs on GPU. For characters where the shape optimizer struggles (like B), InkSight output may be better out of the box. A quality comparison between InkSight strokes vs shape optimizer strokes for problematic characters has not been done.

**Hybrid approach:**
Use InkSight or DiffVG for initial stroke generation, then fine-tune with the existing shape optimizer's scoring function for final cleanup. This would combine ML's ability to quickly find approximate solutions with the optimizer's precise scoring.

---

## Goal

Find fonts with highest `quality_score` = least hand correction needed for SDT training.

Target: 1000+ high-quality fonts with complete ASCII character sets producing clean, OCR-verified stroke data.
