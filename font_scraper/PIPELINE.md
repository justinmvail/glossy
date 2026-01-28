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
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Font quality checks
CREATE TABLE font_checks (
    id INTEGER PRIMARY KEY,
    font_id INTEGER REFERENCES fonts(id),

    -- Completeness
    completeness_score REAL,
    missing_glyphs TEXT,  -- JSON array

    -- Deduplication
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_group_id INTEGER,
    keep_in_group BOOLEAN,

    -- Cursive detection
    connectivity_score REAL,
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

    -- Validation
    ocr_result TEXT,
    ocr_confidence REAL,
    ocr_match BOOLEAN,

    -- Quality
    quality_score REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(font_id, char)
);

-- Indexes for common queries
CREATE INDEX idx_fonts_source ON fonts(source);
CREATE INDEX idx_fonts_category ON fonts(category);
CREATE INDEX idx_font_checks_cursive ON font_checks(is_cursive);
CREATE INDEX idx_font_checks_passed ON font_checks(prefilter_passed);
CREATE INDEX idx_characters_quality ON characters(quality_score);
CREATE INDEX idx_characters_font ON characters(font_id);
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

## Goal

Find fonts with highest `quality_score` = least hand correction needed for SDT training.

Target: 1000+ high-quality fonts with complete ASCII character sets producing clean, OCR-verified stroke data.
