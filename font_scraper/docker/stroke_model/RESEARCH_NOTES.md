# Stroke Model Research Notes
## March 19, 2026 — Deep Analysis & Recommendations

## 1. Current State

### Architecture
Autoregressive stroke predictor:
- ResNet-18 encoder (224x224 glyph image + char embedding) → 49 spatial tokens
- Per-step: ResidualEncoder (canvas state CNN) → StrokeDecoderStep (transformer, 1 query + step embedding) → output heads (existence, points, width, point_count)
- 8 steps max, each renders via Triton and composites onto canvas
- Loss: canvas MSE (rendered vs glyph) + stroke_length penalty

### What Works
- **Autoregressive structure**: Strokes cover different areas (no collapse)
- **Triton renderer**: Fast, differentiable, correct gradients
- **Experiment tracking**: Automated snapshots before restart

### What Doesn't Work
- **Width=8.0**: 1-2 strokes zigzag to cover everything
- **Width=2.0**: Spam of tiny strokes on thick fonts, nothing on thin fonts
- **Stroke length penalty**: Creates short fat rectangles instead of paths
- **No zigzag prevention**: Smoothness collapses training, reversal not tried with autoregressive yet
- **Vast.ai loss weights**: Mismatch — uses total_loss() weights with autoregressive_loss() code

---

## 2. Related Work (Key Papers)

### Most Relevant to Our Approach

**Paint Transformer (ICCV 2021)** — Feed-forward set prediction with Hungarian matching
- Self-training: randomly sample strokes → render → train to recover
- Hungarian matching assigns each prediction to a specific target
- 60% overlap threshold during data generation marks redundant strokes as invalid
- Key insight: infinite paired training data without annotation
- GitHub: Huage001/PaintTransformer

**LIVE (CVPR 2022)** — Progressive path addition (validates our autoregressive approach)
- Adds paths one at a time, each initialized at max reconstruction error
- UDF focal loss (distance-weighted) + Xing (self-crossing) loss
- Key insight: sequential "explain the residual" prevents collapse by construction
- GitHub: ma-xu/LIVE

**StrokeStyles (ACM TOG 2022, Adobe)** — Geometric font stroke segmentation (NO ML)
- Uses medial axis, curvilinear shape features, junction types
- Works across all languages and writing systems without training
- Key insight: clean stroke decomposition is achievable geometrically
- Potential use: generate pseudo-GT for supervised pretraining

**Im2Vec (CVPR 2021)** — Image to vector without vector supervision
- RNN decoder generates closed Bezier paths with depth ordering
- Cyclic convolutions maintain path closure
- Unused paths collapse to single points (naturally eliminated)
- Image-space loss only via DiffVG
- GitHub: preddy5/Im2Vec

**HieroSA (2026)** — RL-based stroke decomposition without language priors
- Fine-tuned VLM generates stroke coordinates as text
- Coverage reward with threshold: each stroke must contribute novel pixels
- Key insight: overlap threshold tau prevents redundancy
- No ground truth stroke labels needed

### Font-Specific

**DeepSVG (NeurIPS 2020)** — Hierarchical Transformer VAE for vector graphics
- Two-level: path encoder → aggregator → path decoder → command decoder
- Learned index embeddings break query symmetry (same idea as our stroke_index_embed)
- Requires GT vector data (SVG supervision)
- GitHub: alexandre01/deepsvg

**DeepVecFont (SIGGRAPH Asia 2021)** — Dual-modality font generation
- CNN image encoder + Transformer sequence encoder, fused
- Image modality captures global structure, sequence captures geometry
- Loss includes DiffVG rasterization comparison
- GitHub: yizhiwang96/deepvecfont

**LVGM (2025)** — Stroke-level font generation with LLM
- VQ-VAE compresses strokes → fine-tuned DeepSeek-Coder predicts stroke tokens
- Generates stroke-by-stroke in writing order
- All SVG commands normalized to cubic Bezier

### Rendering

**Bezier Splatting (NeurIPS 2025)** — 30x faster than DiffVG
- Samples 2D Gaussians along Bezier curves
- Adaptive pruning/densification escapes local minima
- GitHub: xiliu8006/Bezier_splatting

**Stylized Neural Painting (CVPR 2021)** — Neural renderer
- Solves zero-gradient problem via optimal transport perspective
- Disentangles shape from color
- GitHub: jiupinjia/stylized-neural-painting

---

## 3. Diagnosis: Why Current Model Fails

### Problem 1: Width Floor Dilemma
- Width=8.0: too few strokes needed, zigzag covers everything
- Width=2.0: too many strokes needed for thick fonts, nothing for thin fonts
- Width=1.0 + length penalty: short fat rectangles
- **Root cause**: no natural incentive to match stroke width to font thickness

### Problem 2: Zigzag / Winding Paths
- Smoothness penalty (curvature): collapses training when combined with thick strokes
- Reversal penalty: untested with autoregressive model
- Stroke length: penalizes all length, not just wasted length
- **Root cause**: missing sinuosity penalty (path length / endpoint distance)

### Problem 3: Renderer Gradient Starvation
- Triton backward only flows gradients through single closest segment per pixel
- Strokes far from target get zero gradient signal
- Less critical in autoregressive (each stroke rendered alone) but still limits refinement

### Problem 4: No Supervised Signal
- Pure image-space loss makes training extremely hard
- Model must simultaneously learn: where to place strokes, how many to use, what width, how many points
- Every paper that works well has SOME form of supervision or curriculum

---

## 4. Recommended Changes (Priority Order)

### A. Sinuosity Penalty (Replace stroke_length) — HIGH IMPACT, LOW EFFORT
Replace total stroke length with sinuosity = path_length / endpoint_distance.
- Straight stroke: sinuosity ≈ 1.0 (no penalty)
- Zigzag stroke: sinuosity ≈ 5-10x (heavy penalty)
- Long straight stroke: still sinuosity ≈ 1.0 (no penalty — this is the key difference)
- Allows model to use long strokes when needed without penalty

### B. Self-Training Pretraining (Paint Transformer Style) — HIGH IMPACT, MEDIUM EFFORT
Before training on real fonts:
1. Generate random polyline strokes (1-8 strokes, random widths, positions)
2. Render them onto a canvas to create synthetic glyph
3. Train the autoregressive model to recover the strokes
4. Use Chamfer distance or L1 on stroke params (with existence BCE)
5. This gives the model a sense of "what a stroke is" before seeing real fonts
6. Then finetune on real fonts with canvas MSE

Benefits:
- Each prediction step gets clear gradient signal (matched to a specific target stroke)
- Model learns stroke primitives (straight lines, curves, widths) quickly
- Autoregressive ordering emerges naturally (largest strokes first)
- No manual annotation needed

### C. Medial Axis Pseudo-GT (StrokeStyles Approach) — HIGH IMPACT, MEDIUM EFFORT
Use the existing skeleton/medial axis code in the codebase to generate pseudo-ground-truth strokes:
1. For each font glyph, compute medial axis (skeleton)
2. Extract stroke paths from skeleton (we already have this in stroke_core.py!)
3. Use these as supervised targets with Hungarian matching
4. Allows direct stroke-level loss instead of only image-space loss

Benefits:
- Leverages existing codebase (stroke_pipeline.py already does this)
- Provides stroke-level supervision without manual annotation
- Can be combined with image-space loss for refinement

### D. Curriculum: Width Annealing — MEDIUM IMPACT, LOW EFFORT
Instead of fixed width floor:
1. Start training with width floor = 8.0 (easy to cover glyphs)
2. Gradually decrease floor over epochs: 8 → 6 → 4 → 2 → 1
3. Model first learns stroke placement with thick strokes
4. Then refines into thinner, more precise strokes

This avoids the "can't learn anything because strokes are too thin" problem while
eventually allowing thin strokes for fine detail.

### E. Coverage Threshold per Stroke (HieroSA Approach) — MEDIUM IMPACT, LOW EFFORT
Add a per-step check: after rendering stroke k, compute how many NEW glyph pixels
it covers. If below threshold tau, penalize existence.
- Forces each stroke to contribute meaningful new coverage
- Prevents "dot spam" (many tiny strokes doing nothing)
- Natural complement to autoregressive structure

### F. Improve Renderer Gradients — MEDIUM IMPACT, HIGH EFFORT
Consider switching to Bezier Splatting for smoother gradient flow:
- 30x faster forward, 150x faster backward than DiffVG
- Adaptive densification escapes local minima
- Or: modify Triton backward to accumulate gradients from top-K closest segments
  instead of just the single closest

### G. Validation Set + Early Stopping — LOW IMPACT, LOW EFFORT
- Hold out 10% of fonts for validation
- Monitor validation loss to detect overfitting
- Currently no way to know if model is overfit

---

## 5. Proposed Training Pipeline

### Phase 1: Synthetic Pretraining (5-10 epochs)
- Generate random polyline strokes on blank canvas
- Train autoregressive model to recover them
- Loss: Chamfer distance on matched stroke pairs + existence BCE
- Purpose: learn stroke primitives and autoregressive ordering

### Phase 2: Pseudo-GT Finetuning (10-20 epochs)
- Use medial axis to extract stroke paths from real font glyphs
- Train with Hungarian matching + canvas MSE
- Width floor at 4.0, sinuosity penalty on
- Purpose: learn real font structure

### Phase 3: Full Training (50-100 epochs)
- Canvas MSE + sinuosity penalty only
- Width floor at 1.0, fully learnable
- Coverage threshold per stroke
- Purpose: refine to production quality

---

## 6. Quick Wins (Can Do Now)

1. **Sinuosity penalty** — replace stroke_length in autoregressive_loss
2. **Fix Vast.ai loss weight mismatch** — use autoregressive_loss weights
3. **Width curriculum** — start at 8.0, decay to 1.0 over epochs
4. **Add reversal penalty** — already implemented, just set weight > 0
5. **Equal outside penalty** — already done (10x both sides)

---

## 7. Key Insight from Research

The most successful approaches (Paint Transformer, LIVE, DeepSVG, HieroSA) all have
one thing in common: **they don't rely solely on image-space loss for stroke supervision.**
They either:
- Generate synthetic paired data (Paint Transformer)
- Use progressive addition with explicit coverage checks (LIVE, HieroSA)
- Require GT vector data (DeepSVG, DeepVecFont)
- Use RL with shaped rewards (SPIRAL, CalliRewrite)

Our current approach (pure canvas MSE) is the hardest possible setting. The autoregressive
structure helps (LIVE validates this), but we need additional signal — either synthetic
pretraining, pseudo-GT from medial axis, or coverage thresholds — to make it work reliably.

---

## Sources
- Paint Transformer: arxiv.org/abs/2108.03798
- LIVE: ma-xu.github.io/LIVE/
- StrokeStyles: dl.acm.org/doi/10.1145/3505246
- Im2Vec: arxiv.org/abs/2102.02798
- DeepSVG: arxiv.org/abs/2007.11301
- HieroSA: arxiv.org/abs/2601.05508
- CLIPasso: clipasso.github.io/clipasso/
- Bezier Splatting: arxiv.org/abs/2503.16424
- DeepVecFont: github.com/yizhiwang96/deepvecfont
- LVGM: arxiv.org/html/2511.11119v1
- CalliRewrite: github.com/LoYuXr/CalliRewrite
- Sketch-RNN: arxiv.org/abs/1704.03477
- PyTorch-SVGRender: github.com/ximinng/PyTorch-SVGRender
