# InkSight Pipeline Optimizations & Quality Issues

## Current Performance
- ~2.2s per character
- ~3-4 min per font (95 chars + 8s Docker overhead)
- ~14.5 hours for 254 fonts

## Potential Optimizations

### 1. Batch Inference (HIGH IMPACT)
Process multiple characters at once on GPU instead of one at a time.
- Modify `inksight_vectorizer.py` to accept batch of images
- Could potentially 5-10x speedup on inference

### 2. Persistent Container (MEDIUM IMPACT)
Keep one Docker container running for all fonts instead of restarting per font.
- Saves ~8s overhead per font
- 254 fonts × 8s = ~34 min saved

### 3. Parallel Containers (MEDIUM IMPACT)
Run multiple containers if GPU memory allows.
- Check `nvidia-smi` for memory usage
- Could run 2-3 fonts in parallel

### 4. Reduce Character Set (QUICK WIN)
Start with A-Z, a-z, 0-9 (62 chars) instead of full 95 ASCII.
- 35% fewer characters to process
- Add special chars later if needed

## Quality Issues Found

### Low Point Count Characters
Some characters have suspiciously few points - InkSight failed to trace them properly.

**Affected:**
- 'E': 20% have < 10 points (should have 50-100+)
- Other complex letters may be affected

**Expected low points (OK):**
- `.` `:` `,` `-` `'` - simple punctuation

**Solutions:**
1. Set minimum point thresholds per character complexity:
   - Simple punct (. , : -): 1-5 pts OK
   - Letters (A-Z, a-z): minimum 15-20 pts
   - Complex chars (#, @, &): minimum 25+ pts
2. Flag/filter characters below threshold
3. Re-run failed characters with different settings
4. Use quality_score field to track

### Inconsistent Coordinate Sizes
InkSight outputs vary significantly in bounding box size:

**Examples:**
- 'B' ranges from 2×15 (failed) to 123×35 (good)
- Same font: 'A'=24×34, 'B'=121×31, 'O'=23×123

**Problems:**
- No consistent normalization to 0-224 space
- Tiny bounding boxes indicate tracing failures
- Makes training data inconsistent

**Solutions:**
1. Normalize all strokes to consistent coordinate space (0-224)
2. Filter characters with bounding box < 10×10 as failed
3. Flag characters where width/height ratio is extreme

### Post-Processing Quality Checks
- Filter space character (shows 508 pts of noise - should be empty)
- Validate stroke count makes sense for character
- Validate bounding box size is reasonable
- OCR validation on rendered strokes

## Shape Optimizer Performance

The Skel+Resample shape optimizer fits parametric shapes (vline, arc, loop, etc.) to glyph masks using scipy DE + Nelder-Mead.

### Current Performance
- **Phase 0 (affine):** 1-2s, scores 0.35-0.67 depending on font/character
- **Phase 1 (greedy):** 2-5s, scores 0.50-0.65
- **Full cycle (NM→DE→NM):** 20-40s per cycle
- **Convergence:** Typically stagnates after 1-3 cycles (score improves < 0.001)
- **Best scores achieved:** 0.67-0.70 for B across test fonts

### Key Bottleneck
Scoring function called ~10,000+ times per cycle. Each call: build shape points → smooth → snap to mask → KD-tree coverage query → compute penalties. At ~0.3ms per call, a single DE run (200 generations × 20 population) = 4000 evaluations = ~1.2s, but typically needs multiple restarts.

### Problems Identified
1. **Gradient-free on smooth problem** — DE/NM can't see which way to move points. A differentiable renderer (DiffVG/PyTorch) would give exact gradients and converge orders of magnitude faster.
2. **Affine often wins** — The quick affine transform (6 params, 1-2s) frequently outscores the full 15-dim shape optimizer (30s+). Shape optimization may be wasted computation.
3. **NM ignores bounds** — Required manual clamping in every objective function. Easy to miss and causes silent failures.
4. **Post-processing undoes optimization** — Auto-join merged separate strokes. Had to be disabled.
5. **Search space too large** — B has 15 parameters with tight interdependencies (arc heights must not overlap). Box constraints can't express these relationships.

### Potential Improvements
| Approach | Impact | Effort | Status |
|----------|--------|--------|--------|
| DiffVG / PyTorch differentiable rendering | Very high (100x faster convergence) | High | Not attempted |
| Use InkSight output instead of shape optimizer | High (already built) | Low | Not compared |
| GPU-accelerated batch scoring (cupy) | Medium (5-10x per-eval speedup) | Medium | Not attempted |
| Multiprocess DE population evaluation | Medium (4-8x on multi-core) | Medium | Not attempted |
| Reduce point cloud density | Low (minor speedup, quality tradeoff) | Low | Partially done (spacing=3) |

## Next Steps After Current Run
1. Check results quality - identify failed characters
2. Run post-processing (cleanup, snap, smooth)
3. OCR validation on processed strokes
4. Re-run or filter low-quality characters
5. Consider optimizations for re-runs
6. Compare InkSight vs shape optimizer output quality for B and other multi-arc characters
7. Evaluate DiffVG integration feasibility (PyTorch already in repo for EMNIST)
