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

### Post-Processing Quality Checks
- Filter space character (shows 508 pts of noise - should be empty)
- Validate stroke count makes sense for character
- OCR validation on rendered strokes

## Next Steps After Current Run
1. Check results quality - identify failed characters
2. Run post-processing (cleanup, snap, smooth)
3. OCR validation on processed strokes
4. Re-run or filter low-quality characters
5. Consider optimizations for re-runs
