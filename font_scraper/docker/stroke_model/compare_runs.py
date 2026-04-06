#!/usr/bin/env python3
"""Generate an HTML comparison page of all training runs.

Shows tracking images at every ~10th epoch for each run, all on one page.
Handles both old format (epoch_NNN/ directly in run dir) and new format
(tracking/epoch_NNN/).

Usage:
    python3 compare_runs.py
    # Opens compare_runs.html in browser
"""

import os
import glob
import webbrowser
from pathlib import Path

STROKE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = STROKE_DIR / "experiments"
HISTORY_DIR = STROKE_DIR / "history"
OUTPUT_HTML = STROKE_DIR / "compare_runs.html"

# Sample images to show (in order)
SAMPLES = [
    "Brown_Fox_A", "Brown_Fox_R", "Brown_Fox_g", "Brown_Fox_8",
    "Coffee_Milkshake_A", "Coffee_Milkshake_R", "Coffee_Milkshake_g", "Coffee_Milkshake_8",
    "Buka_Bird_A", "Buka_Bird_R", "Buka_Bird_g", "Buka_Bird_8",
]

# Target epochs to show (closest available will be used)
TARGET_EPOCHS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]


def find_epoch_dirs(run_path):
    """Find epoch directories in a run, handling both formats."""
    tracking_dir = run_path / "tracking"
    if tracking_dir.is_dir():
        base = tracking_dir
    else:
        base = run_path

    epochs = {}
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("epoch_"):
            try:
                num = int(d.name.split("_")[1])
                # Verify it has images
                if any(d.glob("*.png")):
                    epochs[num] = d
            except (ValueError, IndexError):
                continue
    return epochs


def pick_epochs(available, targets):
    """Pick the closest available epoch to each target."""
    if not available:
        return []
    avail_sorted = sorted(available.keys())
    picked = []
    seen = set()
    for t in targets:
        closest = min(avail_sorted, key=lambda x: abs(x - t))
        if closest not in seen and abs(closest - t) <= 5:
            picked.append(closest)
            seen.add(closest)
    return picked


def discover_runs():
    """Find all runs with tracking images."""
    runs = []

    # Experiments (old format) — only runs with 5+ epochs
    if EXPERIMENTS_DIR.is_dir():
        for d in sorted(EXPERIMENTS_DIR.iterdir()):
            if not d.is_dir() or d.name == "current":
                continue
            epochs = find_epoch_dirs(d)
            if len(epochs) >= 5:
                runs.append({
                    "name": d.name,
                    "path": str(d),
                    "epochs": epochs,
                    "source": "experiments",
                })

    # History (new format)
    if HISTORY_DIR.is_dir():
        for d in sorted(HISTORY_DIR.iterdir()):
            if not d.is_dir() or d.is_symlink():
                continue
            epochs = find_epoch_dirs(d)
            if epochs:
                runs.append({
                    "name": d.name,
                    "path": str(d),
                    "epochs": epochs,
                    "source": "history",
                })

    # Sort all runs by directory modification time (oldest first)
    runs.sort(key=lambda r: os.path.getmtime(r["path"]))
    return runs


def generate_html(runs):
    """Generate the comparison HTML page."""
    html = ["""<!DOCTYPE html>
<html>
<head>
<title>Training Run Comparison</title>
<style>
body {
    font-family: monospace;
    background: #1a1a1a;
    color: #ddd;
    margin: 20px;
}
h1 { color: #fff; margin-bottom: 5px; }
.subtitle { color: #888; margin-bottom: 30px; }
.run {
    margin-bottom: 40px;
    border: 1px solid #333;
    padding: 15px;
    border-radius: 8px;
    background: #222;
}
.run-header {
    font-size: 14px;
    font-weight: bold;
    color: #4fc3f7;
    margin-bottom: 5px;
}
.run-path {
    font-size: 11px;
    color: #666;
    margin-bottom: 10px;
}
.run-meta {
    font-size: 12px;
    color: #999;
    margin-bottom: 10px;
}
.epoch-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 2px;
    gap: 2px;
}
.epoch-label {
    width: 55px;
    min-width: 55px;
    font-size: 11px;
    color: #aaa;
    padding-top: 8px;
    text-align: right;
    padding-right: 8px;
}
.epoch-images {
    display: flex;
    gap: 1px;
    flex-wrap: nowrap;
}
.epoch-images img {
    width: 80px;
    height: 80px;
    object-fit: contain;
    background: #fff;
    border: 1px solid #333;
}
.sample-headers {
    display: flex;
    gap: 1px;
    margin-left: 63px;
    margin-bottom: 4px;
}
.sample-headers span {
    width: 80px;
    min-width: 80px;
    font-size: 9px;
    color: #666;
    text-align: center;
    border: 1px solid transparent;
}
.separator {
    border-top: 2px solid #444;
    margin: 30px 0;
}
</style>
</head>
<body>
<h1>Training Run Comparison</h1>
<div class="subtitle">"""]

    html.append(f"{len(runs)} runs with tracking images</div>\n")

    # Sample headers (show once at top)
    html.append('<div class="sample-headers">')
    for s in SAMPLES:
        short = s.replace("Coffee_Milkshake_", "CM_").replace("Brown_Fox_", "BF_").replace("Buka_Bird_", "BB_")
        html.append(f'<span>{short}</span>')
    html.append('</div>\n')

    for i, run in enumerate(runs):
        epochs = run["epochs"]
        picked = pick_epochs(epochs, TARGET_EPOCHS)
        if not picked:
            continue

        max_epoch = max(epochs.keys())
        min_epoch = min(epochs.keys())

        html.append(f'<div class="run">')
        html.append(f'<div class="run-header">{run["name"]}</div>')
        html.append(f'<div class="run-path">{run["path"]}</div>')
        html.append(f'<div class="run-meta">Epochs: {min_epoch}-{max_epoch} ({len(epochs)} saved)</div>')

        for ep in picked:
            ep_dir = epochs[ep]
            html.append(f'<div class="epoch-row">')
            html.append(f'<div class="epoch-label">ep {ep}</div>')
            html.append(f'<div class="epoch-images">')

            for sample in SAMPLES:
                img_path = ep_dir / f"{sample}.png"
                if img_path.exists():
                    html.append(f'<img src="file://{img_path}" title="{sample} epoch {ep}">')
                else:
                    html.append(f'<img src="" style="visibility:hidden">')

            html.append('</div></div>\n')

        html.append('</div>\n')

        # Add separator between experiments and history
        if i < len(runs) - 1 and run["source"] != runs[i + 1]["source"]:
            html.append('<div class="separator"></div>\n')

    html.append('</body></html>')
    return '\n'.join(html)


if __name__ == "__main__":
    runs = discover_runs()
    print(f"Found {len(runs)} runs with tracking images:")
    for r in runs:
        epochs = r["epochs"]
        print(f"  {r['name']}: {len(epochs)} epochs ({min(epochs)}-{max(epochs)})")

    html = generate_html(runs)
    OUTPUT_HTML.write_text(html)
    print(f"\nWritten to {OUTPUT_HTML}")
    try:
        webbrowser.open(f"file://{OUTPUT_HTML}")
    except Exception:
        pass
