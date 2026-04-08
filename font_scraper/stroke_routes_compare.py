"""Route for comparing training runs (stroke model diagnostic view).

Serves a live HTML comparison of all runs with tracking images, auto-refreshes
as new runs/epochs appear. Scans both the experiments/ and history/ directories.
"""

import os
from pathlib import Path

from flask import send_file
from stroke_flask import app

STROKE_DIR = Path(__file__).parent / "docker" / "stroke_model"
EXPERIMENTS_DIR = STROKE_DIR / "experiments"
HISTORY_DIR = STROKE_DIR / "history"

SAMPLES = [
    "Brown_Fox_A", "Brown_Fox_R", "Brown_Fox_g", "Brown_Fox_8",
    "Coffee_Milkshake_A", "Coffee_Milkshake_R", "Coffee_Milkshake_g", "Coffee_Milkshake_8",
    "Buka_Bird_A", "Buka_Bird_R", "Buka_Bird_g", "Buka_Bird_8",
]

TARGET_EPOCHS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]


def _find_epoch_dirs(run_path: Path) -> dict:
    """Find epoch directories in a run, handling both formats."""
    tracking_dir = run_path / "tracking"
    base = tracking_dir if tracking_dir.is_dir() else run_path
    epochs = {}
    if not base.is_dir():
        return epochs
    for d in base.iterdir():
        if d.is_dir() and d.name.startswith("epoch_"):
            try:
                num = int(d.name.split("_")[1])
                if any(d.glob("*.png")):
                    epochs[num] = d
            except (ValueError, IndexError):
                continue
    return epochs


def _pick_epochs(available: dict, targets: list) -> list:
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


def _discover_runs():
    runs = []
    if EXPERIMENTS_DIR.is_dir():
        for d in sorted(EXPERIMENTS_DIR.iterdir()):
            if not d.is_dir() or d.name == "current":
                continue
            epochs = _find_epoch_dirs(d)
            if len(epochs) >= 5:
                runs.append({"name": d.name, "path": str(d), "epochs": epochs, "source": "experiments"})
    if HISTORY_DIR.is_dir():
        for d in sorted(HISTORY_DIR.iterdir()):
            if not d.is_dir() or d.is_symlink():
                continue
            epochs = _find_epoch_dirs(d)
            if epochs:
                runs.append({"name": d.name, "path": str(d), "epochs": epochs, "source": "history"})
    runs.sort(key=lambda r: os.path.getmtime(r["path"]))
    return runs


@app.route('/tracking_image/<path:relpath>')
def tracking_image(relpath):
    """Serve a tracking image by path relative to STROKE_DIR."""
    full = STROKE_DIR / relpath
    full = full.resolve()
    # Ensure path stays within STROKE_DIR
    if not str(full).startswith(str(STROKE_DIR.resolve())):
        return "forbidden", 403
    if not full.exists() or not full.is_file():
        return "not found", 404
    return send_file(str(full))


@app.route('/compare_runs')
def compare_runs():
    """Generate live comparison HTML of all training runs."""
    runs = _discover_runs()

    html = ['''<!DOCTYPE html>
<html>
<head>
<title>Training Run Comparison</title>
<meta http-equiv="refresh" content="60">
<script>window.onload = function() { window.scrollTo(0, document.body.scrollHeight); };</script>
<style>
body { font-family: monospace; background: #1a1a1a; color: #ddd; margin: 20px; }
h1 { color: #fff; margin-bottom: 5px; }
.subtitle { color: #888; margin-bottom: 20px; font-size: 12px; }
.run {
    margin-bottom: 40px; border: 1px solid #333; padding: 15px;
    border-radius: 8px; background: #222;
}
.run-header { font-size: 14px; font-weight: bold; color: #4fc3f7; margin-bottom: 5px; }
.run-path { font-size: 11px; color: #666; margin-bottom: 10px; }
.run-meta { font-size: 12px; color: #999; margin-bottom: 10px; }
.epoch-row { display: flex; align-items: flex-start; margin-bottom: 2px; gap: 2px; }
.epoch-label {
    width: 55px; min-width: 55px; font-size: 11px; color: #aaa;
    padding-top: 8px; text-align: right; padding-right: 8px;
}
.epoch-images { display: flex; gap: 1px; flex-wrap: nowrap; }
.epoch-images img {
    width: 80px; height: 80px; object-fit: contain;
    background: #fff; border: 1px solid #333;
}
.sample-headers {
    display: flex; gap: 1px; margin-left: 63px; margin-bottom: 4px; position: sticky;
    top: 0; background: #1a1a1a; padding: 5px 0; z-index: 10;
}
.sample-headers span {
    width: 80px; min-width: 80px; font-size: 9px; color: #666;
    text-align: center; border: 1px solid transparent;
}
.separator { border-top: 2px solid #444; margin: 30px 0; }
.auto-refresh { color: #4fc3f7; font-size: 11px; }
</style>
</head>
<body>
<h1>Training Run Comparison</h1>
<div class="subtitle">''']
    html.append(f'{len(runs)} runs · <span class="auto-refresh">auto-refreshes every 60s</span></div>\n')

    html.append('<div class="sample-headers">')
    for s in SAMPLES:
        short = s.replace("Coffee_Milkshake_", "CM_").replace("Brown_Fox_", "BF_").replace("Buka_Bird_", "BB_")
        html.append(f'<span>{short}</span>')
    html.append('</div>\n')

    for i, run in enumerate(runs):
        epochs = run["epochs"]
        picked = _pick_epochs(epochs, TARGET_EPOCHS)
        if not picked:
            continue
        max_epoch = max(epochs.keys())
        min_epoch = min(epochs.keys())

        html.append('<div class="run">')
        html.append(f'<div class="run-header">{run["name"]}</div>')
        html.append(f'<div class="run-path">{run["path"]}</div>')
        html.append(f'<div class="run-meta">Epochs: {min_epoch}-{max_epoch} ({len(epochs)} saved)</div>')

        for ep in picked:
            ep_dir = epochs[ep]
            rel_base = ep_dir.relative_to(STROKE_DIR)
            html.append('<div class="epoch-row">')
            html.append(f'<div class="epoch-label">ep {ep}</div>')
            html.append('<div class="epoch-images">')
            for sample in SAMPLES:
                img_rel = f"{rel_base}/{sample}.png"
                img_path = ep_dir / f"{sample}.png"
                if img_path.exists():
                    html.append(f'<img src="/tracking_image/{img_rel}" title="{sample} epoch {ep}">')
                else:
                    html.append('<img src="" style="visibility:hidden">')
            html.append('</div></div>\n')
        html.append('</div>\n')

        if i < len(runs) - 1 and run["source"] != runs[i + 1]["source"]:
            html.append('<div class="separator"></div>\n')

    html.append('</body></html>')
    return '\n'.join(html)
