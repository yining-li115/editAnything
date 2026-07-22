#!/usr/bin/env python3
"""Plot per-stage pipeline timing from jobs/*/timing.log — real logged data only.

Reads every jobs/<id>/timing.log (written by orchestrator.py), sums each stage's
duration (retries summed), EXCLUDES evaluate, and writes a stacked bar chart of
seconds-per-stage vs frame count. No projection — only jobs that actually ran.

Usage:
    python plot_timing.py                       # -> timing_plot.png
    python plot_timing.py --out demo_timing.png --jobs jobs
"""
import os
import glob
import argparse
import collections

# Stages shown as their own colour; everything else folds into "other".
COLORS = {
    "sam3_mask":             "#2a78d6",
    "roma_edit_mask":        "#1baf7a",
    "videopainter_generate": "#eda100",
    "composite":             "#4a3aa7",
    "other":                 "#898781",
}
STAGES = list(COLORS)


def parse_timing(path):
    """timing.log lines are 'HH:MM:SS  <stage>  <N>s' -> {stage: total_seconds}."""
    times = collections.defaultdict(float)
    for line in open(path):
        parts = line.split()
        if len(parts) < 3 or not parts[-1].endswith("s"):
            continue
        try:
            times[parts[1]] += float(parts[-1][:-1])
        except ValueError:
            continue
    return times


def frame_count(job_dir):
    return len(glob.glob(os.path.join(job_dir, "frames", "frame_*.png")))


def collect(jobs_root):
    """Return sorted [(frames, job_id, {stage: seconds}), ...] for COMPLETE runs."""
    rows = []
    for tl in sorted(glob.glob(os.path.join(jobs_root, "*", "timing.log"))):
        job = os.path.dirname(tl)
        n = frame_count(job)
        t = parse_timing(tl)
        # a real run must have generated frames (skip junk / partial jobs)
        if n == 0 or t.get("videopainter_generate", 0) < 5:
            continue
        grouped = {s: 0.0 for s in STAGES}
        for stage, sec in t.items():
            if stage == "evaluate":
                continue
            grouped[stage if stage in COLORS else "other"] += sec
        rows.append((n, os.path.basename(job), grouped))
    rows.sort(key=lambda r: r[0])
    return rows


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.1), 5))
    for stage in STAGES:
        vals = np.array([g[stage] for _, _, g in rows])
        ax.bar(x, vals, bottom=bottom, label=stage, color=COLORS[stage], width=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 10:
                c = "#412402" if stage == "videopainter_generate" else "white"
                ax.text(xi, b + v / 2, f"{v:.0f}s", ha="center", va="center", fontsize=8, color=c)
        bottom += vals
    for xi, tot in enumerate(bottom):
        ax.text(xi, tot + 4, f"{tot/60:.1f} min", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}f" for n, _, _ in rows])
    ax.set_ylabel("seconds (excl. evaluate)")
    ax.set_xlabel("frames per clip")
    ax.set_title("Pipeline timing per job — real logged data")
    ax.legend(fontsize=8, ncol=len(STAGES), loc="lower left", frameon=False, bbox_to_anchor=(0, 1.02))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  ({len(rows)} jobs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", default="jobs", help="jobs/ dir to scan")
    ap.add_argument("--out", default="timing_plot.png", help="output PNG path")
    args = ap.parse_args()

    rows = collect(args.jobs)
    if not rows:
        raise SystemExit("no complete jobs with timing.log found")
    for n, jid, g in rows:
        print(f"  {n:4d}f  {jid}  total(excl.eval)={sum(g.values()):.0f}s")
    plot(rows, args.out)
