#!/usr/bin/env python3
"""Aggregate per-case results into the Benchmark Summary Table.

    python eval/aggregate.py --results eval/results

Writes summary.md (Markdown table + averages) and summary.csv next to the
results dir, and prints the table.
"""
import argparse
import csv
import glob
import json
import os


def fmt(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results, "*.json")))
    if not paths:
        ap.error(f"no result JSONs in {args.results}")

    recs = []
    for p in paths:
        with open(p) as f:
            recs.append(json.load(f))

    cols = ["clip_score", "temporal_consistency", "dino_score", "final_score"]
    rows = []
    for r in recs:
        vlm = r.get("vlm_judge") or {}
        rows.append({
            "video_id": r.get("video_id", ""),
            "object_prompt": r.get("object_prompt", ""),
            "replace_prompt": r.get("replace_prompt", ""),
            "model": r.get("model", ""),
            "clip_score": r.get("clip_score"),
            "temporal_consistency": r.get("temporal_consistency"),
            "dino_score": r.get("dino_score"),
            "vlm_overall": vlm.get("overall"),
            "final_score": r.get("final_score"),
        })

    def avg(key):
        vals = [row[key] for row in rows if row.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    averages = {k: avg(k) for k in
                ["clip_score", "temporal_consistency", "dino_score", "vlm_overall", "final_score"]}

    # ---- Markdown ----
    header = "| Video | Object | Replace Prompt | CLIP | Temp. Cons. | DINO | VLM | Final |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["video_id"], row["object_prompt"], row["replace_prompt"],
            fmt(row["clip_score"]), fmt(row["temporal_consistency"]),
            fmt(row["dino_score"]), fmt(row["vlm_overall"], 2), fmt(row["final_score"])))
    lines.append("| **Average** |  |  | {} | {} | {} | {} | {} |".format(
        fmt(averages["clip_score"]), fmt(averages["temporal_consistency"]),
        fmt(averages["dino_score"]), fmt(averages["vlm_overall"], 2),
        fmt(averages["final_score"])))
    table = "\n".join(lines)

    out_md = args.out_md or os.path.join(os.path.dirname(args.results.rstrip("/")), "summary.md")
    with open(out_md, "w") as f:
        f.write("# Benchmark Summary\n\n" + table + "\n")

    # ---- CSV ----
    out_csv = args.out_csv or os.path.join(os.path.dirname(args.results.rstrip("/")), "summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(table)
    print(f"\n[aggregate] {len(rows)} cases -> {out_md}, {out_csv}")


if __name__ == "__main__":
    main()
