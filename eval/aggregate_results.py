#!/usr/bin/env python3
"""Aggregate per-case scores into summary.json / summary.csv + a failure gallery.

Reports the design-doc breakdowns (§10.4 / §11 / §16): overall + per-dimension,
by tier, by failure_tag, by edit_type, and the critical-failure rate. The gallery
surfaces the worst cases and groups failures by tier / tag.

  python scripts/aggregate_results.py \
    --case_scores outputs/eval_results/our_model_internal_case_scores.jsonl \
    --out_dir outputs/reports
"""
import argparse
import csv
import json
import os
from collections import defaultdict

DIMS = ["edit_success", "source_preservation", "temporal_consistency", "rendering_quality"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _group_means(recs, keyfn, val):
    g = defaultdict(list)
    for r in recs:
        for k in keyfn(r):
            g[k].append(val(r))
    return {k: _mean(v) for k, v in sorted(g.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case_scores", default="outputs/eval_results/our_model_internal_case_scores.jsonl")
    ap.add_argument("--out_dir", default="outputs/reports")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    recs = [json.loads(l) for l in open(args.case_scores)]
    fin = lambda r: r.get("final_score")

    def is_crit(r):
        return bool(r.get("caps_applied")) or any((r.get("critical_flags") or {}).values())

    summary = {
        "n_cases": len(recs),
        "overall_score": _mean([fin(r) for r in recs]),
        "dimension_scores": {d: _mean([(r.get("dimensions") or {}).get(d) for r in recs]) for d in DIMS},
        "tier_scores": _group_means(recs, lambda r: [r.get("tier")] if r.get("tier") else [], fin),
        "failure_tag_scores": _group_means(recs, lambda r: r.get("failure_tags", []), fin),
        "edit_type_scores": _group_means(recs, lambda r: [r.get("edit_type")] if r.get("edit_type") else [], fin),
        "critical_failure_rate": round(sum(is_crit(r) for r in recs) / len(recs), 4) if recs else None,
    }
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"),
              indent=2, ensure_ascii=False)

    # per-case CSV (design doc §16.2)
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "benchmark_source", "edit_type", "source_type", "tier",
                    "edit_success", "source_preservation", "temporal_consistency",
                    "rendering_quality", "final_score", "critical_failure", "caps"])
        for r in sorted(recs, key=lambda x: (x.get("final_score") is None, x.get("final_score", 0))):
            d = r.get("dimensions") or {}
            w.writerow([r["case_id"], r.get("benchmark_source"), r.get("edit_type"), "real",
                        r.get("tier"), d.get("edit_success"), d.get("source_preservation"),
                        d.get("temporal_consistency"), d.get("rendering_quality"),
                        r.get("final_score"), is_crit(r), ";".join(r.get("caps_applied", []))])

    _gallery(recs, os.path.join(args.out_dir, "failure_gallery.html"))

    # console print
    print("=== Benchmark Summary ===")
    print(f"cases={summary['n_cases']}  overall={summary['overall_score']}  "
          f"critical_failure_rate={summary['critical_failure_rate']}")
    print("dimensions:", summary["dimension_scores"])
    print("by tier:   ", summary["tier_scores"])
    print("by failure:", summary["failure_tag_scores"])
    print(f"\nwrote summary.json / summary.csv / failure_gallery.html -> {args.out_dir}")


def _gallery(recs, path):
    ok = [r for r in recs if r.get("final_score") is not None]
    worst = sorted(ok, key=lambda r: r["final_score"])[:10]

    def row(r):
        d = r.get("dimensions") or {}
        j = r.get("judge") or {}
        return (f"<tr><td>{r['case_id']}</td><td>{r.get('tier')}</td>"
                f"<td>{','.join(r.get('failure_tags', []))}</td>"
                f"<td><b>{r.get('final_score')}</b></td>"
                f"<td>ES {d.get('edit_success')} · SP {d.get('source_preservation')} · "
                f"TC {d.get('temporal_consistency')} · RQ {d.get('rendering_quality')}</td>"
                f"<td>{'; '.join(r.get('caps_applied', []))}</td>"
                f"<td>{(j.get('notes') or '')[:120]}</td></tr>")

    html = ["<html><head><meta charset='utf-8'><style>",
            "body{font-family:sans-serif;margin:20px} table{border-collapse:collapse;width:100%}",
            "td,th{border:1px solid #ccc;padding:6px;font-size:13px;text-align:left}",
            "h2{margin-top:28px}</style></head><body>",
            f"<h1>Failure Gallery ({len(recs)} cases)</h1>",
            "<h2>Top 10 lowest final scores</h2><table>",
            "<tr><th>case</th><th>tier</th><th>tags</th><th>final</th><th>dimensions</th>"
            "<th>caps</th><th>judge notes</th></tr>",
            *[row(r) for r in worst], "</table>"]
    for tier in ("easy", "medium", "hard"):
        sub = sorted([r for r in ok if r.get("tier") == tier], key=lambda r: r["final_score"])[:5]
        if sub:
            html += [f"<h2>Worst in tier: {tier}</h2><table>",
                     "<tr><th>case</th><th>tier</th><th>tags</th><th>final</th><th>dimensions</th>"
                     "<th>caps</th><th>judge notes</th></tr>", *[row(r) for r in sub], "</table>"]
    html.append("</body></html>")
    open(path, "w").write("\n".join(html))


if __name__ == "__main__":
    main()
