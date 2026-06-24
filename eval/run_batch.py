#!/usr/bin/env python3
"""Run many pipeline.py jobs concurrently with a fixed worker pool.

A Python process pool (not bash `& ... wait`) so each job is independent and
reliable. Each job runs `python pipeline.py --config <cfg> [--resume]`; stdout+err
go to <log-dir>/<name>.log. Prints a live summary and a final pass/fail table.

    python eval/run_batch.py --configs eval/configs/*.yaml --workers 3
    python eval/run_batch.py --configs eval/configs/cup0_*.yaml --workers 2 --resume

--workers: concurrent jobs. With sequential CPU offload (~22GB/run) 3 fit on an
80GB GPU; raise only if you've measured headroom.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def expand(patterns):
    paths = []
    for p in patterns:
        paths += glob.glob(os.path.join(p, "*.yaml")) if os.path.isdir(p) else glob.glob(p)
    return sorted(set(paths))


def name_of(cfg):
    return os.path.splitext(os.path.basename(cfg))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True, help="config files / globs / dir")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--log-dir", default=os.path.join(HERE, "run_logs"))
    ap.add_argument("--resume", action="store_true", help="pass --resume to pipeline.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-done", action="store_true",
                    help="skip configs whose outputs/<name>/final.mp4 already exists")
    args = ap.parse_args()

    cfgs = expand(args.configs)
    if not cfgs:
        ap.error("no configs matched")
    os.makedirs(args.log_dir, exist_ok=True)

    if args.skip_done:
        kept = [c for c in cfgs
                if not os.path.exists(os.path.join(REPO, "outputs", name_of(c), "final.mp4"))]
        print(f"[batch] {len(cfgs)-len(kept)} already done, {len(kept)} to run")
        cfgs = kept

    print(f"[batch] {len(cfgs)} jobs, {args.workers} workers, logs -> {args.log_dir}")
    pending = list(cfgs)
    running = {}          # popen -> (name, t0, logfile)
    done = []             # (name, rc, secs)
    t_start = time.time()

    def launch(cfg):
        name = name_of(cfg)
        logf = open(os.path.join(args.log_dir, f"{name}.log"), "w")
        cmd = [args.python, os.path.join(REPO, "pipeline.py"), "--config", cfg]
        if args.resume:
            cmd.append("--resume")
        p = subprocess.Popen(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
        running[p] = (name, time.time(), logf)
        print(f"  [start] {name}  ({len(running)} running, {len(pending)} queued)")

    while pending or running:
        while pending and len(running) < args.workers:
            launch(pending.pop(0))
        time.sleep(3)
        for p in list(running):
            rc = p.poll()
            if rc is None:
                continue
            name, t0, logf = running.pop(p)
            logf.close()
            secs = round(time.time() - t0)
            done.append((name, rc, secs))
            tag = "OK " if rc == 0 else f"FAIL({rc})"
            print(f"  [{tag}] {name}  {secs}s  ({len(done)}/{len(cfgs)} done)")

    total = round(time.time() - t_start)
    ok = sum(1 for _, rc, _ in done if rc == 0)
    print(f"\n[batch] {ok}/{len(done)} succeeded in {total}s ({total/60:.1f} min)")
    fails = [(n, rc) for n, rc, _ in done if rc != 0]
    if fails:
        print("[batch] FAILURES:")
        for n, rc in fails:
            print(f"    {n} (rc={rc})  -> {os.path.join(args.log_dir, n+'.log')}")


if __name__ == "__main__":
    main()
