#!/usr/bin/env python3
"""
Benchmark sweep over every NN implementation in the repo.

Runs each model on each dataset across a grid of hidden sizes and batch
sizes, repeats every test -n times and reports the average wall time
(plus min/max/std) in comparison tables at the end.

The unoptimized serial version has no batching, so it runs once per
(dataset, hidden) and shows up with batch '-'.

Examples:
    ./benchmark.py -n 3
    ./benchmark.py -n 5 --datasets digits --hidden 128 --batches 64 128
    ./benchmark.py -n 1 --epochs 2 --skip unoptimized-serial-nn
"""

import argparse
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# name -> needs batch args
MODELS = {
    "unoptimized-serial-nn": False,
    "optimized-serial-nn": True,
    "parallel-omp-nn": True,
    "parallel-cblas-nn": True,
    "parallel-cuda-nn": True,
}

ACC_RE = re.compile(r"test accuracy: ([0-9.]+)")
LOSS_RE = re.compile(r"test loss: ([0-9.]+)")

if sys.stdout.isatty():
    BOLD, DIM, GREEN, RED, YELLOW, CYAN, RESET = (
        "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m")
else:
    BOLD = DIM = GREEN = RED = YELLOW = CYAN = RESET = ""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--runs", type=int, default=3,
                   help="times each test is run, reported time is the average (default 3)")
    p.add_argument("--datasets", nargs="+", default=["digits", "letters", "byclass"],
                   choices=["digits", "letters", "byclass"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--hidden", nargs="+", type=int, default=[128, 512])
    p.add_argument("--batches", nargs="+", type=int, default=[32, 128, 512])
    p.add_argument("--eval-batch", type=int, default=100)
    p.add_argument("--timeout", type=float, default=900,
                   help="per-run timeout in seconds, timed out runs show as DNF (default 900)")
    p.add_argument("--skip", nargs="+", default=[], choices=list(MODELS),
                   help="models to leave out")
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def build():
    print(f"{DIM}building...{RESET} ", end="", flush=True)
    r = subprocess.run(["make", "-s", "-C", str(ROOT / "build")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{RED}failed{RESET}\n{r.stderr}")
        sys.exit(1)
    print(f"{GREEN}ok{RESET}")


def run_once(cmd, timeout):
    """-> (seconds, accuracy, loss) or None on failure/timeout"""
    start = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return None
    elapsed = time.perf_counter() - start
    if r.returncode != 0:
        return None
    acc = ACC_RE.search(r.stdout)
    loss = LOSS_RE.search(r.stdout)
    if not acc:
        return None
    return elapsed, float(acc.group(1)), float(loss.group(1)) if loss else math.nan


class Result:
    def __init__(self, times, accs):
        self.times, self.accs = times, accs

    @property
    def ok(self):
        return len(self.times) > 0

    @property
    def avg(self):
        return statistics.mean(self.times)

    @property
    def std(self):
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def acc(self):
        return statistics.mean(self.accs)


def sweep(args):
    models = [m for m in MODELS if m not in args.skip and (ROOT / m).exists()]
    missing = [m for m in MODELS if m not in args.skip and not (ROOT / m).exists()]
    for m in missing:
        print(f"{YELLOW}warning:{RESET} {m} not built, skipping")

    # unoptimized ignores batch size -> one run per (dataset, hidden)
    tests = []
    for ds in args.datasets:
        for h in args.hidden:
            for m in models:
                if MODELS[m]:
                    for bs in args.batches:
                        tests.append((ds, h, bs, m))
                else:
                    tests.append((ds, h, None, m))

    total_runs = len(tests) * args.runs
    print(f"{len(tests)} configs x {args.runs} runs = {total_runs} total runs\n")

    results = {}
    done = 0
    for ds, h, bs, m in tests:
        cmd = [str(ROOT / m), ds, str(args.epochs), str(args.lr), str(h)]
        if bs is not None:
            cmd += [str(bs), str(args.eval_batch)]

        label = f"{ds:8s} hidden={h:<4d} bs={'-' if bs is None else bs:<4} {m}"
        print(f"  {label:70s}", end="", flush=True)

        times, accs = [], []
        for _ in range(args.runs):
            out = run_once(cmd, args.timeout)
            done += 1
            if out is None:
                print(f"{RED}x{RESET}", end="", flush=True)
                break  # timed out / crashed once, no point repeating it
            times.append(out[0])
            accs.append(out[1])
            print(f"{DIM}.{RESET}", end="", flush=True)

        res = Result(times, accs)
        results[(ds, h, bs, m)] = res
        if res.ok:
            print(f"  {BOLD}{res.avg:8.2f}s{RESET} {DIM}±{res.std:.2f}  "
                  f"acc {res.acc:.4f}  [{done}/{total_runs}]{RESET}")
        else:
            print(f"  {RED}DNF{RESET}")
    return results, models


def fmt_cell(res, best):
    if res is None:
        return "-"
    if not res.ok:
        return f"{RED}DNF{RESET}"
    s = f"{res.avg:.2f}s ±{res.std:.2f} ({res.acc * 100:.1f}%)"
    if best:
        return f"{GREEN}{BOLD}{s}{RESET}"
    return s


def visible_len(s):
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def pad(s, w):
    return s + " " * (w - visible_len(s))


def print_table(title, headers, rows):
    print(f"\n{BOLD}{YELLOW}{title}{RESET}")
    widths = [max(visible_len(r[i]) for r in [headers] + rows) for i in range(len(headers))]
    line = "-+-".join("-" * w for w in widths)
    print("  " + " | ".join(pad(h, w) for h, w in zip(headers, widths)))
    print("  " + line)
    for r in rows:
        print("  " + " | ".join(pad(c, w) for c, w in zip(r, widths)))


def report(results, models, args):
    short = {m: m.replace("-nn", "") for m in models}

    for ds in args.datasets:
        for h in args.hidden:
            headers = ["batch"] + [short[m] for m in models]
            rows = []
            batch_rows = [None] + args.batches if any(not MODELS[m] for m in models) else args.batches
            for bs in batch_rows:
                cells = {}
                for m in models:
                    key = (ds, h, None if not MODELS[m] else bs, m)
                    r = results.get(key) if (MODELS[m]) == (bs is not None) else None
                    cells[m] = r
                if all(c is None for c in cells.values()):
                    continue
                oks = [c for c in cells.values() if c is not None and c.ok]
                fastest = min((c.avg for c in oks), default=None)
                row = ["-" if bs is None else str(bs)]
                for m in models:
                    c = cells[m]
                    row.append(fmt_cell(c, c is not None and c.ok and c.avg == fastest))
                rows.append(row)
            print_table(f"{ds}  (hidden={h}, epochs={args.epochs}, lr={args.lr}, "
                        f"avg of {args.runs})", headers, rows)

    # overall: geometric mean speedup vs optimized-serial on shared configs
    base = "optimized-serial-nn"
    headers = ["model", "geomean speedup vs optimized-serial", "fastest in", "avg acc"]
    rows = []
    for m in models:
        ratios, wins, accs, entered = [], 0, [], 0
        for ds in args.datasets:
            for h in args.hidden:
                for bs in args.batches:
                    key = (ds, h, bs if MODELS[m] else None, m)
                    bkey = (ds, h, bs, base)
                    r, b = results.get(key), results.get(bkey)
                    if r is None or not r.ok:
                        continue
                    entered += 1
                    accs.append(r.acc)
                    if b is not None and b.ok:
                        ratios.append(b.avg / r.avg)
                    rivals = [results.get((ds, h, bs if MODELS[x] else None, x)) for x in models]
                    rivals = [x for x in rivals if x is not None and x.ok]
                    if rivals and r.avg == min(x.avg for x in rivals):
                        wins += 1
        geo = math.exp(statistics.mean(map(math.log, ratios))) if ratios else math.nan
        rows.append([short[m],
                     f"{geo:.2f}x" if not math.isnan(geo) else "-",
                     f"{wins}/{entered}",
                     f"{statistics.mean(accs) * 100:.2f}%" if accs else "-"])
    print_table("summary", headers, rows)


def main():
    args = parse_args()
    if not args.skip_build:
        build()
    t0 = time.perf_counter()
    results, models = sweep(args)
    report(results, models, args)
    print(f"\n{DIM}total benchmark time: {time.perf_counter() - t0:.0f}s{RESET}")


if __name__ == "__main__":
    main()
