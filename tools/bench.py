"""The contest benchmark. Drives any executable with the contest command line.

    python tools/bench.py                          # build/rival.exe, all scenarios
    python tools/bench.py --exe path/to/other.exe  # the same matrix for another engine
    python tools/bench.py --runs 3 --only 1000000:300

Each run is a fresh process; one warmup is discarded; median and min..max of
us per tick are reported. Every run of a cell must print the same hash, and
it must equal the recorded one where there is one.
"""
import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = [(100000, 600, 1000), (1000000, 300, 1000), (1000000, 300, 300)]
THREADS = [1, 16]
RECORDED = {(100000, 600, 1000): 'ab8666237ff52a1c', (1000000, 300, 1000): '7a1a32542db8e5ab',
            (1000000, 300, 300): '9f1e580c29fd4f94'}


def run_once(exe, n, ticks, half, threads):
    out = subprocess.run([str(exe), '--bench-sim', '--entities', str(n), '--ticks', str(ticks), '--seed', '1',
                          '--spawn-half', str(half), '--threads', str(threads)],
                         capture_output=True, text=True, timeout=1800).stdout
    us = int(re.search(r'us_per_tick=(\d+)', out).group(1))
    h = re.search(r'hash=([0-9a-f]{16})', out).group(1)
    return us, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', default=str(ROOT / 'build' / 'rival.exe'))
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--only', help='N:TICKS[:HALF]')
    a = ap.parse_args()
    scen = SCENARIOS
    if a.only:
        p = [int(x) for x in a.only.split(':')]
        scen = [(p[0], p[1], p[2] if len(p) > 2 else 1000)]
    bad = 0
    for n, ticks, half in scen:
        for th in THREADS:
            run_once(a.exe, n, ticks, half, th)
            rs = [run_once(a.exe, n, ticks, half, th) for _ in range(a.runs)]
            hs = {h for _, h in rs}
            want = RECORDED.get((n, ticks, half))
            ok = len(hs) == 1 and (want is None or want in hs)
            bad += not ok
            us = [u for u, _ in rs]
            print(f'n={n:>8} half={half:>5} threads={th:>2}: {statistics.median(us):>8} us/tick '
                  f'(min {min(us)}, max {max(us)})  hash {"/".join(sorted(hs))} {"OK" if ok else "WRONG"}', flush=True)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
