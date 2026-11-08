"""Check build/rival.exe against the reference model and the recorded hashes.

    python tools/test.py            # everything
    python tools/test.py --quick    # small scenarios only

Each check prints one line; a check that cannot run counts as a failure.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / 'build' / 'rival.exe'
sys.path.insert(0, str(ROOT / 'tools'))
import sim_reference  # noqa: E402

SMALL = [  # (entities, ticks, seed, spawn_half): checked against the Python model
    (1000, 120, 1, 1000),
    (3000, 300, 5, 20),
    (2000, 600, 11, 8),
    (4000, 400, 7, 150),        # hero survives: live projectiles in the hash
    (4000, 700, 7, 150),        # every ability, twice
]
QUICK = SMALL[:2]
LARGE = [  # recorded hashes: too slow for the model
    (100000, 600, 1, 1000, 'ab8666237ff52a1c'),
    (1000000, 300, 1, 1000, '7a1a32542db8e5ab'),
    (1000000, 300, 1, 300, '9f1e580c29fd4f94'),
]


def hash_of(n, t, seed, half, threads):
    r = subprocess.run([str(EXE), '--bench-sim', '--entities', str(n), '--ticks', str(t), '--seed', str(seed),
                        '--spawn-half', str(half), '--threads', str(threads)], capture_output=True, text=True, timeout=900)
    got = next((l.strip().split('=', 1)[1] for l in r.stdout.splitlines() if l.startswith('hash=')), None)
    return r.returncode, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    failures = 0

    def check(name, ok, detail=''):
        nonlocal failures
        print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail and not ok else ''}", flush=True)
        failures += 0 if ok else 1

    if not EXE.is_file():
        check('executable exists', False, str(EXE))
        sys.exit(1)
    for n, t, seed, half in (QUICK if a.quick else SMALL):
        s = sim_reference.Sim(n, seed, float(half))
        for _ in range(t):
            s.bench_step()
        want = f'{s.hash():016x}'
        for threads in (1, 16):
            code, got = hash_of(n, t, seed, half, threads)
            check(f'model n={n} ticks={t} seed={seed} spawn_half={half} threads={threads}',
                  code == 0 and got == want, f'rival {got} model {want}')
    if not a.quick:
        for n, t, seed, half, want in LARGE:
            for threads in (1, 16):
                code, got = hash_of(n, t, seed, half, threads)
                check(f'recorded n={n} ticks={t} spawn_half={half} threads={threads}',
                      code == 0 and got == want, f'rival {got} recorded {want}')
    print(f'{failures} failure(s)')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
