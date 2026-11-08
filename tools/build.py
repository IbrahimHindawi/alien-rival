"""Build build/rival.exe (the engine) and build/simple.exe (the plain oracle) from scratch.

    python tools/build.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLANG = r'C:\Program Files\LLVM\bin\clang.exe'
FLAGS = ['-std=c17', '-O3', '-march=x86-64-v3', '-ffp-contract=off', '-fuse-ld=lld', '-Wall', '-Wno-unused-function', '-D_CRT_SECURE_NO_WARNINGS']


def build(src, out, extra=()):
    cmd = [CLANG, *FLAGS, *extra, str(ROOT / 'src' / src), '-o', str(ROOT / 'build' / out)]
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    (ROOT / 'build').mkdir(exist_ok=True)
    build('rival.c', 'rival.exe', ['-flto'])
    if '--no-simple' not in sys.argv:
        build('simple.c', 'simple.exe')


if __name__ == '__main__':
    main()
