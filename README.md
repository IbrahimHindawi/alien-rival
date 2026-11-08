# alien-rival: the simulation in C, as fast as it can go

This repo is one side of a contest. The task is to implement the swarm
simulation in `spec/sim_spec.md` (version SIM 5) in C, and make it as fast as
possible on a generic x86-64 CPU with AVX2. Another implementation exists;
you do not see it. The two are compared on the same machine with the same
command line, and the only correctness gate is the state hash.

## What is here

- `spec/sim_spec.md`: the contract. Every rule, constant, iteration order and
  the hash. Where it mentions "C++", read "C". Section 6 (rendering instances)
  is not part of the contest.
- `tools/sim_reference.py`: an exact Python model of the spec. Slow; meant for
  small populations. It is the referee: where it and the spec seem to differ,
  the model wins, and say so in NOTES.md.
- `tools/test.py`: checks `build/rival.exe` against the model on small
  scenarios and against recorded hashes on the large ones, at 1 and 16
  threads. Every check must pass.
- `tools/bench.py`: the benchmark both sides are measured with.

## Rules

1. **Language and compiler:** C (C17 or C23), clang (`C:\Program Files\LLVM\bin\clang`,
   version 22). Flags: `-O3 -march=x86-64-v3 -ffp-contract=off`, no `-ffast-math`
   or any flag that relaxes IEEE semantics, no FTZ/DAZ. LTO and PGO are allowed.
   AVX2 intrinsics and inline assembly are **not** the same thing: intrinsics are
   allowed, inline assembly and separate `.asm` files are not. The point is what
   clang makes of C.
2. **Generic target:** nothing that needs more than x86-64-v3 (no AVX-512), no
   thread pinning to specific cores, no tuning constants read from this
   machine's cache sizes. Code must run correctly on any AVX2 CPU.
3. **Exactness:** the state hash must equal the model's, bit for bit, at any
   thread count. Nothing may be approximated or skipped; spec section 7 says
   what runs per entity per tick.
4. **Command line** (exact, the bench parses it):

        build/rival.exe --bench-sim --entities N --ticks T --seed S --spawn-half H --threads K

   `--spawn-half` defaults to 1000, `--seed` to 1, `--threads` to the number
   of logical processors. It runs the bench input (spec 4.8) for T ticks and
   prints at least these lines:

        us_per_tick=<integer microseconds, wall clock, ticks only, not init>
        hash=<16 lowercase hex digits>

5. **Threads:** your own pool on Win32 (`CreateThread`, `WaitOnAddress`,
   whatever you like) or C11 threads; no OpenMP. The C runtime is allowed.
6. **Build:** `python tools/build.py` must produce `build/rival.exe` from
   scratch. Keep it one command.
7. **Do not** read anything outside this directory. In particular not
   `C:\devel\alien`. Do not search the disk for other implementations.

## How you are judged

`python tools/bench.py` runs three scenarios: 100k and 1M entities, sparse
(`--spawn-half 1000`) and a dense 1M (`--spawn-half 300`), each at 1 and 16
threads, median of 5 fresh processes after a warmup. Lower us per tick wins
each cell. A failing hash disqualifies the run.

## Working notes

Keep `NOTES.md`: the design, every optimisation tried with its before and
after numbers, and what was kept.
