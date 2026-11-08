# rival: SIM 5 in C, notes

Build: `python tools/build.py` (clang 22, `-O3 -march=x86-64-v3 -ffp-contract=off -flto`).
Produces `build/rival.exe` (the engine, `src/rival.c`) and `build/simple.exe`
(`src/simple.c`, a plain single-threaded transcription of the spec kept as a
fast oracle for debugging: it matches the Python model and the recorded hashes).

Profiling: `RIVAL_PROF=1 build/rival.exe ...` prints time per phase.
`RIVAL_BANDS=k` overrides the number of row bands per tick.

Spec vs model: no difference found; both engines (`simple.c` and `rival.c`)
follow the model where the spec is terse (e.g. a projectile with no pierce
left skips its second half-step's hit test; contact uses `dash_ticks` after
the hero's move of the tick).

## Reference points

`simple.exe` (spec transcription, index order, one thread): 100k 2913 us/tick,
1M sparse 75577 us/tick, 1M dense 254286 us/tick.

## Design (final)

See "Final design" at the end; the log below is in chronological order.

## Optimisation log

All numbers are median us/tick from `python tools/bench.py --runs 3` unless
noted; columns: 100k/1T, 100k/16T, 1M sparse/1T, 1M sparse/16T, 1M dense/1T,
1M dense/16T.

### v1: sorted layout, counting sort per tick, scalar-ish separation

Entities are stored in grid order (cell, then creation sequence) instead of
index order. The spec's index order is creation order (stable compaction,
append-only spawns), so the hash is taken at the end by sorting the live
entities by sequence number. The grid is rebuilt each tick from the previous
grid order: each thread owns a band of rows and scans the old rows next to
it; cells are re-sorted by sequence when a mover broke the order.
Separation: per entity, 3 row segments of 8-wide candidate chunks, hits
summed in order by a bit loop. Area effects (pulse, beam, bomb) scan only
the cells their reach can touch, grown by the largest displacement of the
tick. Projectiles/contact/abilities/deaths/spawns are serial (at most 15
projectiles exist at once: fire every 4 ticks, life 60).

    2490  306  16136  1663  74822  9225

### v2: separation with lanes = entities

Lanes are 8 consecutive entities of one row; every candidate of the union
window (rows z-1..z+1, cells from the first lane's cx-1 to the last lane's
cx+1) is broadcast to all lanes. A candidate outside a lane's 3x3 block is
>= 2 m away on one axis, so d2 >= 4 > 1: it never contributes, and each lane
sees exactly its spec neighbours in spec order. Branchless; masked lanes add
+-0.0f, which is exact because the running sum starts at +0.0f and can
never be -0.0f. Dense separation went from ~320 to ~130 cycles per entity
(the old version was bound by branch mispredictions in the hit loop).
Fix-up of the merge restricted to descents within a cell (it was tripping on
every cell boundary).

    ~1601  ~316  ~14958  ~1865  ~35201  ~3559   (single runs, RIVAL_PROF)

### v3: hp by sequence, no per-slot cell array

hp lives in an array indexed by creation sequence (only combat touches it),
so the grid rebuild moves 5 arrays instead of 6. Dead entities get NaN
positions instead of a separate flag array.

    ~1328  ~260  ~12709  ~1552  ~34713  ~3232   (single runs)

### v4: 16-lane groups, lazy cap

Two vectors share each broadcast candidate (half the broadcast cost per
lane, and a 16-entity group spans fewer cells per lane in dense areas). The
16-neighbour cap is not applied unless a lane exceeds 16 hits, in which case
the group is redone with the cap; a group that needed it starts the next
group capped.

    ~1308  ~240  ~12077  ~1506  ~25597  ~2643   (single runs)

### v5: integration counts the next grid (tried, partly kept)

The integration counts, per cell, the entities that stay in their row, and
lists the row changers; cells that took in an entity from another cell are
flagged (bit 31 of the sequence) so only those are re-sorted. Alone this was
slower (13720 / 1941 on 1M sparse): the extra work moved into the
separation pass and every thread scanned every mover list with random
reads. Kept as the basis of v6.

### v6: fused grid rebuild + separation, row buffers

Memory bandwidth measured on this machine: ~36 GB/s one thread, ~46 GB/s
(DRAM) to ~90 GB/s (partly L3) with 16 threads. The 16-thread tick was moving
~140 MB per tick for 1M entities: memory bound. Now each band of rows
rebuilds its rows of the new grid order (plus one neighbour row each side)
into three small per-thread row buffers that stay in L2, and separates and
integrates row r as soon as row r+1 is built. The grid-ordered start-of-tick
arrays never exist in memory; one fused parallel pass and one serial section
per tick (two barriers). Bands are taken from an atomic counter; 4 bands per
thread on 1M (balanced by entity count) beat 1 band per thread by 15-20% at
16 threads; balancing by measured row time was worse (noisy).

    1536  251  12672  1364  26569  2775

### v7-v15: one step at a time (A/B: min of 3 alternating runs per binary)

Run-to-run noise on this machine is about +-5% between sessions, so each step
was measured against the previous binary in alternating runs
(`ab.sh`, 100 ticks for the 1-thread 1M cells). Single-thread cycle
attribution with rdtsc (1M sparse): grid rebuild ~16 cycles/entity,
separation + integration ~37 cycles/entity.

- Vectorised prefix sum of a row's cell counts; 8 stayers copied with one
  vector move per array when their new slots are consecutive. 100k 1T -12%,
  others +-1%. Kept.
- Group width: 8 * G lanes, G = 2, 3, 4 (two, three, four vectors sharing
  each broadcast). G = 3 and 4 were 15-40% slower (register spills). G = 2 kept.
- Integer compare for `d2 < 1` (the bits of a non-negative float order like
  integers), to move the compare off the FP add pipes: no change. Dropped.
- Early exit when all 16 lanes have hit the cap: 15% slower (it stopped the
  compiler unrolling the candidate loops). Dropped.
- Row rebuild as a merge: entities that kept their cell are left-packed
  (pext/pdep permutation) in old order, which is already (cell, sequence)
  order; the few cell changers are merged in. First with a sort + binary
  search per inserted entity (no gain: 10 Mcycles for 78k inserts), then
  with the insert position computed from the cell start (1M sparse -3.6%),
  then with the changers bucketed per cell (linked lists + a 1024-bit map)
  instead of sorted (1M sparse -5.6%, 16T -6%). Kept.
- Software prefetch of the next row's old data: no change (the streams are
  sequential and the hardware prefetchers already cover them). Dropped.
- Streaming (non-temporal) stores for the tick output: 4-18% slower
  everywhere; the output is read back next tick and stays in L3 otherwise.
  Dropped.
- Cell counts of the next layout start as this layout's counts (vector
  difference of cell starts); only cell changers adjust them, instead of 8
  scalar increments per 8 entities. Dense -1.5%, 100k 1T -9%. Kept.
- Integration of a whole row after its separation, instead of after each
  group (independent iterations let the sqrt/div chains overlap). ~1%. Kept.
- Per-half candidate windows: a 16-lane group spans ~16 cells in sparse
  areas, but each 8-lane half only needs the cells next to its own lanes.
  Each window row splits into A-only / both / B-only runs. 1M dense 1T -4%,
  100k 1T -8%, sparse -2%. Kept.
- No self test in the uncapped pass: a lane's own slot is a d2 = 0 hit
  that adds +0.0f (exact) and one to its count, so the uncapped pass
  checks `n > 17` instead and the middle row is no longer split around the
  group's own slots. Sparse 1T -4.5%, dense -2%. Kept.
- Processing a window row as one union run when the side runs are short:
  1-3% slower. Dropped.

After v15 (`bench.py --runs 3`):

    1089  174  10578  1085  23189  2237

### v16-v18: threads

Thread scaling, 1M sparse: 4T 2924, 8T 1661, 16T 1104, 24T 892 us/tick;
per-entity time is the same for 250k, 500k and 1M at 16 threads, so the
16-thread runs are not DRAM-bound at this point. Per thread at 16T an
entity costs ~16 ns against ~11 ns single-threaded (all-core clock, threads
migrating over both CCDs and SMT siblings: Windows moves them constantly;
pinning is not allowed). Wall vs busy time of the fused phase: imbalance
13% (1M) to 28% (100k, one band per thread).

- Barrier: spin, then `SwitchToThread` after ~32k pauses. With 32 threads
  on 32 logical processors the pure spin barrier took 24 ms/tick instead of
  ~1 ms (a descheduled thread stalls everyone). No change at 16 threads. Kept.
- Removing the per-cell stayer counts (deriving the next counts from the
  old cell starts and the leavers, found by walking the old cell starts):
  10-20% slower (the walk). Dropped.
- Bands ordered by last tick's measured cost (largest first): 15% slower
  on 1M sparse 16T (locality: bands in row order share neighbour rows and
  old data in L3). Dropped. Bands split by measured row time instead of
  entity count: also slower (noisy timings). Dropped.
- Guided band sizes for large populations (each band takes 1/(2T) of the
  remaining entities, minimum 1/(8T) of all), still handed out in row
  order: the last bands are small, so the end of the phase is balanced.
  1M dense 16T -5%, 1M sparse 16T -3%. Small populations keep one band per
  thread (more bands cost more rebuilt neighbour rows than they balance).
  Kept.
- Worker threads at above-normal priority: within noise, kept (harmless,
  fewer preemptions by other processes).

After v17 (`bench.py --runs 3`):

    1162  203  10962  1066  23591  2187

### v19-v22: group width per row, work stealing with shared boundary rows

- Initial state sorted by cell at init (stable, so sequence order within a
  cell): tick 0 then sees every entity as a stayer instead of rebuilding the
  grid from a list of one million "movers" (which every band had to scan).
  Within noise on the totals, simpler. Kept.
- Group width chosen per row: 8 lanes when the row has fewer than 4
  entities per occupied cell, 16 otherwise. In sparse rows the two halves of
  a 16-lane group share almost no candidates, and 8-lane groups have less
  overhead per group. 1M sparse 1T -4%, dense unchanged. Kept.
- Work stealing with carried boundary rows. Each thread starts on its own
  contiguous run of bands and goes through it in row order; when the next
  band starts where the last one ended, the two boundary rows are still in
  the thread's row buffers and are not rebuilt (the shared row is published
  to the global arrays then). A thread that runs out takes the upper half of
  the largest remaining run (one 64-bit CAS on front/back). With the
  rebuild of neighbour rows gone, bands can be small, so the end of the phase
  balances well. Guided band sizes dropped (they fought the contiguous runs:
  dense 16T +14%).
- Row changers sorted by destination row at the end of each thread's pass
  (counting sort by row, per-thread offset table): a band reads only the
  movers of its rows instead of filtering every thread's whole list, which
  had made many bands expensive. Spawns go to their own small list.
- Band count: N/4000 clamped to [4T, 32T] (sweep at 16T: 1M best at
  128-512, 100k at 64).

A/B against v20 (min of 3): 100k 16T 135 -> 115, 1M sparse 16T 958 -> 764,
1M dense 16T 2097 -> 1926; 1-thread unchanged.

After v22 (`bench.py --runs 3`, test.py 0 failures):

    1076  120  10083  777  23251  1965

### v23-v24: fixed-size candidate chunks (branch mispredictions)

- A scalar path (the spec loop, one entity at a time) for very sparse rows
  (< 1 entity per 4 cells, i.e. the 100k scenario): 40% slower than the
  8-lane groups (unpredictable short loops). Dropped. That pointed at the
  loop exits: each group runs up to 9 short candidate loops with
  data-dependent trip counts.
- Candidate loops in chunks of 4, running past the end of the window: the
  extra candidates are in cells at least 2 away from every lane of that
  half (d2 >= 4, never a hit), or row-buffer padding (positions 1e30). The
  row buffers now get 8 padding entries after each row. 8-lane groups:
  100k 1T -8%, 1M sparse 1T -9%. Chunks of 8: worse; chunks of 2: same.
- 16-lane groups (dense rows), uncapped pass: each half over its own whole
  window in chunks of 4 (no shared broadcasts, no A-only/both/B-only
  split): dense 1T -3 to -5%. Both halves over the union in chunks: +10%.
  The capped pass keeps the exact segments (chunked variants of it, with
  the self test over the whole own row or only on the other rows, were
  4-18% slower).

A/B v24 against v22 (min of 3, 300 ticks):

    1064 -> 995, 117 -> 107, 10018 -> 9141, 783 -> 736, 23097 -> 22360, 1972 -> 1884

### v25-v26: tidying

Unused state removed, comments brought up to date, row starts of the layout
computed with a vector prefix sum (100k 16T 107 -> 100 us/tick, A/B).
Tried and dropped on the way: prefetching the data of row changers when
the band collects them (no gain), a fixed 16-entry first copy of each kept
run (no gain), a cap hint taken from the row below (per 64 cells: +10% on
dense, too coarse), an early exit of capped groups when all lanes are at
the cap (no gain: lanes rarely all saturate at a row boundary).

## Final design

**Data.** Entities live in grid order: sorted by (cell of the start-of-tick
position, creation sequence). Creation order is the spec's index order
(stable compaction, append-only spawns), so it is all the order there is;
the hash sorts the live entities by sequence at the end, outside the timed
loop. Per slot: position and velocity after integration (4 x f32, double
buffered by tick parity), sequence (u32, bit 31 = changed cell within its
row this tick), cell starts per layout (u32 x 2^20). hp lives in an array
indexed by sequence: only combat touches it. Dead entities get NaN
positions and are dropped by the next rebuild.

**One parallel pass per tick**, then one short serial section.
Rows are grouped in bands; band k owns rows [r0, r1) and

1. rebuilds rows r0-1 .. r1 of the new grid order from last tick's output,
   one row at a time, into three per-thread row buffers (they stay in L2):
   the entities that kept their cell are left-packed in old order (already
   the right order); the ones that changed cell (flagged, or listed by
   destination row because they changed row, or spawned) are bucketed per
   cell, sorted by sequence and merged in. Cell counts come from the
   integration of the previous tick (counts of the old cells adjusted by the
   cell changers), turned into cell starts by a vector prefix sum.
2. separates and integrates row r as soon as row r+1 is built: 8 or 16
   consecutive entities of the row are lanes; every candidate of their
   3-row window is broadcast to them in grid order. Candidates outside a
   lane's 3x3 cells are >= 2 m away, so they can never hit, which makes the
   per-lane result exactly the spec's (same neighbours, same order, same
   sequential float sums; masked lanes add +-0.0f). Each 8-lane half scans
   only its own window, in chunks of 4 that may run past its end (cells >= 2
   away, or padding). The 16-neighbour cap is only applied when a lane
   exceeded it (then the group is redone with the cap and the self test).
   The integration of a row also counts the next tick's cells and lists the
   row changers.

Bands: ~4000 entities each (4 to 32 per thread), balanced by entity count.
Each thread starts on a contiguous run of bands, so consecutive bands
reuse their shared boundary rows from the row buffers; idle threads steal
the upper half of the largest remaining run (CAS on a packed front/back).
Barrier: spin, then yield.

**Serial section** (thread 0): fire, projectiles (at most 15 alive), area
effects (pulse, beam, bomb scan only the cells their reach can touch, grown
by the largest displacement of the tick), contact, deaths, spawns, next
input and hero move. A few microseconds per tick.

**Exactness.** Every float operation is the spec's, in the spec's order:
SSE/AVX `add/sub/mul/div/sqrt/min/max` are correctly rounded, no FMA
(`-ffp-contract=off`, no fma intrinsics), the integer-only reorderings
(grid order, lanes) never change which floats are added in which order for
a given entity. Results do not depend on the thread count or the band split.

## Final numbers

`python tools/test.py`: 0 failure(s) (all model and recorded checks, 1 and 16 threads).
`python tools/bench.py` (default: 5 fresh processes after a warmup, median us/tick):

| scenario | 1 thread | 16 threads |
|---|---:|---:|
| 100k, spawn-half 1000 | 996 | 115 |
| 1M, spawn-half 1000 | 9138 | 737 |
| 1M, spawn-half 300 (dense) | 22425 | 1914 |

Against the plain transcription (`simple.exe`, 1 thread): 2.9x (100k),
8.3x (1M sparse), 11.3x (1M dense) single-threaded. Biggest contributors,
roughly in order: lanes = entities with broadcast candidates (dense 2.5x
on separation), the grid rebuilt from the previous grid order instead of a
counting sort (and fused with the separation into cache-resident row
buffers), work stealing with shared boundary rows (16T -8 to -20%), fixed
chunk candidate loops (-5 to -9%), per-half windows / group width per row.

