# MILLION DEAD: simulation specification (parity contract)

Both engines implement exactly this. The **state hash after N ticks must be
bit-identical** between the two engines for the same scenario: that
is the correctness gate for every benchmark (a benchmark of an engine that
fails parity is invalid).

How exactness is achieved:

- Only IEEE-754 single-precision operations that are correctly rounded:
  `+ - * /`, `sqrt`, `min`/`max` (with the operand order given), compares,
  conversions stated below. **No `rsqrt`/`rcp` approximations, no FMA** in the
  parity category, no `sin`/`cos`/`atan2`/`exp` anywhere in the simulation.
- Every expression is evaluated in the order and grouping written here. C++
  builds with `-ffp-contract=off` and without fast-math, so clang may
  vectorise but never reassociate or contract.
- Iteration orders that affect results (neighbour sums, spawning, grid
  scatter) are fixed below.
- Default MXCSR (round to nearest, no FTZ/DAZ). Both engines must leave it
  alone.

Version: **SIM 5** (milestone 6: abilities: dash, gravity pulse, beam, orbital bombardment). Bump the number
on any rule change; the scenario fixtures record the version they were made
with.

## 1. Constants

    TICK_HZ          60
    DT               0.016666668f          (the f32 nearest 1/60: 0x3C888889)
    WORLD_HALF       1024.0f               world is [-1024, 1024) on X and Z
    CELL_SIZE        2.0f
    INV_CELL         0.5f
    GRID_DIM         1024                  cells per axis
    GRID_CELLS       1048576               GRID_DIM * GRID_DIM
    MAX_ENTITIES     1048576
    SEP_RADIUS       1.0f                  R
    SEP_RADIUS2      1.0f                  R * R
    SEP_MAX_NEIGH    16                    contributions per entity per tick, at most
    SEP_STRENGTH     6.0f
    STEER            0.125f                fraction of the velocity error corrected per tick
    SCUTTLER_SPEED   7.0f                  m/s
    STOP_R2          0.81f                 enemies stop pursuing inside 0.9 m of the hero (SIM 4; was 1e-4)
    HERO_SPEED       9.0f                  m/s
    DIAG             0.70710677f           0x3F3504F3
    BOUND_LO         -1023.5f
    BOUND_HI         1023.5f

## 2. State (structure of arrays)

All arrays have `MAX_ENTITIES` elements, 64-byte aligned, entity `i` at index
`i`. Live entities are `0 .. count-1` (dense; removal is swap-with-last,
milestone 4+).

    pos_x, pos_z, vel_x, vel_z : f32
    hero_x, hero_z             : f32 (one hero)
    tick                       : u32
    count                      : u32

Scratch, rebuilt every tick (contents are part of the work, not the hash):

    cell       : u32[MAX_ENTITIES]     cell index of entity i
    cell_start : u32[GRID_CELLS + 1]   exclusive prefix sums of cell counts
    sorted_x, sorted_z : f32[MAX_ENTITIES]   positions in cell order
    sorted_id  : u32[MAX_ENTITIES]     entity index in cell order

## 3. Random numbers and initial state

PCG32 (`state`, `inc` u64; the standard pcg32 XSH RR output function):

    pcg32_next(rng):
        old = rng.state
        rng.state = old * 6364136223846793005 + rng.inc      (mod 2^64)
        xorshifted = u32(((old >> 18) ^ old) >> 27)
        rot = u32(old >> 59)
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
    pcg32_seed(rng, seed):
        rng.state = 0; rng.inc = (seed << 1) | 1
        pcg32_next(rng); rng.state += seed; pcg32_next(rng)
    unit_f32(u) = f32(u >> 8) * 5.9604645e-08f          (exact: 2^-24)

Scenario init with `seed` and `count`:

    pcg32_seed(rng, seed)
    hero_x = 0.0f; hero_z = 0.0f; tick = 0
    for i in 0 .. count-1:
        a = unit_f32(pcg32_next(rng))
        b = unit_f32(pcg32_next(rng))
        pos_x[i] = (a * 2.0f - 1.0f) * SPAWN_HALF
        pos_z[i] = (b * 2.0f - 1.0f) * SPAWN_HALF
        vel_x[i] = 0.0f; vel_z[i] = 0.0f

with `SPAWN_HALF` given by the scenario (default 1000.0f). Spawning that lands
within 1 m of the hero is allowed.

## 4. One tick

Input for the tick: `keys`, a u8 of bits `W=1 A=2 S=4 D=8` (the recorded
input stream gives one byte per tick).

### 4.1 Hero

    ix = (D ? 1 : 0) - (A ? 1 : 0)        (as f32: -1, 0, 1)
    iz = (W ? 1 : 0) - (S ? 1 : 0)
    if ix != 0 and iz != 0: ix = ix * DIAG; iz = iz * DIAG
    hero_x = hero_x + (ix * HERO_SPEED) * DT
    hero_z = hero_z + (iz * HERO_SPEED) * DT
    hero_x = min(max(hero_x, BOUND_LO), BOUND_HI)      (max first, then min)
    hero_z = min(max(hero_z, BOUND_LO), BOUND_HI)

### 4.2 Grid build (counting sort)

    for i: cx = i32_trunc((pos_x[i] + WORLD_HALF) * INV_CELL)
           cz = i32_trunc((pos_z[i] + WORLD_HALF) * INV_CELL)
           (positions are always inside the bounds, so 0 <= cx, cz < GRID_DIM;
            engines may clamp anyway, which changes nothing)
           cell[i] = cz * GRID_DIM + cx
    counts: cell_start[0 .. GRID_CELLS] = 0
    for i: cell_start[cell[i] + 1] += 1
    prefix: for c in 1 .. GRID_CELLS: cell_start[c] += cell_start[c - 1]
    scatter, i ascending (stable): slot = cursor[cell[i]]++ (cursor starts at cell_start)
           sorted_x[slot] = pos_x[i]; sorted_z[slot] = pos_z[i]; sorted_id[slot] = i

Engines may implement the cursor however they like (a copy of `cell_start`,
or counting down from the end of the cell, provided the resulting order is
the stable ascending one above).

### 4.3 Separation

For each entity `i` (any order: the result for `i` reads only positions from
the start of the tick):

    px = pos_x[i]; pz = pos_z[i]
    cx = cell[i] % GRID_DIM; cz = cell[i] / GRID_DIM
    sx = 0.0f; sz = 0.0f; n = 0
    for oz in -1, 0, 1:                      (outer: rows, ascending z)
      for ox in -1, 0, 1:                    (inner: ascending x)
        if cx+ox or cz+oz is outside 0 .. GRID_DIM-1: skip
        c = (cz+oz) * GRID_DIM + (cx+ox)
        for s in cell_start[c] .. cell_start[c+1]-1:   (ascending)
          if sorted_id[s] == i: continue
          dx = px - sorted_x[s]
          dz = pz - sorted_z[s]
          d2 = dx * dx + dz * dz              ((dx*dx) + (dz*dz))
          if d2 < SEP_RADIUS2:
            w = (SEP_RADIUS2 - d2) * 1.0f     (R2 is 1.0: w = 1.0f - d2)
            sx = sx + dx * w
            sz = sz + dz * w
            n = n + 1
            if n == SEP_MAX_NEIGH: stop all loops for i
    sep_x[i] = sx; sep_z[i] = sz              (per-entity scratch)

Note `d2 < R2` with `d2 == 0` (two entities on the same point) contributes
zero force: that is intended and keeps the rule exact.

### 4.4 Steering and integration

For each entity `i`, after separation is complete for **all** entities:

    dx = hero_x - pos_x[i]
    dz = hero_z - pos_z[i]
    d2 = dx * dx + dz * dz
    if d2 > STOP_R2:
        inv = 1.0f / sqrt(d2)
        tx = (dx * inv) * SCUTTLER_SPEED
        tz = (dz * inv) * SCUTTLER_SPEED
    else:
        tx = 0.0f; tz = 0.0f
    tx = tx + sep_x[i] * SEP_STRENGTH
    tz = tz + sep_z[i] * SEP_STRENGTH
    vx = vel_x[i] + (tx - vel_x[i]) * STEER
    vz = vel_z[i] + (tz - vel_z[i]) * STEER
    nx = pos_x[i] + vx * DT
    nz = pos_z[i] + vz * DT
    pos_x[i] = min(max(nx, BOUND_LO), BOUND_HI)
    pos_z[i] = min(max(nz, BOUND_LO), BOUND_HI)
    vel_x[i] = vx; vel_z[i] = vz

`1.0f / sqrt(d2)` is two correctly rounded operations (sqrtps then divps), not
an `rsqrt`. The branch may be implemented as a mask: compute with `d2`
replaced by 1.0f where `d2 <= STOP_R2`, then select zero.

`min`/`max`: x86 `maxps a, b` returns `b` when either is NaN; no NaN can
occur here (all inputs finite), so any correct min/max is equivalent.

### 4.5 Combat (SIM 4)

State added: `hp[i]` (f32, per entity, `ENEMY_HP` at spawn), hero `hero_hp`
(f32), `hero_alive` (u32), `fire_cooldown` (u32), `kills` (u32), `spawned`
(u32), `target_count` (u32), the RNG (continues from init), and projectiles
(structure of arrays, at most `PROJ_MAX`, dense, index order is creation
order): `proj_x, proj_z, proj_vx, proj_vz` (f32), `proj_life, proj_pierce`
(u32), `proj_count`.

    ENEMY_HP        1.0f
    PROJ_MAX        4096
    PROJ_SPEED      60.0f
    PROJ_LIFE       60          ticks
    PROJ_PIERCE     3           enemies a projectile can kill
    PROJ_R2         0.36f       hit radius 0.6 m, squared
    PROJ_DAMAGE     1.0f
    FIRE_INTERVAL   4           ticks between shots
    SPREAD_C        0.99862953f cos 3 deg
    SPREAD_S        0.05233596f sin 3 deg
    HERO_HP         100.0f
    CONTACT_R2      1.21f       1.1 m, squared
    CONTACT_MAX     20          touching enemies counted, at most
    CONTACT_DPS     4.0f        hp per second per touching enemy
    SPAWN_PER_TICK  64
    SPAWN_MIN       70.0f       spawn ring around the hero
    SPAWN_SPAN      50.0f

Tick input, besides `keys`: `fire` (bit 16 of keys) and the aim point
`(aim_x, aim_z)` in world space.

After 4.4, in this order:

**Fire.** If `hero_alive` and `fire_cooldown == 0` and fire is held and
`proj_count < PROJ_MAX`:

    dx = aim_x - hero_x; dz = aim_z - hero_z; d2 = dx*dx + dz*dz
    if d2 > 1.0e-4f: inv = 1.0f / sqrt(d2); dx = dx * inv; dz = dz * inv
    else: dx = 0.0f; dz = 1.0f
    k = spawned_shots % 3               (a u32 counter of shots fired)
    k == 1: (dx, dz) = (dx*SPREAD_C - dz*SPREAD_S, dx*SPREAD_S + dz*SPREAD_C)
    k == 2: (dx, dz) = (dx*SPREAD_C + dz*SPREAD_S, dz*SPREAD_C - dx*SPREAD_S)
    append: x = hero_x + dx * 0.8f, z = hero_z + dz * 0.8f,
            vx = dx * PROJ_SPEED, vz = dz * PROJ_SPEED, life = PROJ_LIFE, pierce = PROJ_PIERCE
    spawned_shots = spawned_shots + 1; fire_cooldown = FIRE_INTERVAL
    (each expression above in the order written: `dx*SPREAD_C - dz*SPREAD_S` is
     `(dx*SPREAD_C) - (dz*SPREAD_S)`)

Then, if `fire_cooldown > 0`: `fire_cooldown = fire_cooldown - 1`.

**Projectiles.** For each projectile `p` in index order, two half steps:

    repeat 2:
        proj_x = proj_x + proj_vx * HALF_DT;  proj_z = proj_z + proj_vz * HALF_DT    (HALF_DT = 0.008333334f)
        hits: the grid of 4.2 (built from start-of-tick positions) gives
        candidates from the 3x3 cells around the projectile's cell (cell of the
        projectile computed as in 4.2; a projectile outside the world is
        skipped); rows ascending, x ascending, slots ascending; for each
        candidate id: if hp[id] > 0 and
            ((proj_x - pos_x[id])^2 + (proj_z - pos_z[id])^2) < PROJ_R2
        (current positions, after 4.4; dx*dx + dz*dz as in 4.3):
            hp[id] = hp[id] - PROJ_DAMAGE; proj_pierce = proj_pierce - 1
            if proj_pierce == 0: stop this projectile's hits
    proj_life = proj_life - 1

Projectiles with `life == 0` or `pierce == 0` are then removed, keeping the
order of the rest (stable compaction).

**Contact.** If `hero_alive`: count enemies with `hp > 0` and
`(pos - hero)^2 < CONTACT_R2` (dx = pos_x - hero_x etc.) among the 3x3 cells
around the hero's cell, in the grid's order, stopping at CONTACT_MAX;
`hero_hp = hero_hp - (f32(count) * CONTACT_DPS) * DT`; if `hero_hp <= 0.0f`:
`hero_hp = 0.0f; hero_alive = 0`.

**Deaths.** Remove every entity with `hp <= 0.0f`, keeping the order of the
rest (stable compaction of pos, vel, hp); `kills = kills + removed`.

**Spawning.** `n = min(SPAWN_PER_TICK, target_count - count)` (0 when
`count >= target_count`); for each of the `n`, in order:

    loop: a = unit_f32(rng) * 2.0f - 1.0f; b = unit_f32(rng) * 2.0f - 1.0f
          l2 = a*a + b*b; if l2 > 1.0f or l2 < 0.0625f: repeat
    inv = 1.0f / sqrt(l2)
    r = SPAWN_MIN + unit_f32(rng) * SPAWN_SPAN
    x = hero_x + (a * inv) * r;  z = hero_z + (b * inv) * r     (then clamped to the bounds, max then min)
    append with vel 0, hp ENEMY_HP;  spawned = spawned + 1

`target_count` is set by the scenario (the bench keeps it at the initial
count, so the population stays constant as enemies die).

**Hero death.** While `hero_alive == 0` the hero does not move or fire;
enemies continue. Restart is a scenario re-init, outside the simulation.

### 4.6 Abilities (SIM 5)

Input bits: `DASH = 32`, `PULSE = 64`, `BEAM = 128`, `BOMB = 256`.

    DASH_TICKS 10      DASH_SPEED 32.0f    DASH_CD 45
    PULSE_CD 240       PULSE_STEP 0.75f    PULSE_MAX 36.0f    PULSE_PUSH 26.0f
    BEAM_LEN 45.0f     BEAM_W2 0.64f       BEAM_DPT 0.2f
    ENERGY_MAX 100.0f  ENERGY_REGEN 0.25f  BEAM_COST 0.75f
    BOMB_CD 480        BOMB_DELAY 50       BOMB_R 12.0f       BOMB_R2 144.0f
    BOMB_DAMAGE 4.0f   BOMB_PUSH 40.0f

State: `dash_ticks, dash_cd` (u32), `dash_x, dash_z` (f32); `pulse_cd`,
`pulse_on` (u32), `pulse_r, pulse_cx, pulse_cz` (f32); `energy` (f32,
`ENERGY_MAX` at init), `beam_on` (u32); `bomb_cd, bomb_timer` (u32),
`bomb_x, bomb_z` (f32). All zero at init except `energy`.

**Hero (replaces 4.1's movement while dashing).** At the start of the tick,
if alive, `DASH` held, `dash_cd == 0` and `dash_ticks == 0`: the dash
direction is the move vector `(mx, mz)` normalised (`l2 = mx*mx + mz*mz`; if
`l2 > 1e-4f`: divide each by `sqrt(l2)`), else the aim direction normalised
as in Fire, else `(0, 1)`; `dash_ticks = DASH_TICKS; dash_cd = DASH_CD`.
Then, if alive: when `dash_ticks > 0`, `hero = clamp(hero + (dash * DASH_SPEED)
* DT)` per axis and `dash_ticks = dash_ticks - 1`; otherwise the move of 4.1.
Then `if dash_cd > 0: dash_cd = dash_cd - 1`.

**Area effects**, after Projectiles and before Contact. First the effects of
the tick are decided, in this order:

- *Pulse*: if alive, `PULSE` held and `pulse_cd == 0`: `pulse_on = 1,
  pulse_r = 0, pulse_c = hero, pulse_cd = PULSE_CD`. If `pulse_on`: the shell
  is `r0 = pulse_r`, `r1 = r0 + PULSE_STEP`, strength
  `k = PULSE_PUSH * (1.0f - r0 / PULSE_MAX)`; afterwards `pulse_r = r1`, and
  `pulse_on = 0` when `r1 >= PULSE_MAX`. Then `if pulse_cd > 0: pulse_cd -= 1`.
- *Beam*: if alive, `BEAM` held and `energy >= BEAM_COST`: `energy = energy -
  BEAM_COST`, `beam_on = 1`, direction `u` = the aim direction normalised as
  in Fire; else `beam_on = 0` and `energy = min(energy + ENERGY_REGEN,
  ENERGY_MAX)`.
- *Bomb*: if alive, `BOMB` held, `bomb_cd == 0` and `bomb_timer == 0`:
  `bomb_timer = BOMB_DELAY, bomb = aim, bomb_cd = BOMB_CD`. Otherwise, if
  `bomb_timer > 0`: `bomb_timer = bomb_timer - 1`, and the strike happens
  this tick when it becomes 0. Then `if bomb_cd > 0: bomb_cd -= 1`.

Then one pass over every entity `i` (any order: each touches only its own
values), applying the active effects in this order, positions as after 4.4:

    pulse (if this tick has a shell):
        dx = pos_x - pulse_cx; dz = pos_z - pulse_cz; d2 = dx*dx + dz*dz
        if d2 >= r0*r0 and d2 < r1*r1 and d2 > 1e-4f:
            inv = 1.0f / sqrt(d2); vel_x = vel_x + (dx * inv) * k; vel_z likewise
    beam (if beam_on):
        dx = pos_x - hero_x; dz = pos_z - hero_z
        t = dx*ux + dz*uz;  d2 = dx*dx + dz*dz;  perp2 = d2 - t*t
        if t >= 0.0f and t <= BEAM_LEN and perp2 < BEAM_W2: hp = hp - BEAM_DPT
    bomb (if it strikes this tick):
        dx = pos_x - bomb_x; dz = pos_z - bomb_z; d2 = dx*dx + dz*dz
        if d2 < BOMB_R2:
            hp = hp - BOMB_DAMAGE
            if d2 > 1e-4f: d = sqrt(d2); inv = 1.0f / d
                vel_x = vel_x + (dx * inv) * (BOMB_PUSH * (1.0f - d / BOMB_R)); vel_z likewise

(`dx*ux + dz*uz` is `(dx*ux) + (dz*uz)`; `r0*r0`, `r1*r1` are computed once.)

**Contact** is skipped while `dash_ticks > 0` (the dash makes the hero
invulnerable), as is when the hero is dead.

### 4.7 End of tick

    tick = tick + 1

### 4.8 Bench input

The recorded bench input is generated: `keys = BENCH_KEYS[(tick / 120) & 3]
| FIRE`, plus (SIM 5) `DASH` when `tick % 180 == 0`, `PULSE` when
`tick % 300 == 60`, `BEAM` when `(tick / 90) % 2 == 1`, `BOMB` when
`tick % 600 == 120` with `BENCH_KEYS = {W, D, S, A}`, and the aim
`aim = hero + BENCH_AIM[(tick / 30) & 3]` (computed with the hero position
at the start of the tick: `aim_x = hero_x + ax` as one f32 add), with
`BENCH_AIM = {(10, 0), (0, 10), (-10, 0), (0, -10)}`.

## 5. State hash

FNV-1a 64 (offset `0xcbf29ce484222325`, prime `0x100000001b3`) over, in
order: `count` (4 bytes), `tick` (4), `hero_x`, `hero_z` (4 each), then the
raw bytes of `pos_x[0..count)`, `pos_z[0..count)`, `vel_x[0..count)`,
`vel_z[0..count)`, then (SIM 4) `hp[0..count)`, `hero_hp`, `hero_alive`,
`kills`, `proj_count` (4 each), `proj_x, proj_z[0..proj_count)`, then (SIM 5)
`energy`, `pulse_r` (f32), `dash_ticks`, `pulse_on`, `bomb_timer` (u32). Printed as 16
lowercase hex digits.

Init (SIM 4) additionally sets `hp[i] = ENEMY_HP`, `hero_hp = HERO_HP`,
`hero_alive = 1`, `fire_cooldown = 0`, `kills = 0`, `spawned = 0`,
`spawned_shots = 0`, `proj_count = 0`, `target_count = count`.

## 6. Rendering instance generation (part of the competed CPU work)

Per frame, after the simulation:

- Camera frustum, projected to the ground plane, is given as 4 half-planes
  `(a, b, c)`: a point `(x, z)` is inside when `a*x + b*z + c >= -CULL_MARGIN`
  for all four (`CULL_MARGIN = 1.0f`). The camera module computes the planes;
  the plane computation itself need not be bit-exact between engines, but
  for benchmark parity both engines read the planes from the scenario file.
- LOD by squared distance to the camera's ground focus point `(fx, fz)`:
  `d2 < LOD0_D2` LOD0, `< LOD1_D2` LOD1, otherwise LOD2, with
  `LOD0_D2 = 900.0f` (30 m), `LOD1_D2 = 6400.0f` (80 m).
- Each visible entity writes one 16-byte instance `float4(x, z, vx, vz)` to
  the instance buffer region of its LOD, entities in ascending index order
  within a region. The vertex shader derives heading from `(vx, vz)` and
  the animation phase from the instance id and time, so no transcendental
  runs on the CPU.
- Instance parity: counts per LOD and an FNV-1a hash of each region must
  match between engines.

## 7. What is per entity and what is approximated (reporting)

| system | rate | exact? |
|---|---|---|
| grid build | every tick, every entity | exact |
| separation | every tick, every entity, at most 16 neighbours | exact, capped |
| pursuit + integration | every tick, every entity | exact |
| culling + LOD + instance write | every rendered frame, every entity | exact |

Later milestones add combat, health, death, archetypes and state machines
here, with the same table kept honest.
