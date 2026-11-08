"""Independent reference model of shared/specs/sim_spec.md (SIM 5).

Pure Python, written from the spec, not from the assembly. Every f32
operation is computed in double precision and rounded to f32: for + - * / and
sqrt of f32 operands, double has enough precision that this double rounding
equals the correctly rounded f32 result, so the model is exact.

    python tools/sim_reference.py --entities 1000 --ticks 120 --seed 1
prints the state hash, which must equal the engine's `--bench-sim` with the
same arguments. Slow: meant for small populations.
"""
import argparse
import math
import struct

M64 = (1 << 64) - 1
_f = struct.Struct('<f')


def f32(x):
    return _f.unpack(_f.pack(x))[0]


DT = f32(1.0 / 60.0)
assert struct.pack('<f', DT) == bytes.fromhex('8988883c')
WORLD_HALF, INV_CELL, GRID_DIM = 1024.0, 0.5, 1024
SEP_MAX = 16
SEP_STRENGTH, STEER, SPEED, HERO_SPEED = 6.0, 0.125, 7.0, 9.0
DIAG = f32(0.70710677)
LO, HI = -1023.5, 1023.5
EPS = f32(1.0e-4)
STOP_R2 = f32(0.81)
BENCH_KEYS = [1, 8, 4, 2]
BENCH_AIM = [(10.0, 0.0), (0.0, 10.0), (-10.0, 0.0), (0.0, -10.0)]
FIRE = 16
ENEMY_HP, PROJ_MAX, PROJ_SPEED, PROJ_LIFE, PROJ_PIERCE = 1.0, 4096, 60.0, 60, 3
PROJ_R2, PROJ_DAMAGE, FIRE_INTERVAL = f32(0.36), 1.0, 4
SPREAD_C, SPREAD_S = f32(0.99862953), f32(0.05233596)
HERO_HP, CONTACT_R2, CONTACT_MAX, CONTACT_DPS = 100.0, f32(1.21), 20, 4.0
SPAWN_PER_TICK, SPAWN_MIN, SPAWN_SPAN = 64, 70.0, 50.0
HALF_DT = f32(0.008333334)
UNIT = f32(5.9604645e-08)
MUZZLE = f32(0.8)
DASH, PULSE, BEAM, BOMB = 32, 64, 128, 256
DASH_TICKS, DASH_SPEED, DASH_CD = 10, 32.0, 45
PULSE_CD, PULSE_STEP, PULSE_MAX, PULSE_PUSH = 240, 0.75, 36.0, 26.0
BEAM_LEN, BEAM_W2, BEAM_DPT = 45.0, f32(0.64), f32(0.2)
ENERGY_MAX, ENERGY_REGEN, BEAM_COST = 100.0, 0.25, 0.75
BOMB_CD, BOMB_DELAY, BOMB_R, BOMB_R2, BOMB_DAMAGE, BOMB_PUSH = 480, 50, 12.0, 144.0, 4.0, 40.0


class Pcg:
    def __init__(self, seed):
        self.state, self.inc = 0, ((seed << 1) | 1) & M64
        self.next()
        self.state = (self.state + seed) & M64
        self.next()

    def next(self):
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & M64
        x = (((old >> 18) ^ old) >> 27) & 0xffffffff
        r = old >> 59
        return ((x >> r) | (x << ((-r) & 31))) & 0xffffffff


def add(a, b): return f32(a + b)
def sub(a, b): return f32(a - b)
def mul(a, b): return f32(a * b)
def div(a, b): return f32(a / b)


def clamp(v):
    return min(max(v, LO), HI)


class Sim:
    def __init__(self, count, seed, spawn_half=1000.0):
        rng = Pcg(seed)
        self.rng = rng
        self.count, self.tick = count, 0
        self.hx = self.hz = 0.0
        self.px, self.pz = [0.0] * count, [0.0] * count
        self.vx, self.vz = [0.0] * count, [0.0] * count
        for i in range(count):
            a = mul(float(rng.next() >> 8), f32(5.9604645e-08))
            b = mul(float(rng.next() >> 8), f32(5.9604645e-08))
            self.px[i] = mul(sub(mul(a, 2.0), 1.0), spawn_half)
            self.pz[i] = mul(sub(mul(b, 2.0), 1.0), spawn_half)
        self.hp = [ENEMY_HP] * count
        self.hero_hp, self.hero_alive, self.cool = HERO_HP, 1, 0
        self.kills = self.spawned = self.shots = 0
        self.target = count
        self.proj = []   # [x, z, vx, vz, life, pierce]
        self.dash_ticks = self.dash_cd = 0
        self.dash_x = self.dash_z = 0.0
        self.pulse_cd = self.pulse_on = 0
        self.pulse_r = self.pulse_cx = self.pulse_cz = 0.0
        self.energy, self.beam_on = ENERGY_MAX, 0
        self.bomb_cd = self.bomb_timer = 0
        self.bomb_x = self.bomb_z = 0.0

    def step(self, keys, aim=(0.0, 1.0)):
        ix = float((1 if keys & 8 else 0) - (1 if keys & 2 else 0))
        iz = float((1 if keys & 1 else 0) - (1 if keys & 4 else 0))
        if ix != 0 and iz != 0:
            ix, iz = mul(ix, DIAG), mul(iz, DIAG)
        if self.hero_alive and (keys & DASH) and self.dash_cd == 0 and self.dash_ticks == 0:
            l2 = add(mul(ix, ix), mul(iz, iz))
            if l2 > EPS:
                r = f32(math.sqrt(l2))
                self.dash_x, self.dash_z = div(ix, r), div(iz, r)
            else:
                self.dash_x, self.dash_z = self.aim_dir(aim)
            self.dash_ticks, self.dash_cd = DASH_TICKS, DASH_CD
        if self.hero_alive:
            if self.dash_ticks > 0:
                self.hx = clamp(add(self.hx, mul(mul(self.dash_x, DASH_SPEED), DT)))
                self.hz = clamp(add(self.hz, mul(mul(self.dash_z, DASH_SPEED), DT)))
                self.dash_ticks -= 1
            else:
                self.hx = clamp(add(self.hx, mul(mul(ix, HERO_SPEED), DT)))
                self.hz = clamp(add(self.hz, mul(mul(iz, HERO_SPEED), DT)))
        if self.dash_cd > 0:
            self.dash_cd -= 1
        n = self.count
        # grid
        cells = {}
        cell = [0] * n
        for i in range(n):
            cx = int(mul(add(self.px[i], WORLD_HALF), INV_CELL))
            cz = int(mul(add(self.pz[i], WORLD_HALF), INV_CELL))
            cell[i] = (cx, cz)
            cells.setdefault((cx, cz), []).append(i)   # ascending i: stable
        # separation
        sep = [(0.0, 0.0)] * n
        for i in range(n):
            px, pz = self.px[i], self.pz[i]
            cx, cz = cell[i]
            sx = sz = 0.0
            k = 0
            done = False
            for oz in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    c = (cx + ox, cz + oz)
                    if not (0 <= c[0] < GRID_DIM and 0 <= c[1] < GRID_DIM):
                        continue
                    for j in cells.get(c, ()):
                        if j == i:
                            continue
                        dx, dz = sub(px, self.px[j]), sub(pz, self.pz[j])
                        d2 = add(mul(dx, dx), mul(dz, dz))
                        if d2 < 1.0:
                            w = sub(1.0, d2)
                            sx, sz = add(sx, mul(dx, w)), add(sz, mul(dz, w))
                            k += 1
                            if k == SEP_MAX:
                                done = True
                                break
                    if done:
                        break
                if done:
                    break
            sep[i] = (sx, sz)
        # integrate
        for i in range(n):
            dx, dz = sub(self.hx, self.px[i]), sub(self.hz, self.pz[i])
            d2 = add(mul(dx, dx), mul(dz, dz))
            if d2 > STOP_R2:
                inv = div(1.0, f32(math.sqrt(d2)))
                tx, tz = mul(mul(dx, inv), SPEED), mul(mul(dz, inv), SPEED)
            else:
                tx = tz = 0.0
            tx = add(tx, mul(sep[i][0], SEP_STRENGTH))
            tz = add(tz, mul(sep[i][1], SEP_STRENGTH))
            vx = add(self.vx[i], mul(sub(tx, self.vx[i]), STEER))
            vz = add(self.vz[i], mul(sub(tz, self.vz[i]), STEER))
            self.px[i] = clamp(add(self.px[i], mul(vx, DT)))
            self.pz[i] = clamp(add(self.pz[i], mul(vz, DT)))
            self.vx[i], self.vz[i] = vx, vz
        self.combat(keys, aim, cells)
        self.tick += 1

    def candidates(self, x, z, cells):
        cx = int(mul(add(x, WORLD_HALF), INV_CELL))
        cz = int(mul(add(z, WORLD_HALF), INV_CELL))
        for oz in (-1, 0, 1):
            for ox in (-1, 0, 1):
                c = (cx + ox, cz + oz)
                if 0 <= c[0] < GRID_DIM and 0 <= c[1] < GRID_DIM:
                    yield from cells.get(c, ())

    def aim_dir(self, aim):
        dx, dz = sub(aim[0], self.hx), sub(aim[1], self.hz)
        d2 = add(mul(dx, dx), mul(dz, dz))
        if d2 > EPS:
            inv = div(1.0, f32(math.sqrt(d2)))
            return mul(dx, inv), mul(dz, inv)
        return 0.0, 1.0

    def abilities(self, keys, aim):
        shell = None
        if self.hero_alive and (keys & PULSE) and self.pulse_cd == 0:
            self.pulse_on, self.pulse_r, self.pulse_cx, self.pulse_cz, self.pulse_cd = 1, 0.0, self.hx, self.hz, PULSE_CD
        if self.pulse_on:
            r0 = self.pulse_r
            r1 = add(r0, PULSE_STEP)
            k = mul(PULSE_PUSH, sub(1.0, div(r0, PULSE_MAX)))
            shell = (mul(r0, r0), mul(r1, r1), k)
            self.pulse_r = r1
            if r1 >= PULSE_MAX:
                self.pulse_on = 0
        if self.pulse_cd > 0:
            self.pulse_cd -= 1
        beam = None
        if self.hero_alive and (keys & BEAM) and self.energy >= BEAM_COST:
            self.energy = sub(self.energy, BEAM_COST)
            self.beam_on = 1
            beam = self.aim_dir(aim)
        else:
            self.beam_on = 0
            self.energy = min(add(self.energy, ENERGY_REGEN), ENERGY_MAX)
        strike = False
        if self.hero_alive and (keys & BOMB) and self.bomb_cd == 0 and self.bomb_timer == 0:
            self.bomb_timer, self.bomb_x, self.bomb_z, self.bomb_cd = BOMB_DELAY, aim[0], aim[1], BOMB_CD
        elif self.bomb_timer > 0:
            self.bomb_timer -= 1
            strike = self.bomb_timer == 0
        if self.bomb_cd > 0:
            self.bomb_cd -= 1
        for i in range(self.count):
            px, pz = self.px[i], self.pz[i]
            if shell:
                dx, dz = sub(px, self.pulse_cx), sub(pz, self.pulse_cz)
                d2 = add(mul(dx, dx), mul(dz, dz))
                if d2 >= shell[0] and d2 < shell[1] and d2 > EPS:
                    inv = div(1.0, f32(math.sqrt(d2)))
                    self.vx[i] = add(self.vx[i], mul(mul(dx, inv), shell[2]))
                    self.vz[i] = add(self.vz[i], mul(mul(dz, inv), shell[2]))
            if beam:
                dx, dz = sub(px, self.hx), sub(pz, self.hz)
                t = add(mul(dx, beam[0]), mul(dz, beam[1]))
                d2 = add(mul(dx, dx), mul(dz, dz))
                perp2 = sub(d2, mul(t, t))
                if t >= 0.0 and t <= BEAM_LEN and perp2 < BEAM_W2:
                    self.hp[i] = sub(self.hp[i], BEAM_DPT)
            if strike:
                dx, dz = sub(px, self.bomb_x), sub(pz, self.bomb_z)
                d2 = add(mul(dx, dx), mul(dz, dz))
                if d2 < BOMB_R2:
                    self.hp[i] = sub(self.hp[i], BOMB_DAMAGE)
                    if d2 > EPS:
                        d = f32(math.sqrt(d2))
                        inv = div(1.0, d)
                        k = mul(BOMB_PUSH, sub(1.0, div(d, BOMB_R)))
                        self.vx[i] = add(self.vx[i], mul(mul(dx, inv), k))
                        self.vz[i] = add(self.vz[i], mul(mul(dz, inv), k))

    def combat(self, keys, aim, cells):
        # fire
        if self.hero_alive and self.cool == 0 and (keys & FIRE) and len(self.proj) < PROJ_MAX:
            dx, dz = self.aim_dir(aim)
            k = self.shots % 3
            if k == 1:
                dx, dz = sub(mul(dx, SPREAD_C), mul(dz, SPREAD_S)), add(mul(dx, SPREAD_S), mul(dz, SPREAD_C))
            elif k == 2:
                dx, dz = add(mul(dx, SPREAD_C), mul(dz, SPREAD_S)), sub(mul(dz, SPREAD_C), mul(dx, SPREAD_S))
            self.proj.append([add(self.hx, mul(dx, MUZZLE)), add(self.hz, mul(dz, MUZZLE)),
                              mul(dx, PROJ_SPEED), mul(dz, PROJ_SPEED), PROJ_LIFE, PROJ_PIERCE])
            self.shots += 1
            self.cool = FIRE_INTERVAL
        if self.cool > 0:
            self.cool -= 1
        # projectiles
        for p in self.proj:
            for _ in range(2):
                p[0] = add(p[0], mul(p[2], HALF_DT))
                p[1] = add(p[1], mul(p[3], HALF_DT))
                if p[5] == 0 or not (-WORLD_HALF <= p[0] < WORLD_HALF and -WORLD_HALF <= p[1] < WORLD_HALF):
                    continue
                for j in self.candidates(p[0], p[1], cells):
                    if self.hp[j] > 0.0:
                        dx, dz = sub(p[0], self.px[j]), sub(p[1], self.pz[j])
                        if add(mul(dx, dx), mul(dz, dz)) < PROJ_R2:
                            self.hp[j] = sub(self.hp[j], PROJ_DAMAGE)
                            p[5] -= 1
                            if p[5] == 0:
                                break
            p[4] -= 1
        self.proj = [p for p in self.proj if p[4] > 0 and p[5] > 0]
        self.abilities(keys, aim)
        # contact
        if self.hero_alive and self.dash_ticks == 0:
            n = 0
            for j in self.candidates(self.hx, self.hz, cells):
                if self.hp[j] > 0.0:
                    dx, dz = sub(self.px[j], self.hx), sub(self.pz[j], self.hz)
                    if add(mul(dx, dx), mul(dz, dz)) < CONTACT_R2:
                        n += 1
                        if n == CONTACT_MAX:
                            break
            self.hero_hp = sub(self.hero_hp, mul(mul(float(n), CONTACT_DPS), DT))
            if self.hero_hp <= 0.0:
                self.hero_hp, self.hero_alive = 0.0, 0
        # deaths
        keep = [i for i in range(self.count) if self.hp[i] > 0.0]
        self.kills += self.count - len(keep)
        for col in ('px', 'pz', 'vx', 'vz', 'hp'):
            c = getattr(self, col)
            setattr(self, col, [c[i] for i in keep])
        self.count = len(keep)
        # spawning
        n = min(SPAWN_PER_TICK, self.target - self.count) if self.count < self.target else 0
        for _ in range(n):
            while True:
                a = sub(mul(mul(float(self.rng.next() >> 8), UNIT), 2.0), 1.0)
                b = sub(mul(mul(float(self.rng.next() >> 8), UNIT), 2.0), 1.0)
                l2 = add(mul(a, a), mul(b, b))
                if not (l2 > 1.0 or l2 < 0.0625):
                    break
            inv = div(1.0, f32(math.sqrt(l2)))
            r = add(SPAWN_MIN, mul(mul(float(self.rng.next() >> 8), UNIT), SPAWN_SPAN))
            self.px.append(clamp(add(self.hx, mul(mul(a, inv), r))))
            self.pz.append(clamp(add(self.hz, mul(mul(b, inv), r))))
            self.vx.append(0.0)
            self.vz.append(0.0)
            self.hp.append(ENEMY_HP)
            self.count += 1
            self.spawned += 1

    def bench_keys(self):
        k = BENCH_KEYS[(self.tick // 120) & 3] | FIRE
        t = self.tick
        if t % 180 == 0:
            k |= DASH
        if t % 300 == 60:
            k |= PULSE
        if (t // 90) % 2 == 1:
            k |= BEAM
        if t % 600 == 120:
            k |= BOMB
        return k

    def bench_aim(self):
        ax, az = BENCH_AIM[(self.tick // 30) & 3]
        return (add(self.hx, ax), add(self.hz, az))

    def bench_step(self):
        self.step(self.bench_keys(), self.bench_aim())

    def hash(self):
        h = 0xcbf29ce484222325
        data = struct.pack('<IIff', self.count, self.tick, self.hx, self.hz)
        for col in (self.px, self.pz, self.vx, self.vz, self.hp):
            data += struct.pack(f'<{self.count}f', *col)
        data += struct.pack('<fIII', self.hero_hp, self.hero_alive, self.kills, len(self.proj))
        data += struct.pack(f'<{len(self.proj)}f', *(p[0] for p in self.proj))
        data += struct.pack(f'<{len(self.proj)}f', *(p[1] for p in self.proj))
        data += struct.pack('<ffIII', self.energy, self.pulse_r, self.dash_ticks, self.pulse_on, self.bomb_timer)
        for b in data:
            h = ((h ^ b) * 0x100000001b3) & M64
        return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entities', type=int, default=1000)
    ap.add_argument('--ticks', type=int, default=120)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--spawn-half', type=int, default=1000)
    a = ap.parse_args()
    s = Sim(a.entities, a.seed, float(a.spawn_half))
    for _ in range(a.ticks):
        s.bench_step()
    print(f'hash={s.hash():016x} kills={s.kills} hero_hp={s.hero_hp} alive={s.hero_alive} proj={len(s.proj)} count={s.count}')


if __name__ == '__main__':
    main()
