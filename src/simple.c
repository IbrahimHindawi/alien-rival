/* Straightforward single-threaded SIM 5, written directly from the spec.
   Kept as a debugging oracle for the optimised engine (build/simple.exe). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define GRID_DIM 1024
#define GRID_CELLS (GRID_DIM * GRID_DIM)
#define MAXE (1 << 20)
static const float DT = 0.016666668f, HALF_DT = 0.008333334f;
static const float LO = -1023.5f, HI = 1023.5f, EPS = 1.0e-4f;

typedef struct { uint64_t state, inc; } pcg;
static uint32_t pcg_next(pcg *r) {
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + r->inc;
    uint32_t x = (uint32_t)(((old >> 18) ^ old) >> 27), rot = (uint32_t)(old >> 59);
    return (x >> rot) | (x << ((0u - rot) & 31));
}
static void pcg_seed(pcg *r, uint64_t seed) {
    r->state = 0; r->inc = (seed << 1) | 1; pcg_next(r); r->state += seed; pcg_next(r);
}
static float unitf(uint32_t u) { return (float)(u >> 8) * 5.9604645e-08f; }
static float clampf(float v) { v = v > LO ? v : LO; return v < HI ? v : HI; }

static float px[MAXE], pz[MAXE], vx[MAXE], vz[MAXE], hp[MAXE], sepx[MAXE], sepz[MAXE];
static uint32_t cell[MAXE], cs[GRID_CELLS + 1], cur[GRID_CELLS], sid[MAXE];
static float sx_[MAXE], sz_[MAXE];
static float prx[4096], prz[4096], prvx[4096], prvz[4096];
static uint32_t prl[4096], prp[4096], npr;
static uint32_t count, tick, target, alive = 1, cool, kills, spawned, shots;
static float hx, hz, hhp = 100.0f;
static uint32_t dash_ticks, dash_cd, pulse_cd, pulse_on, beam_on, bomb_cd, bomb_timer;
static float dash_x, dash_z, pulse_r, pulse_cx, pulse_cz, energy = 100.0f, bomb_x, bomb_z;
static pcg rng;

static void cell_of(float x, float z, int *cx, int *cz) {
    *cx = (int)((x + 1024.0f) * 0.5f); *cz = (int)((z + 1024.0f) * 0.5f);
}
static void aim_dir(float ax, float az, float *ox, float *oz) {
    float dx = ax - hx, dz = az - hz, d2 = dx * dx + dz * dz;
    if (d2 > EPS) { float inv = 1.0f / sqrtf(d2); *ox = dx * inv; *oz = dz * inv; }
    else { *ox = 0.0f; *oz = 1.0f; }
}

static void step(uint32_t keys, float aimx, float aimz) {
    float ix = (float)(((keys & 8) ? 1 : 0) - ((keys & 2) ? 1 : 0));
    float iz = (float)(((keys & 1) ? 1 : 0) - ((keys & 4) ? 1 : 0));
    if (ix != 0 && iz != 0) { ix = ix * 0.70710677f; iz = iz * 0.70710677f; }
    if (alive && (keys & 32) && dash_cd == 0 && dash_ticks == 0) {
        float l2 = ix * ix + iz * iz;
        if (l2 > EPS) { float r = sqrtf(l2); dash_x = ix / r; dash_z = iz / r; }
        else aim_dir(aimx, aimz, &dash_x, &dash_z);
        dash_ticks = 10; dash_cd = 45;
    }
    if (alive) {
        if (dash_ticks > 0) {
            hx = clampf(hx + (dash_x * 32.0f) * DT); hz = clampf(hz + (dash_z * 32.0f) * DT); dash_ticks--;
        } else {
            hx = clampf(hx + (ix * 9.0f) * DT); hz = clampf(hz + (iz * 9.0f) * DT);
        }
    }
    if (dash_cd > 0) dash_cd--;
    uint32_t n = count;
    for (uint32_t i = 0; i < n; i++) {
        int cx, cz; cell_of(px[i], pz[i], &cx, &cz); cell[i] = (uint32_t)(cz * GRID_DIM + cx);
    }
    memset(cs, 0, sizeof cs);
    for (uint32_t i = 0; i < n; i++) cs[cell[i] + 1]++;
    for (uint32_t c = 1; c <= GRID_CELLS; c++) cs[c] += cs[c - 1];
    memcpy(cur, cs, sizeof cur);
    for (uint32_t i = 0; i < n; i++) { uint32_t s = cur[cell[i]]++; sx_[s] = px[i]; sz_[s] = pz[i]; sid[s] = i; }
    for (uint32_t i = 0; i < n; i++) {
        float x = px[i], z = pz[i], sx = 0, sz = 0; int k = 0;
        int cx = (int)(cell[i] % GRID_DIM), cz = (int)(cell[i] / GRID_DIM);
        for (int oz = -1; oz <= 1; oz++) for (int ox = -1; ox <= 1; ox++) {
            int x2 = cx + ox, z2 = cz + oz;
            if (x2 < 0 || z2 < 0 || x2 >= GRID_DIM || z2 >= GRID_DIM) continue;
            uint32_t c = (uint32_t)(z2 * GRID_DIM + x2);
            for (uint32_t s = cs[c]; s < cs[c + 1]; s++) {
                if (sid[s] == i) continue;
                float dx = x - sx_[s], dz = z - sz_[s], d2 = dx * dx + dz * dz;
                if (d2 < 1.0f) { float w = 1.0f - d2; sx = sx + dx * w; sz = sz + dz * w; if (++k == 16) goto done; }
            }
        }
    done:
        sepx[i] = sx; sepz[i] = sz;
    }
    for (uint32_t i = 0; i < n; i++) {
        float dx = hx - px[i], dz = hz - pz[i], d2 = dx * dx + dz * dz, tx = 0, tz = 0;
        if (d2 > 0.81f) { float inv = 1.0f / sqrtf(d2); tx = (dx * inv) * 7.0f; tz = (dz * inv) * 7.0f; }
        tx = tx + sepx[i] * 6.0f; tz = tz + sepz[i] * 6.0f;
        float nvx = vx[i] + (tx - vx[i]) * 0.125f, nvz = vz[i] + (tz - vz[i]) * 0.125f;
        px[i] = clampf(px[i] + nvx * DT); pz[i] = clampf(pz[i] + nvz * DT);
        vx[i] = nvx; vz[i] = nvz;
    }
    /* fire */
    if (alive && cool == 0 && (keys & 16) && npr < 4096) {
        float dx, dz; aim_dir(aimx, aimz, &dx, &dz);
        uint32_t k = shots % 3;
        const float C = 0.99862953f, S = 0.05233596f;
        if (k == 1) { float a = dx * C - dz * S, b = dx * S + dz * C; dx = a; dz = b; }
        else if (k == 2) { float a = dx * C + dz * S, b = dz * C - dx * S; dx = a; dz = b; }
        prx[npr] = hx + dx * 0.8f; prz[npr] = hz + dz * 0.8f; prvx[npr] = dx * 60.0f; prvz[npr] = dz * 60.0f;
        prl[npr] = 60; prp[npr] = 3; npr++; shots++; cool = 4;
    }
    if (cool > 0) cool--;
    for (uint32_t p = 0; p < npr; p++) {
        for (int h = 0; h < 2; h++) {
            prx[p] = prx[p] + prvx[p] * HALF_DT; prz[p] = prz[p] + prvz[p] * HALF_DT;
            if (prp[p] == 0 || !(prx[p] >= -1024.0f && prx[p] < 1024.0f && prz[p] >= -1024.0f && prz[p] < 1024.0f)) continue;
            int cx, cz; cell_of(prx[p], prz[p], &cx, &cz);
            for (int oz = -1; oz <= 1; oz++) for (int ox = -1; ox <= 1; ox++) {
                int x2 = cx + ox, z2 = cz + oz;
                if (x2 < 0 || z2 < 0 || x2 >= GRID_DIM || z2 >= GRID_DIM) continue;
                uint32_t c = (uint32_t)(z2 * GRID_DIM + x2);
                for (uint32_t s = cs[c]; s < cs[c + 1]; s++) {
                    uint32_t j = sid[s];
                    if (hp[j] > 0.0f) {
                        float dx = prx[p] - px[j], dz = prz[p] - pz[j];
                        if (dx * dx + dz * dz < 0.36f) { hp[j] = hp[j] - 1.0f; if (--prp[p] == 0) goto pdone; }
                    }
                }
            }
        pdone:;
        }
        prl[p]--;
    }
    { uint32_t m = 0;
      for (uint32_t p = 0; p < npr; p++) if (prl[p] > 0 && prp[p] > 0) {
          prx[m] = prx[p]; prz[m] = prz[p]; prvx[m] = prvx[p]; prvz[m] = prvz[p]; prl[m] = prl[p]; prp[m] = prp[p]; m++; }
      npr = m; }
    /* abilities */
    int shell = 0; float s0 = 0, s1 = 0, sk = 0;
    if (alive && (keys & 64) && pulse_cd == 0) { pulse_on = 1; pulse_r = 0; pulse_cx = hx; pulse_cz = hz; pulse_cd = 240; }
    if (pulse_on) {
        float r0 = pulse_r, r1 = r0 + 0.75f; sk = 26.0f * (1.0f - r0 / 36.0f);
        s0 = r0 * r0; s1 = r1 * r1; shell = 1; pulse_r = r1; if (r1 >= 36.0f) pulse_on = 0;
    }
    if (pulse_cd > 0) pulse_cd--;
    int beam = 0; float ux = 0, uz = 0;
    if (alive && (keys & 128) && energy >= 0.75f) { energy = energy - 0.75f; beam_on = 1; beam = 1; aim_dir(aimx, aimz, &ux, &uz); }
    else { beam_on = 0; float e = energy + 0.25f; energy = e < 100.0f ? e : 100.0f; }
    int strike = 0;
    if (alive && (keys & 256) && bomb_cd == 0 && bomb_timer == 0) { bomb_timer = 50; bomb_x = aimx; bomb_z = aimz; bomb_cd = 480; }
    else if (bomb_timer > 0) { bomb_timer--; strike = bomb_timer == 0; }
    if (bomb_cd > 0) bomb_cd--;
    for (uint32_t i = 0; i < n; i++) {
        if (shell) {
            float dx = px[i] - pulse_cx, dz = pz[i] - pulse_cz, d2 = dx * dx + dz * dz;
            if (d2 >= s0 && d2 < s1 && d2 > EPS) { float inv = 1.0f / sqrtf(d2); vx[i] = vx[i] + (dx * inv) * sk; vz[i] = vz[i] + (dz * inv) * sk; }
        }
        if (beam) {
            float dx = px[i] - hx, dz = pz[i] - hz, t = dx * ux + dz * uz, d2 = dx * dx + dz * dz, p2 = d2 - t * t;
            if (t >= 0.0f && t <= 45.0f && p2 < 0.64f) hp[i] = hp[i] - 0.2f;
        }
        if (strike) {
            float dx = px[i] - bomb_x, dz = pz[i] - bomb_z, d2 = dx * dx + dz * dz;
            if (d2 < 144.0f) {
                hp[i] = hp[i] - 4.0f;
                if (d2 > EPS) { float d = sqrtf(d2), inv = 1.0f / d, k = 40.0f * (1.0f - d / 12.0f);
                    vx[i] = vx[i] + (dx * inv) * k; vz[i] = vz[i] + (dz * inv) * k; }
            }
        }
    }
    /* contact */
    if (alive && dash_ticks == 0) {
        int cx, cz, cnt = 0; cell_of(hx, hz, &cx, &cz);
        for (int oz = -1; oz <= 1; oz++) for (int ox = -1; ox <= 1; ox++) {
            int x2 = cx + ox, z2 = cz + oz;
            if (x2 < 0 || z2 < 0 || x2 >= GRID_DIM || z2 >= GRID_DIM) continue;
            uint32_t c = (uint32_t)(z2 * GRID_DIM + x2);
            for (uint32_t s = cs[c]; s < cs[c + 1]; s++) {
                uint32_t j = sid[s];
                if (hp[j] > 0.0f) { float dx = px[j] - hx, dz = pz[j] - hz; if (dx * dx + dz * dz < 1.21f) { if (++cnt == 20) goto cdone; } }
            }
        }
    cdone:
        hhp = hhp - ((float)cnt * 4.0f) * DT;
        if (hhp <= 0.0f) { hhp = 0.0f; alive = 0; }
    }
    { uint32_t m = 0;
      for (uint32_t i = 0; i < n; i++) if (hp[i] > 0.0f) { px[m] = px[i]; pz[m] = pz[i]; vx[m] = vx[i]; vz[m] = vz[i]; hp[m] = hp[i]; m++; }
      kills += n - m; count = m; }
    uint32_t ns = count < target ? (target - count < 64 ? target - count : 64) : 0;
    for (uint32_t q = 0; q < ns; q++) {
        float a, b, l2;
        for (;;) { a = unitf(pcg_next(&rng)) * 2.0f - 1.0f; b = unitf(pcg_next(&rng)) * 2.0f - 1.0f; l2 = a * a + b * b; if (!(l2 > 1.0f || l2 < 0.0625f)) break; }
        float inv = 1.0f / sqrtf(l2), r = 70.0f + unitf(pcg_next(&rng)) * 50.0f;
        px[count] = clampf(hx + (a * inv) * r); pz[count] = clampf(hz + (b * inv) * r);
        vx[count] = 0; vz[count] = 0; hp[count] = 1.0f; count++; spawned++;
    }
    tick++;
}

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ULL; } return h;
}

int main(int argc, char **argv) {
    uint32_t N = 1000, T = 120, seed = 1; float half = 1000.0f;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--entities")) N = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ticks")) T = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")) seed = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--spawn-half")) half = (float)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")) ++i;
    }
    pcg_seed(&rng, seed);
    for (uint32_t i = 0; i < N; i++) {
        float a = unitf(pcg_next(&rng)), b = unitf(pcg_next(&rng));
        px[i] = (a * 2.0f - 1.0f) * half; pz[i] = (b * 2.0f - 1.0f) * half; hp[i] = 1.0f;
    }
    count = target = N;
    LARGE_INTEGER f, t0, t1; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t0);
    static const uint32_t BK[4] = {1, 8, 4, 2};
    static const float AX[4] = {10, 0, -10, 0}, AZ[4] = {0, 10, 0, -10};
    for (uint32_t t = 0; t < T; t++) {
        uint32_t k = BK[(tick / 120) & 3] | 16;
        if (tick % 180 == 0) k |= 32;
        if (tick % 300 == 60) k |= 64;
        if ((tick / 90) % 2 == 1) k |= 128;
        if (tick % 600 == 120) k |= 256;
        step(k, hx + AX[(tick / 30) & 3], hz + AZ[(tick / 30) & 3]);
    }
    QueryPerformanceCounter(&t1);
    uint64_t h = 0xcbf29ce484222325ULL;
    h = fnv(h, &count, 4); h = fnv(h, &tick, 4); h = fnv(h, &hx, 4); h = fnv(h, &hz, 4);
    h = fnv(h, px, count * 4); h = fnv(h, pz, count * 4); h = fnv(h, vx, count * 4); h = fnv(h, vz, count * 4);
    h = fnv(h, hp, count * 4); h = fnv(h, &hhp, 4); h = fnv(h, &alive, 4); h = fnv(h, &kills, 4); h = fnv(h, &npr, 4);
    h = fnv(h, prx, npr * 4); h = fnv(h, prz, npr * 4);
    h = fnv(h, &energy, 4); h = fnv(h, &pulse_r, 4); h = fnv(h, &dash_ticks, 4); h = fnv(h, &pulse_on, 4); h = fnv(h, &bomb_timer, 4);
    printf("us_per_tick=%llu\n", (unsigned long long)((t1.QuadPart - t0.QuadPart) * 1000000 / f.QuadPart / (T ? T : 1)));
    printf("hash=%016llx\n", (unsigned long long)h);
    return 0;
}
