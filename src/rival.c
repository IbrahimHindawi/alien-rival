/* rival: SIM 5 swarm simulation, optimised. See NOTES.md for the design. */
#include <immintrin.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define GD 1024u
#define GC (GD * GD)
#define MAXE (1u << 20)
#define CAP (MAXE + 4096u)
#define MAXT 64
#define MAXB 1024               /* bands per tick, at most */
#define MOVED 0x80000000u       /* SEQ flag: the entity changed cell (within its row) this tick */

static const float DT = 0.016666668f, HALF_DT = 0.008333334f;
static const float LO = -1023.5f, HI = 1023.5f, EPS = 1.0e-4f;

/* ---------------------------------------------------------------- state */
/* The grid order of tick t (the "layout"): entities sorted by (cell of the
   start-of-tick position, creation sequence). The spec's index order is
   sequence order: deaths compact stably and spawns append, so relative order
   never changes and the hash can be taken in sequence order at the end.

   Tick t, slot s of layout t:
     OUTX/OUTZ/OUTVX/OUTVZ[t&1][s]  state after integration (NaN x, z: dead)
     SEQ[t&1][s]                    creation sequence (| MOVED)
     CS[t&1]                        cell_start of layout t
   The start-of-tick positions and velocities in layout order never exist as
   whole arrays: each band of rows rebuilds them, a row at a time, into
   small per-thread row buffers that stay in cache while the separation of
   the band reads them (merge and separation are one fused pass). */
static float *OUTX[2], *OUTZ[2], *OUTVX[2], *OUTVZ[2];
static uint32_t *SEQ[2];
static uint32_t *CS[2];
static uint32_t *STAYC[2];      /* per cell: entities that stayed in their row (next layout) */
static uint32_t ROWST[2][GD];   /* per row: same, summed */
static uint32_t *MROW[2][MAXT]; /* per thread, per row: row changers landing there */
typedef struct { uint32_t s, c; } mover;
static mover *MOV[2][MAXT];     /* row changers (and spawns), per thread */
static uint32_t *MOFF[2][MAXT];  /* per thread, per row: start of the row's movers in MOV (sorted by row) */
static mover *MOVTMP[MAXT];
static mover SPAWNL[2][64];     /* this tick's spawns (row changers from nowhere) */
static uint32_t SPAWNN[2];
static float *HPS;              /* hp by creation sequence (combat touches few entities) */
/* current output, for the serial part and the hash */
static float *X1, *Z1, *VX1, *VZ1;

static uint32_t g_nslots;       /* slots in the current layout (live + dead + spawned tail) */
static uint32_t g_count, g_target, g_seq_next;
static uint32_t g_tick;
static float hx, hz, hhp = 100.0f;
static uint32_t alive = 1, cool, kills, spawned, shots;
static uint32_t dash_ticks, dash_cd, pulse_cd, pulse_on, beam_on, bomb_cd, bomb_timer;
static float dash_x, dash_z, pulse_r, pulse_cx, pulse_cz, energy = 100.0f, bomb_x, bomb_z;
static float prx[4096], prz[4096], prvx[4096], prvz[4096];
static uint32_t prl[4096], prp[4096], npr;
static uint32_t g_keys;
static float g_aimx, g_aimz;

typedef struct { uint64_t state, inc; } pcg;
static pcg rng;

static int g_threads = 1;
static float g_maxdisp[MAXT * 16];
typedef struct { _Atomic uint64_t fb; char pad[56]; } bandq;   /* front << 32 | back */
static _Alignas(64) bandq g_q[MAXT];
static uint32_t g_nbands;
static uint32_t g_band_r[MAXB + 1];   /* band k = rows [g_band_r[k], g_band_r[k+1]) */

/* ---------------------------------------------------------------- helpers */
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
static inline uint32_t cellx(float x) { return (uint32_t)(int)((x + 1024.0f) * 0.5f); }
static uint32_t cell_of(float x, float z) { return cellx(z) * GD + cellx(x); }
static void aim_dir(float ax, float az, float *ox, float *oz) {
    float dx = ax - hx, dz = az - hz, d2 = dx * dx + dz * dz;
    if (d2 > EPS) { float inv = 1.0f / sqrtf(d2); *ox = dx * inv; *oz = dz * inv; }
    else { *ox = 0.0f; *oz = 1.0f; }
}
/* Every array starts at a different offset modulo 4 KiB, so the streams do
   not share L1 sets or 4K-alias. */
static void *zalloc(size_t bytes) {
    static unsigned k;
    size_t skew = (size_t)((k++ * 17u) % 64u) * 64u;
    char *p = VirtualAlloc(NULL, bytes + 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) { fprintf(stderr, "out of memory%c", 10); exit(1); }
    return p + skew;
}

/* inclusive prefix sum of 8 lanes */
static inline __m256i prefix8(__m256i k) {
    __m256i p = _mm256_add_epi32(k, _mm256_slli_si256(k, 4));
    p = _mm256_add_epi32(p, _mm256_slli_si256(p, 8));
    __m256i hi = _mm256_shuffle_epi32(p, 0xFF);
    return _mm256_add_epi32(p, _mm256_permute2x128_si256(hi, hi, 0x08));
}

/* ---------------------------------------------------------------- barrier */
static _Alignas(64) _Atomic int bar_count;
static _Alignas(64) _Atomic int bar_gen;
static void barrier(void) {
    if (g_threads == 1) return;
    int g = atomic_load_explicit(&bar_gen, memory_order_acquire);
    if (atomic_fetch_add_explicit(&bar_count, 1, memory_order_acq_rel) == g_threads - 1) {
        atomic_store_explicit(&bar_count, 0, memory_order_relaxed);
        atomic_store_explicit(&bar_gen, g + 1, memory_order_release);
    } else {
        /* spin first (phases are short); yield if it takes long, so that more
           threads than free logical processors still make progress */
        for (int k = 0; atomic_load_explicit(&bar_gen, memory_order_acquire) == g; k++) {
            if (k < 32768) _mm_pause();
            else SwitchToThread();
        }
    }
}

/* ---------------------------------------------------------------- per-thread workspace */
typedef struct {
    /* row buffers: 3 rows of the new layout (x, z, vx, vz, seq) */
    float *rx[3], *rz[3], *rvx[3], *rvz[3];
    uint32_t *rsq[3];
    uint32_t rcap;
    uint32_t rcs[3][GD + 1];    /* cell starts (global slots) of the rows in the buffers */
    /* movers landing in the band's rows, bucketed by row */
    mover *bm; uint32_t bmcap;
    mover *bm2;
    uint32_t bmrow[GD + 2];
    uint32_t cur[GD];           /* cell counts of the row being built */
    /* kept entities of the row being built (same cell as last tick), compacted in old order */
    float *kx, *kz, *kvx, *kvz; uint32_t *kq; uint64_t *kk;
    /* inserted entities (changed cell), sorted by (cell x, sequence) before the merge */
    struct ins { uint64_t key; uint32_t s, next; } *ins, *ins2;
    uint32_t ihead[GD];         /* per cell: last inserted entity (linked through .next) */
    uint64_t ibits[GD / 64];    /* cells holding inserted entities */
    /* outputs of the separation pass */
    uint32_t *stayc; uint32_t *rowst; uint32_t *mrow;
    mover *mov; uint32_t nmov;
    uint32_t *seqn;
    float *ox, *oz, *ovx, *ovz;
    uint32_t rowbase[GD + 1];   /* slot of the first entity of each row of layout t */
    uint32_t last_r1, last_t;   /* the band this thread did last: its rows stay in the buffers */
    float *sepx, *sepz;         /* separation sums of the current row */
    __m256 vmd;
} work;
static work *g_work[MAXT];

static void *xmalloc(size_t n) {
    void *p = _aligned_malloc(n, 64);
    if (!p) { fprintf(stderr, "out of memory%c", 10); exit(1); }
    return p;
}
static void ensure_rows(work *w, uint32_t need) {
    if (need <= w->rcap) return;
    uint32_t cap = need + need / 2 + 64;
    for (int k = 0; k < 3; k++) {
        _aligned_free(w->rx[k]); _aligned_free(w->rz[k]); _aligned_free(w->rvx[k]); _aligned_free(w->rvz[k]);
        _aligned_free(w->rsq[k]);
        w->rx[k] = xmalloc((size_t)cap * 4 + 64); w->rz[k] = xmalloc((size_t)cap * 4 + 64);
        w->rvx[k] = xmalloc((size_t)cap * 4 + 64); w->rvz[k] = xmalloc((size_t)cap * 4 + 64);
        w->rsq[k] = xmalloc((size_t)cap * 4 + 64);
        memset(w->rx[k], 0, (size_t)cap * 4 + 64); memset(w->rz[k], 0, (size_t)cap * 4 + 64);
    }
    _aligned_free(w->kx); _aligned_free(w->kz); _aligned_free(w->kvx); _aligned_free(w->kvz);
    _aligned_free(w->kq); _aligned_free(w->kk); _aligned_free(w->ins); _aligned_free(w->ins2);
    _aligned_free(w->sepx); _aligned_free(w->sepz);
    w->sepx = xmalloc((size_t)cap * 4 + 64); w->sepz = xmalloc((size_t)cap * 4 + 64);
    w->kx = xmalloc((size_t)cap * 4 + 64); w->kz = xmalloc((size_t)cap * 4 + 64);
    w->kvx = xmalloc((size_t)cap * 4 + 64); w->kvz = xmalloc((size_t)cap * 4 + 64);
    w->kq = xmalloc((size_t)cap * 4 + 64); w->kk = xmalloc((size_t)cap * 8 + 64);
    w->ins = xmalloc((size_t)cap * sizeof(*w->ins) + 64); w->ins2 = xmalloc((size_t)cap * sizeof(*w->ins) + 64);
    w->rcap = cap;
}

/* ---------------------------------------------------------------- separation + integration */
/* Lanes are 8 or 16 consecutive entities of one row (one or two vectors).
   Every candidate of the window (rows z-1..z+1, cells from the first lane's
   cx-1 to the last lane's cx+1) is broadcast to the lanes in sorted order. A candidate
   outside a lane's own 3x3 block is at least 2 m away on some axis, so
   d2 >= 4 and it never contributes: each lane sees exactly its spec
   neighbours, in spec order, and sums them sequentially. Masked lanes add
   +-0.0f, which is exact (the running sum starts at +0.0f and so is never -0.0f). */
#define SEP_HALF(PX, PZ, SX, SZ, N, SLOTV, XJ, ZJ, J, SELF, CAP)                                    \
    do {                                                                                          \
        __m256 dx_ = _mm256_sub_ps(PX, XJ), dz_ = _mm256_sub_ps(PZ, ZJ);                          \
        __m256 d2_ = _mm256_add_ps(_mm256_mul_ps(dx_, dx_), _mm256_mul_ps(dz_, dz_));             \
        __m256 m_ = _mm256_cmp_ps(d2_, one, _CMP_LT_OQ);                                          \
        if (CAP) m_ = _mm256_and_ps(m_, _mm256_castsi256_ps(_mm256_cmpgt_epi32(sixteen, N)));     \
        if (SELF) m_ = _mm256_andnot_ps(                                                          \
            _mm256_castsi256_ps(_mm256_cmpeq_epi32(SLOTV, _mm256_set1_epi32((int)(J)))), m_);     \
        __m256 w_ = _mm256_and_ps(m_, _mm256_sub_ps(one, d2_));                                   \
        SX = _mm256_add_ps(SX, _mm256_mul_ps(dx_, w_));                                           \
        SZ = _mm256_add_ps(SZ, _mm256_mul_ps(dz_, w_));                                           \
        N = _mm256_sub_epi32(N, _mm256_castps_si256(m_));                                         \
    } while (0)
#define SEPG 2                  /* vectors per group: 16 lanes */
#define SEP_CANDAB(XR, ZR, J, SELF, CAP, DOA, DOB)                                                \
    do {                                                                                          \
        __m256 xj_ = _mm256_broadcast_ss((XR) + (J)), zj_ = _mm256_broadcast_ss((ZR) + (J));      \
        if (DOA) SEP_HALF(px[0], pz[0], sx[0], sz[0], nn[0], slot[0], xj_, zj_, J, SELF, CAP);    \
        if (DOB) SEP_HALF(px[1], pz[1], sx[1], sz[1], nn[1], slot[1], xj_, zj_, J, SELF, CAP);    \
    } while (0)
/* candidates [LO, HI) of window row k for the halves DOA / DOB; in the
   group's own row the lanes' own slots get the self test */
#define SEP_SEG(LO, HI, DOA, DOB, CAP)                                                            \
    do {                                                                                          \
        uint32_t j_ = (LO), e_ = (HI);                                                            \
        if (CAP && k == self_k) {                                                                      \
            uint32_t s0_ = g0 > j_ ? g0 : j_, s1_ = g0 + ng < e_ ? g0 + ng : e_;                  \
            if (s0_ > e_) s0_ = e_;                                                               \
            if (s1_ < s0_) s1_ = s0_;                                                             \
            for (; j_ < s0_; j_++) SEP_CANDAB(xr, zr, j_, 0, CAP, DOA, DOB);                      \
            for (; j_ < s1_; j_++) SEP_CANDAB(xr, zr, j_, 1, CAP, DOA, DOB);                      \
        }                                                                                         \
        for (; j_ < e_; j_++) SEP_CANDAB(xr, zr, j_, 0, CAP, DOA, DOB);                           \
    } while (0)
/* Each half only needs the cells next to its own lanes: per window row the
   candidates split into A-only, both, and B-only runs (a skipped candidate
   is >= 2 m from every lane of that half, as above). */
#define SEP_GROUP(CAP)                                                                            \
    do {                                                                                          \
        for (int v_ = 0; v_ < SEPG; v_++) {                                                       \
            sx[v_] = sz[v_] = _mm256_setzero_ps(); nn[v_] = _mm256_setzero_si256();               \
        }                                                                                         \
        for (int k = 0; k < nwin; k++) {                                                          \
            const float *xr = wx[k], *zr = wz[k];                                                 \
            uint32_t aa = wcs[k][xal] - wbase[k], ab = wcs[k][xah] - wbase[k];                    \
            if (GL == 8 && !(CAP)) { /* past the window end: cells >= 2 away, or padding */ \
                for (uint32_t j4 = aa; j4 < ab; j4 += 4) { \
                    SEP_CANDAB(xr, zr, j4, 0, 0, 1, 0); SEP_CANDAB(xr, zr, j4 + 1, 0, 0, 1, 0); \
                    SEP_CANDAB(xr, zr, j4 + 2, 0, 0, 1, 0); SEP_CANDAB(xr, zr, j4 + 3, 0, 0, 1, 0); \
                } \
                continue; \
            } \
            uint32_t ba = ng > 8 ? wcs[k][xbl] - wbase[k] : ab, bb = ng > 8 ? wcs[k][xbh] - wbase[k] : ab; \
            uint32_t m1 = ab < ba ? ab : ba, m2 = ab > ba ? ab : ba;                              \
            if (!(CAP)) { /* each half over its own window, in fours; past a window's end the cells are >= 2 away, or padding */ \
                for (uint32_t j4 = aa; j4 < ab; j4 += 4) { \
                    SEP_CANDAB(xr, zr, j4, 0, 0, 1, 0); SEP_CANDAB(xr, zr, j4 + 1, 0, 0, 1, 0); \
                    SEP_CANDAB(xr, zr, j4 + 2, 0, 0, 1, 0); SEP_CANDAB(xr, zr, j4 + 3, 0, 0, 1, 0); \
                } \
                for (uint32_t j4 = ba; j4 < bb; j4 += 4) { \
                    SEP_CANDAB(xr, zr, j4, 0, 0, 0, 1); SEP_CANDAB(xr, zr, j4 + 1, 0, 0, 0, 1); \
                    SEP_CANDAB(xr, zr, j4 + 2, 0, 0, 0, 1); SEP_CANDAB(xr, zr, j4 + 3, 0, 0, 0, 1); \
                } \
                continue; \
            } \
            SEP_SEG(aa, m1, 1, 0, CAP);                                                           \
            if (ba < ab) SEP_SEG(ba, ab, 1, 1, CAP);                                              \
            SEP_SEG(m2, bb, 0, 1, CAP);                                                           \
        }                                                                                         \
    } while (0)

/* integrate 8 lanes: local slots l..l+ng of the row buffer, global slot gs */
static inline void integrate8(work *w, const float *vx0, const float *vz0, uint32_t l, uint32_t gs, uint32_t ng,
                              __m256 px, __m256 pz, __m256 sx, __m256 sz, uint32_t cz, uint32_t *stay) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 wh = _mm256_set1_ps(1024.0f), half = _mm256_set1_ps(0.5f);
    __m256 dx = _mm256_sub_ps(_mm256_set1_ps(hx), px), dz = _mm256_sub_ps(_mm256_set1_ps(hz), pz);
    __m256 d2 = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dz, dz));
    __m256 m = _mm256_cmp_ps(d2, _mm256_set1_ps(0.81f), _CMP_GT_OQ);
    __m256 inv = _mm256_div_ps(one, _mm256_sqrt_ps(_mm256_blendv_ps(one, d2, m)));
    const __m256 seven = _mm256_set1_ps(7.0f), six = _mm256_set1_ps(6.0f);
    __m256 tx = _mm256_and_ps(m, _mm256_mul_ps(_mm256_mul_ps(dx, inv), seven));
    __m256 tz = _mm256_and_ps(m, _mm256_mul_ps(_mm256_mul_ps(dz, inv), seven));
    tx = _mm256_add_ps(tx, _mm256_mul_ps(sx, six));
    tz = _mm256_add_ps(tz, _mm256_mul_ps(sz, six));
    __m256 vx = _mm256_loadu_ps(vx0 + l), vz = _mm256_loadu_ps(vz0 + l);
    const __m256 steer = _mm256_set1_ps(0.125f), dt = _mm256_set1_ps(DT);
    vx = _mm256_add_ps(vx, _mm256_mul_ps(_mm256_sub_ps(tx, vx), steer));
    vz = _mm256_add_ps(vz, _mm256_mul_ps(_mm256_sub_ps(tz, vz), steer));
    const __m256 lo = _mm256_set1_ps(LO), hi = _mm256_set1_ps(HI);
    __m256 nx = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(px, _mm256_mul_ps(vx, dt)), lo), hi);
    __m256 nz = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(pz, _mm256_mul_ps(vz, dt)), lo), hi);
    __m256i cxn = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_add_ps(nx, wh), half));
    __m256i czn = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_add_ps(nz, wh), half));
    __m256i cxo = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_add_ps(px, wh), half));
    __m256i cn = _mm256_add_epi32(_mm256_slli_epi32(czn, 10), cxn);
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 disp = _mm256_max_ps(_mm256_and_ps(_mm256_sub_ps(nx, px), absm), _mm256_and_ps(_mm256_sub_ps(nz, pz), absm));
    __m256i lm = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)ng), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    if (ng == 8) {
        _mm256_storeu_ps(w->ox + gs, nx); _mm256_storeu_ps(w->oz + gs, nz);
        _mm256_storeu_ps(w->ovx + gs, vx); _mm256_storeu_ps(w->ovz + gs, vz);
        w->vmd = _mm256_max_ps(w->vmd, disp);
    } else {
        _mm256_maskstore_ps(w->ox + gs, lm, nx); _mm256_maskstore_ps(w->oz + gs, lm, nz);
        _mm256_maskstore_ps(w->ovx + gs, lm, vx); _mm256_maskstore_ps(w->ovz + gs, lm, vz);
        w->vmd = _mm256_max_ps(w->vmd, _mm256_and_ps(_mm256_castsi256_ps(lm), disp));
    }
    const __m256i czv = _mm256_set1_epi32((int)cz);
    __m256i st = _mm256_and_si256(_mm256_cmpeq_epi32(czn, czv), lm);
    __m256i moved = _mm256_andnot_si256(_mm256_cmpeq_epi32(cxn, cxo), st);
    uint32_t ms = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(st));
    uint32_t mm = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(moved));
    uint32_t mrow = ((1u << ng) - 1) & ~ms;
    *stay += (uint32_t)_mm_popcnt_u32(ms);
    if (!(mm | mrow)) return;
    /* the row's counts start as this layout's cell counts: only cell changers adjust them */
    _Alignas(32) uint32_t cc[8], co[8];
    _mm256_store_si256((__m256i *)cc, cn);
    _mm256_store_si256((__m256i *)co, cxo);
    uint32_t *cnt = w->stayc + cz * GD;
    while (mm) {
        uint32_t k = (uint32_t)_tzcnt_u32(mm); mm &= mm - 1;
        w->seqn[gs + k] |= MOVED;
        cnt[co[k]]--; cnt[cc[k] & (GD - 1)]++;
    }
    while (mrow) {
        uint32_t k = (uint32_t)_tzcnt_u32(mrow); mrow &= mrow - 1;
        cnt[co[k]]--;
        w->mov[w->nmov].s = gs + k; w->mov[w->nmov].c = cc[k]; w->nmov++;
        w->mrow[cc[k] / GD]++;
    }
}

/* the candidate pass of one row, in groups of GL = 8 or 16 lanes */
static inline __attribute__((always_inline)) void sep_groups(work *w, const float *X0, const float *Z0, uint32_t n,
        const float *const *wx, const float *const *wz, const uint32_t *const *wcs, const uint32_t *wbase,
        int nwin, int self_k, const uint32_t GL) {
    const __m256 one = _mm256_set1_ps(1.0f), big = _mm256_set1_ps(1.0e30f);
    const __m256i sixteen = _mm256_set1_epi32(16), lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    int capped = 0;
    for (uint32_t g0 = 0; g0 < n; g0 += GL) {
        uint32_t ng = n - g0 < GL ? n - g0 : GL;
        __m256 px[SEPG], pz[SEPG], sx[SEPG], sz[SEPG];
        __m256i nn[SEPG], slot[SEPG];
        for (int v = 0; v < SEPG; v++) {
            px[v] = _mm256_loadu_ps(X0 + g0 + 8 * v); pz[v] = _mm256_loadu_ps(Z0 + g0 + 8 * v);
            slot[v] = _mm256_add_epi32(_mm256_set1_epi32((int)(g0 + 8 * v)), lane);
            if (ng < 8 * SEPG) {
                __m256 va = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32((int)ng - 8 * v), lane));
                px[v] = _mm256_blendv_ps(big, px[v], va); pz[v] = _mm256_blendv_ps(big, pz[v], va);
            }
        }
        uint32_t na8 = ng < 8 ? ng : 8;
        uint32_t ca0 = cellx(X0[g0]), ca1 = cellx(X0[g0 + na8 - 1]);
        uint32_t xal = ca0 > 0 ? ca0 - 1 : 0, xah = ca1 < GD - 1 ? ca1 + 2 : GD, xbl = 0, xbh = 0;
        if (ng > 8) {
            uint32_t cb0 = cellx(X0[g0 + 8]), cb1 = cellx(X0[g0 + ng - 1]);
            xbl = cb0 > 0 ? cb0 - 1 : 0; xbh = cb1 < GD - 1 ? cb1 + 2 : GD;
        }
        if (!capped) {
            /* without the cap the self test is skipped: a lane's own slot is a
               d2 = 0 hit that adds +0.0f to the sums and one to the count */
            SEP_GROUP(0);
            __m256i over = _mm256_setzero_si256(), seventeen = _mm256_set1_epi32(17);
            for (int v = 0; v < SEPG; v++) over = _mm256_or_si256(over, _mm256_cmpgt_epi32(nn[v], seventeen));
            if (!_mm256_testz_si256(over, over)) { capped = 1; SEP_GROUP(1); }
        } else {
            SEP_GROUP(1);
            __m256i full = _mm256_setzero_si256();
            for (int v = 0; v < SEPG; v++) full = _mm256_or_si256(full, _mm256_cmpeq_epi32(nn[v], sixteen));
            capped = !_mm256_testz_si256(full, full);
        }
        for (int v = 0; v < SEPG; v++) {
            _mm256_storeu_ps(w->sepx + g0 + 8 * v, sx[v]);
            _mm256_storeu_ps(w->sepz + g0 + 8 * v, sz[v]);
        }
    }
}

/* separation + integration of row cz, held in buffer bi; buffers bl / bh hold
   rows cz-1 / cz+1 (or -1 when outside the grid) */
static void sep_row(work *w, uint32_t cz, int bl, int bi, int bh) {
    const uint32_t *ocs = w->rcs[bi];
    uint32_t rbase = ocs[0], n = ocs[GD] - rbase;
    w->rowst[cz] = 0;
    {   /* counts of the next layout for this row: start from this layout's */
        uint32_t *stc = w->stayc + cz * GD;
        for (uint32_t c = 0; c < GD; c += 8)
            _mm256_storeu_si256((__m256i *)(stc + c), _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)(ocs + c + 1)),
                                                                        _mm256_loadu_si256((const __m256i *)(ocs + c))));
    }
    if (n == 0) return;
    const float *wx[3], *wz[3]; const uint32_t *wcs[3]; uint32_t wbase[3];
    int nwin = 0, self_k = 0;
    int bufs[3] = {bl, bi, bh};
    for (int k = 0; k < 3; k++) {
        int b = bufs[k];
        if (b < 0) continue;
        if (b == bi) self_k = nwin;
        wx[nwin] = w->rx[b]; wz[nwin] = w->rz[b]; wcs[nwin] = w->rcs[b]; wbase[nwin] = w->rcs[b][0];
        nwin++;
    }
    const float *X0 = w->rx[bi], *Z0 = w->rz[bi];
    uint32_t stay = 0;
    /* sparse rows: 8-lane groups (the halves of a 16-lane group would share
       almost no candidates); dense rows: 16 lanes sharing each broadcast */
    {
        uint32_t span = cellx(X0[n - 1]) - cellx(X0[0]) + 1;
        if (n < 4 * span) sep_groups(w, X0, Z0, n, wx, wz, wcs, wbase, nwin, self_k, 8);
        else sep_groups(w, X0, Z0, n, wx, wz, wcs, wbase, nwin, self_k, 16);
    }
    /* integration of the whole row: independent iterations, so the long
       sqrt/div chains of consecutive vectors overlap */
    for (uint32_t l = 0; l < n; l += 8) {
        uint32_t nv = n - l < 8 ? n - l : 8;
        integrate8(w, w->rvx[bi], w->rvz[bi], l, rbase + l, nv, _mm256_loadu_ps(X0 + l), _mm256_loadu_ps(Z0 + l),
                   _mm256_loadu_ps(w->sepx + l), _mm256_loadu_ps(w->sepz + l), cz, &stay);
    }
    w->rowst[cz] = stay;
}

/* ---------------------------------------------------------------- the fused pass */
/* Band k owns rows [r0, r1). It rebuilds rows r0-1 .. r1 of layout t (its
   own rows plus one neighbour row each side, which it needs as separation
   candidates; the lower two are already there when the same thread just did
   the band below) from the output of tick t-1. The integration of tick t-1
   counted, per cell, the entities that stayed in their row, flagged the ones
   that changed cell within it (MOVED) and listed the ones that changed row.
   As soon as row r+1 is built, row r is separated and integrated. */
static inline void copy_kept(work *w, int bi, uint32_t d, uint32_t j, uint32_t e) {
    float *X = w->rx[bi] + d, *Z = w->rz[bi] + d, *VX = w->rvx[bi] + d, *VZ = w->rvz[bi] + d;
    uint32_t *Q = w->rsq[bi] + d;
    for (uint32_t k = 0; k < e - j; k += 8) {
        _mm256_storeu_ps(X + k, _mm256_loadu_ps(w->kx + j + k));
        _mm256_storeu_ps(Z + k, _mm256_loadu_ps(w->kz + j + k));
        _mm256_storeu_ps(VX + k, _mm256_loadu_ps(w->kvx + j + k));
        _mm256_storeu_ps(VZ + k, _mm256_loadu_ps(w->kvz + j + k));
        _mm256_storeu_si256((__m256i *)(Q + k), _mm256_loadu_si256((const __m256i *)(w->kq + j + k)));
    }
}

static inline void add_ins(work *w, uint32_t i, uint32_t c, uint32_t q, uint32_t s) {
    w->ins[i].key = ((uint64_t)c << 32) | q;
    w->ins[i].s = s;
    uint64_t b = 1ull << (c & 63);
    if (w->ibits[c >> 6] & b) w->ins[i].next = w->ihead[c];
    else { w->ibits[c >> 6] |= b; w->ins[i].next = 0xffffffffu; }
    w->ihead[c] = i;
}

/* Row r of layout t into row buffer bi. Kept entities (same cell as in
   layout t-1) keep their relative order, which is already (cell, sequence)
   order: they are compacted with a vector left-pack. The entities that
   changed cell (within the row, flagged MOVED, or from another row) are
   bucketed by cell, sorted by sequence within it, and merged in. */
static void build_row(work *w, uint32_t t, uint32_t r, int bi, uint32_t rbase, int own, uint32_t rlo) {
    const uint32_t *stc = STAYC[t & 1] + r * GD;
    const uint32_t *cso = CS[(t + 1) & 1];
    const float *ox = OUTX[(t + 1) & 1], *oz = OUTZ[(t + 1) & 1], *ovx = OUTVX[(t + 1) & 1], *ovz = OUTVZ[(t + 1) & 1];
    const uint32_t *sqo = SEQ[(t + 1) & 1];
    uint32_t *cs = w->rcs[bi];
    const mover *bm = w->bm2 + w->bmrow[r - rlo];
    uint32_t nbm = w->bmrow[r - rlo + 1] - w->bmrow[r - rlo];
    /* counts -> cell starts (global slots) */
    memcpy(w->cur, stc, GD * 4);
    for (uint32_t i = 0; i < nbm; i++) w->cur[bm[i].c & (GD - 1)]++;
    {
        uint32_t *gcs = own ? CS[t & 1] + r * GD : NULL;
        __m256i carry = _mm256_set1_epi32((int)rbase);
        const __m256i seven = _mm256_set1_epi32(7);
        for (uint32_t c = 0; c < GD; c += 8) {
            __m256i k = _mm256_loadu_si256((const __m256i *)(w->cur + c));
            __m256i p = prefix8(k);
            __m256i st = _mm256_add_epi32(_mm256_sub_epi32(p, k), carry);
            _mm256_storeu_si256((__m256i *)(cs + c), st);
            if (gcs) _mm256_storeu_si256((__m256i *)(gcs + c), st);
            carry = _mm256_add_epi32(carry, _mm256_permutevar8x32_epi32(p, seven));
        }
        cs[GD] = (uint32_t)_mm256_cvtsi256_si32(carry);
    }
    uint32_t run = cs[GD];
    /* scan the old row: compact the kept, list the cell changers */
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256 wh = _mm256_set1_ps(1024.0f), half = _mm256_set1_ps(0.5f);
    const __m256i movedv = _mm256_set1_epi32((int)MOVED);
    uint32_t sa = cso[r * GD], sb = cso[(r + 1) * GD];
    const __m256i rv = _mm256_set1_epi32((int)r);
    uint32_t nk = 0, ni = 0;
    memset(w->ibits, 0, sizeof w->ibits);
    for (uint32_t base = sa; base < sb; base += 8) {
        __m256 x = _mm256_loadu_ps(ox + base), z = _mm256_loadu_ps(oz + base);
        __m256i cx = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_add_ps(x, wh), half));
        __m256i cz = _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_add_ps(z, wh), half));
        __m256i sq = _mm256_loadu_si256((const __m256i *)(sqo + base));
        __m256i stay = _mm256_and_si256(_mm256_cmpeq_epi32(cz, rv),
                                        _mm256_cmpgt_epi32(_mm256_set1_epi32((int)(sb - base)), lane));
        __m256i mv = _mm256_and_si256(stay, _mm256_cmpeq_epi32(_mm256_and_si256(sq, movedv), movedv));
        uint32_t mk = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_andnot_si256(mv, stay)));
        uint32_t mm = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(mv));
        /* left-pack the kept lanes */
        uint64_t idx = _pext_u64(0x0706050403020100ull, _pdep_u64(mk, 0x0101010101010101ull) * 0xFF);
        __m256i perm = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)idx));
        _mm256_storeu_ps(w->kx + nk, _mm256_permutevar8x32_ps(x, perm));
        _mm256_storeu_ps(w->kz + nk, _mm256_permutevar8x32_ps(z, perm));
        _mm256_storeu_ps(w->kvx + nk, _mm256_permutevar8x32_ps(_mm256_loadu_ps(ovx + base), perm));
        _mm256_storeu_ps(w->kvz + nk, _mm256_permutevar8x32_ps(_mm256_loadu_ps(ovz + base), perm));
        __m256i pq = _mm256_permutevar8x32_epi32(sq, perm), pc = _mm256_permutevar8x32_epi32(cx, perm);
        _mm256_storeu_si256((__m256i *)(w->kq + nk), pq);
        /* 64-bit keys (cell x << 32 | sequence) for the merge */
        _mm256_storeu_si256((__m256i *)(w->kk + nk), _mm256_unpacklo_epi32(_mm256_permute4x64_epi64(pq, 0x50), _mm256_permute4x64_epi64(pc, 0x50)));
        _mm256_storeu_si256((__m256i *)(w->kk + nk + 4), _mm256_unpacklo_epi32(_mm256_permute4x64_epi64(pq, 0xFA), _mm256_permute4x64_epi64(pc, 0xFA)));
        nk += (uint32_t)_mm_popcnt_u32(mk);
        if (mm) {
            _Alignas(32) uint32_t cc[8], qq[8];
            _mm256_store_si256((__m256i *)cc, cx);
            _mm256_store_si256((__m256i *)qq, sq);
            while (mm) {
                uint32_t q = (uint32_t)_tzcnt_u32(mm); mm &= mm - 1;
                add_ins(w, ni++, cc[q], qq[q] & ~MOVED, base + q);
            }
        }
    }
    /* row changers */
    for (uint32_t i = 0; i < nbm; i++) {
        uint32_t s = bm[i].s;
        add_ins(w, ni++, bm[i].c & (GD - 1), sqo[s] & ~MOVED, s);
    }
    /* merge, cell by cell where entities were inserted */
    float *X = w->rx[bi], *Z = w->rz[bi], *VX = w->rvx[bi], *VZ = w->rvz[bi];
    uint32_t *Q = w->rsq[bi];
    uint32_t j = 0, d = 0, before = 0;
    const uint64_t *kk = w->kk;
    struct ins *it = w->ins2;
    for (uint32_t wd = 0; wd < GD / 64; wd++) {
        uint64_t bits = w->ibits[wd];
        while (bits) {
            uint32_t c = wd * 64 + (uint32_t)_tzcnt_u64(bits); bits &= bits - 1;
            /* the inserted entities of cell c, by sequence */
            uint32_t n = 0;
            for (uint32_t i = w->ihead[c]; i != 0xffffffffu; i = w->ins[i].next) {
                struct ins v = w->ins[i];
                uint32_t k = n++;
                while (k > 0 && it[k - 1].key > v.key) { it[k] = it[k - 1]; k--; }
                it[k] = v;
            }
            /* kept entities of the cells before c go first */
            uint32_t kb = cs[c] - rbase - before;
            if (kb > j) { copy_kept(w, bi, d, j, kb); d += kb - j; j = kb; }
            for (uint32_t k = 0; k < n; k++) {
                uint64_t key = it[k].key;
                while (j < nk && kk[j] < key) {
                    X[d] = w->kx[j]; Z[d] = w->kz[j]; VX[d] = w->kvx[j]; VZ[d] = w->kvz[j]; Q[d] = w->kq[j]; d++; j++;
                }
                uint32_t s = it[k].s;
                X[d] = ox[s]; Z[d] = oz[s]; VX[d] = ovx[s]; VZ[d] = ovz[s]; Q[d] = (uint32_t)key; d++;
            }
            before += n;
        }
    }
    if (j < nk) { copy_kept(w, bi, d, j, nk); d += nk - j; }
    for (int k = 0; k < 8; k++) { X[d + k] = 1.0e30f; Z[d + k] = 1.0e30f; }   /* padding: never a neighbour */
    if (own) memcpy(SEQ[t & 1] + rbase, Q, (size_t)(run - rbase) * 4);
}

static void band_pass(int tid, uint32_t t, uint32_t r0, uint32_t r1) {
    work *w = g_work[tid];
    uint32_t rlo = r0 > 0 ? r0 - 1 : 0, rhi = r1 < GD ? r1 : GD - 1;   /* rows needed, inclusive */
    const uint32_t *rbs = w->rowbase;
    uint32_t maxrow = 0;
    for (uint32_t r = rlo; r <= rhi; r++) if (rbs[r + 1] - rbs[r] > maxrow) maxrow = rbs[r + 1] - rbs[r];
    /* when this thread just did the band below, rows r0-1 and r0 are already
       in the row buffers (r0 as a neighbour row: publish it now) */
    int carry = r0 > 0 && w->last_t == t && w->last_r1 == r0 && maxrow + 16 <= w->rcap;
    uint32_t bstart = carry ? r0 + 1 : rlo;
    if (!carry) ensure_rows(w, maxrow + 16);
    w->last_t = t; w->last_r1 = r1;
    if (carry) {
        memcpy(CS[t & 1] + r0 * GD, w->rcs[r0 % 3], GD * 4);
        memcpy(SEQ[t & 1] + rbs[r0], w->rsq[r0 % 3], (size_t)(rbs[r0 + 1] - rbs[r0]) * 4);
    }
    /* row changers landing in rows bstart..rhi, bucketed by row */
    uint32_t nrows = bstart <= rhi ? rhi - bstart + 1 : 0, nb = 0;
    {
        uint32_t need = rbs[rhi + 1] - rbs[rlo];
        if (need + 16 > w->bmcap) {
            _aligned_free(w->bm); _aligned_free(w->bm2);
            w->bmcap = need + need / 2 + 1024;
            w->bm = xmalloc((size_t)w->bmcap * sizeof(mover)); w->bm2 = xmalloc((size_t)w->bmcap * sizeof(mover));
        }
    }
    const float *ox = OUTX[(t + 1) & 1];
    memset(w->bmrow, 0, (nrows + 2) * 4);
    uint32_t clo = bstart * GD, cspan = nrows * GD;
    if (nrows) for (int k = 0; k < g_threads; k++) {
        const mover *mv = MOV[t & 1][k];
        const uint32_t *mo = MOFF[t & 1][k];
        for (uint32_t i = mo[bstart], e = mo[rhi + 1]; i < e; i++) {
            if (ox[mv[i].s] != ox[mv[i].s]) continue;     /* died */
            w->bm[nb++] = mv[i];
            w->bmrow[(mv[i].c >> 10) - bstart + 1]++;
        }
    }
    for (uint32_t i = 0; i < SPAWNN[t & 1]; i++) {
        const mover *mv = &SPAWNL[t & 1][i];
        if (mv->c - clo >= cspan || ox[mv->s] != ox[mv->s]) continue;
        w->bm[nb++] = *mv;
        w->bmrow[(mv->c >> 10) - bstart + 1]++;
    }
    for (uint32_t r = 1; r <= nrows; r++) w->bmrow[r] += w->bmrow[r - 1];
    {
        uint32_t pos[GD + 1];
        memcpy(pos, w->bmrow, (nrows + 1) * 4);
        for (uint32_t i = 0; i < nb; i++) w->bm2[pos[(w->bm[i].c >> 10) - bstart]++] = w->bm[i];
    }
    /* build and separate, row by row; buffer of row r is r % 3 */
    for (uint32_t r = bstart; r <= rhi; r++) {
        build_row(w, t, r, (int)(r % 3), rbs[r], r >= r0 && r < r1, bstart);
        /* row r-1 has all its neighbours now */
        if (r >= 1 && r - 1 >= r0 && r - 1 < r1)
            sep_row(w, r - 1, r >= 2 && r - 2 >= rlo ? (int)((r - 2) % 3) : -1, (int)((r - 1) % 3), (int)(r % 3));
    }
    if (r1 == GD && r1 - 1 >= r0)   /* the last grid row has no row above it */
        sep_row(w, GD - 1, GD - 2 >= rlo ? (int)((GD - 2) % 3) : -1, (int)((GD - 1) % 3), -1);
    if (r1 == GD) CS[t & 1][GC] = rbs[GD];
}

/* bands of layout t, balanced by entity count of layout t-1 (any split is exact) */
static void make_bands(uint32_t t) {
    const uint32_t *cso = CS[(t + 1) & 1];
    uint32_t n = cso[GC];
    uint32_t nb = g_nbands;
    g_band_r[0] = 0;
    if (n == 0) {   /* first tick: no previous layout */
        for (uint32_t k = 1; k < nb; k++) g_band_r[k] = (uint32_t)((uint64_t)GD * k / nb);
        g_band_r[nb] = GD;
        return;
    }
    for (uint32_t k = 1; k < nb; k++) {
        uint32_t target = (uint32_t)((uint64_t)n * k / nb);
        uint32_t a = g_band_r[k - 1], b = GD;
        while (a < b) { uint32_t m = (a + b) / 2; if (cso[m * GD] >= target) b = m; else a = m + 1; }
        g_band_r[k] = a;
    }
    g_band_r[nb] = GD;
}

/* Each thread starts on its own contiguous run of bands and works through it
   in row order, so consecutive bands share their boundary rows. A thread
   that runs out takes the upper half of the largest remaining run. */
static void queues_init(void) {
    uint32_t nb = g_nbands, T = (uint32_t)g_threads;
    for (uint32_t k = 0; k < T; k++) {
        uint64_t f = (uint64_t)nb * k / T, b = (uint64_t)nb * (k + 1) / T;
        atomic_store_explicit(&g_q[k].fb, (f << 32) | b, memory_order_relaxed);
    }
}
static int band_pop(int tid, uint32_t *band) {
    for (;;) {
        uint64_t v = atomic_load_explicit(&g_q[tid].fb, memory_order_acquire);
        uint32_t f = (uint32_t)(v >> 32), b = (uint32_t)v;
        if (f < b) {
            if (atomic_compare_exchange_weak_explicit(&g_q[tid].fb, &v, ((uint64_t)(f + 1) << 32) | b,
                                                      memory_order_acq_rel, memory_order_relaxed)) { *band = f; return 1; }
            continue;
        }
        /* steal */
        int victim = -1; uint32_t best = 0;
        for (int k = 0; k < g_threads; k++) {
            uint64_t u = atomic_load_explicit(&g_q[k].fb, memory_order_relaxed);
            uint32_t len = (uint32_t)u - (uint32_t)(u >> 32);
            if ((uint32_t)(u >> 32) < (uint32_t)u && len > best) { best = len; victim = k; }
        }
        if (victim < 0) return 0;
        uint64_t u = atomic_load_explicit(&g_q[victim].fb, memory_order_acquire);
        uint32_t vf = (uint32_t)(u >> 32), vb = (uint32_t)u;
        if (vf >= vb) continue;
        uint32_t m = vf + (vb - vf) / 2;
        if (atomic_compare_exchange_weak_explicit(&g_q[victim].fb, &u, ((uint64_t)vf << 32) | m,
                                                  memory_order_acq_rel, memory_order_relaxed)) {
            atomic_store_explicit(&g_q[tid].fb, ((uint64_t)m << 32) | vb, memory_order_release);
        }
    }
}

static void fused_phase(int tid, uint32_t t) {
    work *w = g_work[tid];
    w->mov = MOV[(t + 1) & 1][tid];
    w->nmov = 0;
    w->mrow = MROW[(t + 1) & 1][tid];
    memset(w->mrow, 0, GD * 4);
    w->vmd = _mm256_setzero_ps();
    w->rowst = ROWST[(t + 1) & 1];
    w->stayc = STAYC[(t + 1) & 1];
    w->seqn = SEQ[t & 1];
    w->ox = OUTX[t & 1]; w->oz = OUTZ[t & 1]; w->ovx = OUTVX[t & 1]; w->ovz = OUTVZ[t & 1];
    /* row starts of layout t: stayers + row changers landing in each row */
    {
        const uint32_t *rst = ROWST[t & 1];
        __m256i carry = _mm256_setzero_si256();
        for (uint32_t r = 0; r < GD; r += 8) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(rst + r));
            for (int k = 0; k < g_threads; k++) v = _mm256_add_epi32(v, _mm256_loadu_si256((const __m256i *)(MROW[t & 1][k] + r)));
            __m256i p = prefix8(v);
            _mm256_storeu_si256((__m256i *)(w->rowbase + r), _mm256_add_epi32(_mm256_sub_epi32(p, v), carry));
            carry = _mm256_add_epi32(carry, _mm256_permutevar8x32_epi32(p, _mm256_set1_epi32(7)));
        }
        w->rowbase[GD] = (uint32_t)_mm256_cvtsi256_si32(carry);
    }
    uint32_t k;
    while (band_pop(tid, &k))
        if (g_band_r[k] < g_band_r[k + 1]) band_pass(tid, t, g_band_r[k], g_band_r[k + 1]);
    {   /* the row changers by destination row, so that each band reads only its own */
        uint32_t *mo = MOFF[(t + 1) & 1][tid];
        memset(mo, 0, (GD + 1) * 4);
        mover *mv = w->mov, *tmp = MOVTMP[tid];
        for (uint32_t i = 0; i < w->nmov; i++) mo[(mv[i].c >> 10) + 1]++;
        for (uint32_t r = 1; r <= GD; r++) mo[r] += mo[r - 1];
        uint32_t pos[GD];
        memcpy(pos, mo, GD * 4);
        for (uint32_t i = 0; i < w->nmov; i++) tmp[pos[mv[i].c >> 10]++] = mv[i];
        memcpy(mv, tmp, (size_t)w->nmov * sizeof(mover));
    }
    float m8[8]; _mm256_storeu_ps(m8, w->vmd);
    float m = 0.0f;
    for (int q = 0; q < 8; q++) if (m8[q] > m) m = m8[q];
    g_maxdisp[tid * 16] = m;
}

/* ---------------------------------------------------------------- serial combat */
static uint32_t g_dead;
static uint32_t g_t_serial;

/* A dead entity gets NaN positions: the next pass drops it. The counts it
   contributed to the next layout are taken back. */
static inline void mark_dead(uint32_t s) {
    if (X1[s] != X1[s]) return;
    uint32_t t = g_t_serial;
    const uint32_t *cs = CS[t & 1];
    uint32_t a = 0, b = GD;     /* row of slot s in layout t */
    while (b - a > 1) { uint32_t m = (a + b) / 2; if (cs[m * GD] <= s) a = m; else b = m; }
    uint32_t c = cell_of(X1[s], Z1[s]);
    if (c / GD == a) { STAYC[(t + 1) & 1][c]--; ROWST[(t + 1) & 1][a]--; }
    else MROW[(t + 1) & 1][0][c / GD]--;
    X1[s] = NAN; Z1[s] = NAN; g_dead++;
}

static void box_cells(float x0, float x1, float z0, float z1, int *cx0, int *cx1, int *cz0, int *cz1) {
    int a = (int)floorf((x0 + 1024.0f) * 0.5f), b = (int)floorf((x1 + 1024.0f) * 0.5f);
    int c = (int)floorf((z0 + 1024.0f) * 0.5f), d = (int)floorf((z1 + 1024.0f) * 0.5f);
    *cx0 = a < 0 ? 0 : (a > (int)GD - 1 ? (int)GD - 1 : a);
    *cx1 = b < 0 ? 0 : (b > (int)GD - 1 ? (int)GD - 1 : b);
    *cz0 = c < 0 ? 0 : (c > (int)GD - 1 ? (int)GD - 1 : c);
    *cz1 = d < 0 ? 0 : (d > (int)GD - 1 ? (int)GD - 1 : d);
}

static void serial_post(uint32_t t, uint32_t keys, float aimx, float aimz) {
    const uint32_t *cs = CS[t & 1];
    uint32_t *seq = SEQ[t & 1];
    X1 = OUTX[t & 1]; Z1 = OUTZ[t & 1]; VX1 = OUTVX[t & 1]; VZ1 = OUTVZ[t & 1];
    g_t_serial = t;
    float *hps = HPS;
    uint32_t n = cs[GC];
    float md = 0.0f;
    for (int t = 0; t < g_threads; t++) if (g_maxdisp[t * 16] > md) md = g_maxdisp[t * 16];
    /* An entity's cell comes from its start-of-tick position; it has moved at
       most md per axis since. Area effects scan the cells of their reach
       grown by this margin (plus slack), which covers every entity they can touch. */
    const float margin = md + 2.0f;
    g_dead = 0;
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
            float x = prx[p], z = prz[p];
            if (prp[p] == 0 || !(x >= -1024.0f && x < 1024.0f && z >= -1024.0f && z < 1024.0f)) continue;
            uint32_t c = cell_of(x, z), cx = c % GD, cz = c / GD;
            uint32_t xl = cx > 0 ? cx - 1 : 0, xh = cx < GD - 1 ? cx + 2 : GD;
            for (uint32_t rz = cz > 0 ? cz - 1 : 0; rz <= cz + 1 && rz < GD; rz++) {
                for (uint32_t s = cs[rz * GD + xl], e = cs[rz * GD + xh]; s < e; s++) {
                    float *hp = &hps[seq[s] & ~MOVED];
                    if (*hp > 0.0f) {
                        float dx = x - X1[s], dz = z - Z1[s];
                        if (dx * dx + dz * dz < 0.36f) {
                            *hp = *hp - 1.0f;
                            if (*hp <= 0.0f) mark_dead(s);
                            if (--prp[p] == 0) goto pdone;
                        }
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
    /* abilities: decisions */
    int shell = 0; float s0 = 0, s1 = 0, sk = 0, r1v = 0;
    if (alive && (keys & 64) && pulse_cd == 0) { pulse_on = 1; pulse_r = 0; pulse_cx = hx; pulse_cz = hz; pulse_cd = 240; }
    if (pulse_on) {
        float r0 = pulse_r, r1 = r0 + 0.75f; sk = 26.0f * (1.0f - r0 / 36.0f);
        s0 = r0 * r0; s1 = r1 * r1; shell = 1; r1v = r1; pulse_r = r1; if (r1 >= 36.0f) pulse_on = 0;
    }
    if (pulse_cd > 0) pulse_cd--;
    int beam = 0; float ux = 0, uz = 0;
    if (alive && (keys & 128) && energy >= 0.75f) { energy = energy - 0.75f; beam_on = 1; beam = 1; aim_dir(aimx, aimz, &ux, &uz); }
    else { beam_on = 0; float e = energy + 0.25f; energy = e < 100.0f ? e : 100.0f; }
    int strike = 0;
    if (alive && (keys & 256) && bomb_cd == 0 && bomb_timer == 0) { bomb_timer = 50; bomb_x = aimx; bomb_z = aimz; bomb_cd = 480; }
    else if (bomb_timer > 0) { bomb_timer--; strike = bomb_timer == 0; }
    if (bomb_cd > 0) bomb_cd--;
    /* abilities: per-entity effects, in spec order pulse, beam, bomb per entity */
    int cx0, cx1, cz0, cz1;
    if (shell) {
        float R = r1v + margin;
        box_cells(pulse_cx - R, pulse_cx + R, pulse_cz - R, pulse_cz + R, &cx0, &cx1, &cz0, &cz1);
        for (int rz = cz0; rz <= cz1; rz++)
            for (uint32_t s = cs[rz * GD + cx0], e = cs[rz * GD + cx1 + 1]; s < e; s++) {
                float dx = X1[s] - pulse_cx, dz = Z1[s] - pulse_cz, d2 = dx * dx + dz * dz;
                if (d2 >= s0 && d2 < s1 && d2 > EPS) {
                    float inv = 1.0f / sqrtf(d2);
                    VX1[s] = VX1[s] + (dx * inv) * sk; VZ1[s] = VZ1[s] + (dz * inv) * sk;
                }
            }
    }
    if (beam) {
        float ex = hx + ux * 45.0f, ez = hz + uz * 45.0f, M = margin + 1.0f;
        box_cells(fminf(hx, ex) - M, fmaxf(hx, ex) + M, fminf(hz, ez) - M, fmaxf(hz, ez) + M, &cx0, &cx1, &cz0, &cz1);
        for (int rz = cz0; rz <= cz1; rz++)
            for (uint32_t s = cs[rz * GD + cx0], e = cs[rz * GD + cx1 + 1]; s < e; s++) {
                float dx = X1[s] - hx, dz = Z1[s] - hz, t = dx * ux + dz * uz, d2 = dx * dx + dz * dz, p2 = d2 - t * t;
                if (t >= 0.0f && t <= 45.0f && p2 < 0.64f) {
                    float *hp = &hps[seq[s] & ~MOVED];
                    *hp = *hp - 0.2f;
                    if (*hp <= 0.0f) mark_dead(s);
                }
            }
    }
    if (strike) {
        float R = 12.0f + margin;
        box_cells(bomb_x - R, bomb_x + R, bomb_z - R, bomb_z + R, &cx0, &cx1, &cz0, &cz1);
        for (int rz = cz0; rz <= cz1; rz++)
            for (uint32_t s = cs[rz * GD + cx0], e = cs[rz * GD + cx1 + 1]; s < e; s++) {
                float dx = X1[s] - bomb_x, dz = Z1[s] - bomb_z, d2 = dx * dx + dz * dz;
                if (d2 < 144.0f) {
                    float *hp = &hps[seq[s] & ~MOVED];
                    *hp = *hp - 4.0f;
                    if (*hp <= 0.0f) mark_dead(s);
                    if (d2 > EPS) {
                        float d = sqrtf(d2), inv = 1.0f / d, k = 40.0f * (1.0f - d / 12.0f);
                        VX1[s] = VX1[s] + (dx * inv) * k; VZ1[s] = VZ1[s] + (dz * inv) * k;
                    }
                }
            }
    }
    /* contact */
    if (alive && dash_ticks == 0) {
        uint32_t c = cell_of(hx, hz), cx = c % GD, cz = c / GD;
        uint32_t xl = cx > 0 ? cx - 1 : 0, xh = cx < GD - 1 ? cx + 2 : GD;
        int cnt = 0;
        for (uint32_t rz = cz > 0 ? cz - 1 : 0; rz <= cz + 1 && rz < GD; rz++)
            for (uint32_t s = cs[rz * GD + xl], e = cs[rz * GD + xh]; s < e; s++)
                if (hps[seq[s] & ~MOVED] > 0.0f) {
                    float dx = X1[s] - hx, dz = Z1[s] - hz;
                    if (dx * dx + dz * dz < 1.21f) { if (++cnt == 20) goto cdone; }
                }
    cdone:
        hhp = hhp - ((float)cnt * 4.0f) * DT;
        if (hhp <= 0.0f) { hhp = 0.0f; alive = 0; }
    }
    /* deaths: removed by the next merge */
    kills += g_dead;
    g_count -= g_dead;
    /* spawning: appended to the current layout as row movers */
    g_nslots = n;
    mover *mv = SPAWNL[(t + 1) & 1];
    uint32_t *nmv = &SPAWNN[(t + 1) & 1];
    *nmv = 0;
    uint32_t ns = g_count < g_target ? (g_target - g_count < 64 ? g_target - g_count : 64) : 0;
    for (uint32_t q = 0; q < ns; q++) {
        float a, b, l2;
        for (;;) { a = unitf(pcg_next(&rng)) * 2.0f - 1.0f; b = unitf(pcg_next(&rng)) * 2.0f - 1.0f; l2 = a * a + b * b; if (!(l2 > 1.0f || l2 < 0.0625f)) break; }
        float inv = 1.0f / sqrtf(l2), r = 70.0f + unitf(pcg_next(&rng)) * 50.0f;
        float x = clampf(hx + (a * inv) * r), z = clampf(hz + (b * inv) * r);
        uint32_t s = g_nslots++;
        X1[s] = x; Z1[s] = z; VX1[s] = 0.0f; VZ1[s] = 0.0f; hps[g_seq_next] = 1.0f; seq[s] = g_seq_next++;
        uint32_t c = cell_of(x, z);
        mv[*nmv].s = s; mv[*nmv].c = c; (*nmv)++;
        MROW[(t + 1) & 1][0][c / GD]++;
        g_count++; spawned++;
    }
    g_tick++;
}

/* ---------------------------------------------------------------- hero + input */
/* ---------------------------------------------------------------- hero + input */
static void bench_input(void) {
    static const uint32_t BK[4] = {1, 8, 4, 2};
    static const float AX[4] = {10, 0, -10, 0}, AZ[4] = {0, 10, 0, -10};
    uint32_t t = g_tick, k = BK[(t / 120) & 3] | 16;
    if (t % 180 == 0) k |= 32;
    if (t % 300 == 60) k |= 64;
    if ((t / 90) % 2 == 1) k |= 128;
    if (t % 600 == 120) k |= 256;
    g_keys = k;
    g_aimx = hx + AX[(t / 30) & 3];
    g_aimz = hz + AZ[(t / 30) & 3];
}

static void hero_move(uint32_t keys, float aimx, float aimz) {
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
}

/* ---------------------------------------------------------------- driver */
/* ---------------------------------------------------------------- driver */
static uint32_t g_T;
static LARGE_INTEGER g_t0, g_t1;

static double g_prof[8];
static int g_profon;
static inline void prof(int tid, int k, LARGE_INTEGER *last) {
    if (tid || !g_profon) return;
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    g_prof[k] += (double)(now.QuadPart - last->QuadPart);
    *last = now;
}

/* Per tick: grid rebuild + separation + integration (parallel, by bands of rows) |
   combat, abilities, deaths, spawns, next input (thread 0). */
static DWORD WINAPI sim_thread(LPVOID arg) {
    int tid = (int)(intptr_t)arg;
    LARGE_INTEGER pl; QueryPerformanceCounter(&pl);
    barrier();
    if (tid == 0) {
        QueryPerformanceCounter(&g_t0);
        if (g_T) { bench_input(); hero_move(g_keys, g_aimx, g_aimz); }
        make_bands(0);
        queues_init();
    }
    barrier();
    prof(tid, 7, &pl);
    for (uint32_t t = 0; t < g_T; t++) {
        fused_phase(tid, t);
        barrier();
        prof(tid, 2, &pl);
        if (tid == 0) {
            serial_post(t, g_keys, g_aimx, g_aimz);
            if (t + 1 < g_T) {
                bench_input(); hero_move(g_keys, g_aimx, g_aimz);
                make_bands(t + 1);
                queues_init();
            }
        }
        prof(tid, 3, &pl);
        barrier();
        prof(tid, 4, &pl);
    }
    if (tid == 0) QueryPerformanceCounter(&g_t1);
    return 0;
}

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ULL; } return h;
}

int main(int argc, char **argv) {
    uint32_t N = 1000, T = 120, seed = 1; float half = 1000.0f;
    SYSTEM_INFO si; GetSystemInfo(&si);
    int threads = (int)si.dwNumberOfProcessors;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--entities") && i + 1 < argc) N = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ticks") && i + 1 < argc) T = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--spawn-half") && i + 1 < argc) half = (float)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    }
    if (N > MAXE) N = MAXE;
    if (threads < 1) threads = 1;
    if (threads > MAXT) threads = MAXT;
    g_threads = threads;
    g_profon = getenv("RIVAL_PROF") != NULL;
    g_T = T;
    /* bands: one per ~4000 entities, at least 4 per thread so that the
       stealing has something to balance with, at most 32 per thread */
    {
        uint32_t nb = N / 4000, lo = 4u * (uint32_t)threads, hi = 32u * (uint32_t)threads;
        g_nbands = threads == 1 ? 1 : nb < lo ? lo : nb > hi ? hi : nb;
    }
    if (getenv("RIVAL_BANDS")) g_nbands = (uint32_t)atoi(getenv("RIVAL_BANDS"));
    if (g_nbands < 1) g_nbands = 1;
    if (g_nbands > MAXB) g_nbands = MAXB;

    size_t fb = (size_t)CAP * 4;
    for (int l = 0; l < 2; l++) {
        OUTX[l] = zalloc(fb); OUTZ[l] = zalloc(fb); OUTVX[l] = zalloc(fb); OUTVZ[l] = zalloc(fb);
        SEQ[l] = zalloc(fb);
        CS[l] = zalloc(((size_t)GC + 16) * 4); STAYC[l] = zalloc(((size_t)GC + 16) * 4);
    }
    HPS = zalloc(((size_t)N + 64ull * T + 64) * 4);
    for (int t = 0; t < threads; t++) {
        MOV[0][t] = zalloc((size_t)CAP * sizeof(mover));
        MOV[1][t] = zalloc((size_t)CAP * sizeof(mover));
        MOVTMP[t] = zalloc((size_t)CAP * sizeof(mover));
        MOFF[0][t] = zalloc((GD + 1) * 4); MOFF[1][t] = zalloc((GD + 1) * 4);
        MROW[0][t] = zalloc(GD * 4); MROW[1][t] = zalloc(GD * 4);
        g_work[t] = xmalloc(sizeof(work));
        memset(g_work[t], 0, sizeof(work));
    }
    /* init: the spawn, sorted by cell (stable, so by sequence within a cell),
       is the output of a tick -1 in which nobody moved */
    pcg_seed(&rng, seed);
    {
        float *ix = malloc((size_t)N * 4 + 4), *iz = malloc((size_t)N * 4 + 4);
        uint32_t *ic = malloc((size_t)N * 4 + 4);
        uint32_t *cs = CS[1], *cnt = STAYC[0];
        for (uint32_t i = 0; i < N; i++) {
            float a = unitf(pcg_next(&rng)), b = unitf(pcg_next(&rng));
            ix[i] = (a * 2.0f - 1.0f) * half; iz[i] = (b * 2.0f - 1.0f) * half;
            ic[i] = cell_of(ix[i], iz[i]);
            cnt[ic[i]]++;
            ROWST[0][ic[i] / GD]++;
            HPS[i] = 1.0f;
        }
        uint32_t run = 0;
        for (uint32_t c = 0; c < GC; c++) { cs[c] = run; run += cnt[c]; }
        cs[GC] = run;
        uint32_t *cur = malloc((size_t)GC * 4);
        memcpy(cur, cs, (size_t)GC * 4);
        for (uint32_t i = 0; i < N; i++) {
            uint32_t s = cur[ic[i]]++;
            OUTX[1][s] = ix[i]; OUTZ[1][s] = iz[i]; OUTVX[1][s] = 0.0f; OUTVZ[1][s] = 0.0f; SEQ[1][s] = i;
        }
        free(cur); free(ix); free(iz); free(ic);
    }
    g_nslots = N; g_count = g_target = N; g_seq_next = N;

    HANDLE th[MAXT];
    for (int t = 1; t < threads; t++) { th[t] = CreateThread(NULL, 0, sim_thread, (LPVOID)(intptr_t)t, 0, NULL); SetThreadPriority(th[t], THREAD_PRIORITY_ABOVE_NORMAL); }
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    sim_thread((LPVOID)0);
    for (int t = 1; t < threads; t++) { WaitForSingleObject(th[t], INFINITE); CloseHandle(th[t]); }

    /* hash: live entities in sequence order */
    int fin = T ? (int)((T - 1) & 1) : 1;
    const uint32_t *seq = SEQ[fin];
    X1 = OUTX[fin]; Z1 = OUTZ[fin]; VX1 = OUTVX[fin]; VZ1 = OUTVZ[fin];
    uint32_t nseq = g_seq_next;
    uint32_t *slot_of = malloc((size_t)nseq * 4 + 4);
    memset(slot_of, 0xff, (size_t)nseq * 4);
    for (uint32_t s = 0; s < g_nslots; s++) if (X1[s] == X1[s]) slot_of[seq[s] & ~MOVED] = s;
    uint32_t cnt = 0;
    float *o[5]; for (int k = 0; k < 5; k++) o[k] = malloc((size_t)g_count * 4 + 4);
    for (uint32_t q = 0; q < nseq; q++) {
        uint32_t s = slot_of[q];
        if (s == 0xffffffffu) continue;
        if (cnt >= g_count) { cnt++; continue; }
        o[0][cnt] = X1[s]; o[1][cnt] = Z1[s]; o[2][cnt] = VX1[s]; o[3][cnt] = VZ1[s]; o[4][cnt] = HPS[q];
        cnt++;
    }
    if (cnt != g_count) { fprintf(stderr, "internal: count mismatch %u vs %u%c", cnt, g_count, 10); cnt = g_count; }
    uint64_t h = 0xcbf29ce484222325ULL;
    h = fnv(h, &cnt, 4); h = fnv(h, &g_tick, 4); h = fnv(h, &hx, 4); h = fnv(h, &hz, 4);
    for (int k = 0; k < 5; k++) h = fnv(h, o[k], (size_t)cnt * 4);
    h = fnv(h, &hhp, 4); h = fnv(h, &alive, 4); h = fnv(h, &kills, 4); h = fnv(h, &npr, 4);
    h = fnv(h, prx, npr * 4); h = fnv(h, prz, npr * 4);
    h = fnv(h, &energy, 4); h = fnv(h, &pulse_r, 4); h = fnv(h, &dash_ticks, 4); h = fnv(h, &pulse_on, 4); h = fnv(h, &bomb_timer, 4);
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    double tf = (double)f.QuadPart * (T ? T : 1) / 1e6;
    if (g_profon)
        printf("prof us/tick: fused %.1f serial %.1f barrier %.1f%c", g_prof[2] / tf, g_prof[3] / tf, g_prof[4] / tf, 10);
    uint64_t us = (uint64_t)((g_t1.QuadPart - g_t0.QuadPart) * 1000000 / f.QuadPart);
    printf("us_per_tick=%llu%c", (unsigned long long)(T ? us / T : 0), 10);
    printf("hash=%016llx%c", (unsigned long long)h, 10);
    printf("count=%u kills=%u hero_hp=%g alive=%u proj=%u threads=%d%c", cnt, kills, (double)hhp, alive, npr, threads, 10);
    return 0;
}
