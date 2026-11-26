/*
 * tinyfuse CPU kernels for dense layers.
 *
 * linear_simd() computes a 4 x (2 * VW) output tile at a time, where VW is the
 * SIMD width (16 floats with AVX-512, 8 with AVX2). The tile lives in 8 vector
 * registers for the whole K loop, so each weight load is reused 4 times and
 * each broadcast input value is reused across 2 vectors. Weights are packed as
 * Wt [K][N] so a column panel of width 2*VW is read contiguously, and the outer
 * loop walks panels so one panel (K * 2 * VW floats) stays hot in L2 while every
 * row of the batch passes over it.
 */
#include "kernels.h"

#include <math.h>
#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#define VW 16
typedef __m512 vec;
#define VLOAD(p) _mm512_loadu_ps(p)
#define VSTORE(p, v) _mm512_storeu_ps((p), (v))
#define VSET1(s) _mm512_set1_ps(s)
#define VZERO() _mm512_setzero_ps()
#define VFMA(a, b, c) _mm512_fmadd_ps((a), (b), (c))
#define VADD(a, b) _mm512_add_ps((a), (b))
#define VMAX(a, b) _mm512_max_ps((a), (b))
#define VMIN(a, b) _mm512_min_ps((a), (b))
#define VMUL(a, b) _mm512_mul_ps((a), (b))
#define VSUB(a, b) _mm512_sub_ps((a), (b))
#define VDIV(a, b) _mm512_div_ps((a), (b))
#define VROUND(a) _mm512_roundscale_ps((a), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#define VSCALE2N(y, n) _mm512_scalef_ps((y), (n)) /* y * 2^n */
#define ISA_NAME "avx512"
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define VW 8
typedef __m256 vec;
#define VLOAD(p) _mm256_loadu_ps(p)
#define VSTORE(p, v) _mm256_storeu_ps((p), (v))
#define VSET1(s) _mm256_set1_ps(s)
#define VZERO() _mm256_setzero_ps()
#define VFMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#define VADD(a, b) _mm256_add_ps((a), (b))
#define VMAX(a, b) _mm256_max_ps((a), (b))
#define VMIN(a, b) _mm256_min_ps((a), (b))
#define VMUL(a, b) _mm256_mul_ps((a), (b))
#define VSUB(a, b) _mm256_sub_ps((a), (b))
#define VDIV(a, b) _mm256_div_ps((a), (b))
#define VROUND(a) _mm256_round_ps((a), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
/* y * 2^n: build 2^n by writing n + 127 into the float exponent bits. */
#define VSCALE2N(y, n)                                                          \
    _mm256_mul_ps((y), _mm256_castsi256_ps(_mm256_slli_epi32(                   \
                           _mm256_add_epi32(_mm256_cvtps_epi32(n),              \
                                            _mm256_set1_epi32(127)),            \
                           23)))
#define ISA_NAME "avx2+fma"
#else
#define ISA_NAME "scalar"
#endif

const char *simd_isa(void) { return ISA_NAME; }

static inline float act_scalar(float v, act_t act) {
    switch (act) {
    case ACT_RELU:
        return v > 0.0f ? v : 0.0f;
    case ACT_CELU:
        return v > 0.0f ? v : expf(v) - 1.0f;
    case ACT_SIGMOID:
        return 1.0f / (1.0f + expf(-v));
    default:
        return v;
    }
}

#ifdef VW
/* Vectorized exp(x) for float32, Cephes-style:
 *   n = round(x / ln2),  r = x - n * ln2  (|r| <= ln2 / 2),
 *   exp(r) ~= 1 + r + r^2 * P(r)  with a degree-5 minimax polynomial,
 *   exp(x) = exp(r) * 2^n.
 * Max relative error vs expf() is about 2e-7 (checked in bench). */
static inline vec vexp(vec x) {
    x = VMIN(x, VSET1(88.3762626647949f));
    x = VMAX(x, VSET1(-87.3365447505531f));
    vec n = VROUND(VMUL(x, VSET1(1.44269504088896341f)));
    vec r = VSUB(x, VMUL(n, VSET1(0.693359375f)));      /* ln2 split in two parts */
    r = VSUB(r, VMUL(n, VSET1(-2.12194440e-4f)));       /* for extra precision    */
    vec p = VSET1(1.9875691500e-4f);
    p = VFMA(p, r, VSET1(1.3981999507e-3f));
    p = VFMA(p, r, VSET1(8.3334519073e-3f));
    p = VFMA(p, r, VSET1(4.1665795894e-2f));
    p = VFMA(p, r, VSET1(1.6666665459e-1f));
    p = VFMA(p, r, VSET1(5.0000001201e-1f));
    vec y = VFMA(p, VMUL(r, r), VADD(r, VSET1(1.0f)));
    return VSCALE2N(y, n);
}

/* Activation entirely in registers. CELU(x) = max(x,0) + (exp(min(x,0)) - 1). */
static inline vec vact(vec v, act_t act) {
    switch (act) {
    case ACT_RELU:
        return VMAX(v, VZERO());
    case ACT_CELU:
        return VADD(VMAX(v, VZERO()),
                    VSUB(vexp(VMIN(v, VZERO())), VSET1(1.0f)));
    case ACT_SIGMOID:
        return VDIV(VSET1(1.0f),
                    VADD(VSET1(1.0f), vexp(VSUB(VZERO(), v))));
    default:
        return v;
    }
}
#endif

/* ------------------------------------------------------------------ */
/* Reference and unfused helpers                                       */
/* ------------------------------------------------------------------ */

void linear_naive(const float *x, const float *W, const float *bias, float *y,
                  int B, int K, int N) {
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += x[(size_t)b * K + k] * W[(size_t)n * K + k];
            y[(size_t)b * N + n] = s + bias[n];
        }
    }
}

void bias_add(float *y, const float *bias, int B, int N) {
    for (int b = 0; b < B; b++)
        for (int n = 0; n < N; n++)
            y[(size_t)b * N + n] += bias[n];
}

void activation(float *y, int B, int N, act_t act) {
    if (act == ACT_NONE)
        return;
    size_t total = (size_t)B * N, i = 0;
#ifdef VW
    for (; i + VW <= total; i += VW)
        VSTORE(y + i, vact(VLOAD(y + i), act));
#endif
    for (; i < total; i++)
        y[i] = act_scalar(y[i], act);
}

/* Panel width: the number of output columns one micro-kernel tile covers. */
#ifdef VW
#define NB (2 * VW)
#else
#define NB 1
#endif

int panel_width(void) {
#ifdef VW
    return NB;
#else
    return 0;
#endif
}

/* Packed layout ("panel-major"):
 *   for each full panel p of NB columns:  [K][NB] floats, contiguous
 *   then the leftover columns (N % NB):   [K][N % NB] floats, contiguous
 * Inside the K loop the micro-kernel reads NB consecutive floats per step, so
 * the whole panel streams through memory sequentially instead of jumping
 * N * 4 bytes (a new 4 KB page for N = 1000) on every k. */
void pack_weights(const float *W, float *Wp, int K, int N) {
#ifdef VW
    const int n_main = N - N % NB;
#else
    const int n_main = 0;
#endif
    for (int n0 = 0; n0 < n_main; n0 += NB) {
        float *panel = Wp + (size_t)n0 * K;
        for (int k = 0; k < K; k++)
            for (int j = 0; j < NB; j++)
                panel[(size_t)k * NB + j] = W[(size_t)(n0 + j) * K + k];
    }
    const int tw = N - n_main;
    float *tail = Wp + (size_t)n_main * K;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < tw; j++)
            tail[(size_t)k * tw + j] = W[(size_t)(n_main + j) * K + k];
}

/* ------------------------------------------------------------------ */
/* SIMD micro-kernels                                                  */
/* ------------------------------------------------------------------ */

/* Leftover columns [n_main, N) for one row, reading the contiguous tail block. */
#define TAIL_MAX 64
#define NARROW_TAIL 4
#define OMP_MIN_MACS 100000.0
static inline void row_tail(const float *xrow, const float *Wp, const float *bias,
                            float *yrow, int K, int N, int n_main, act_t act,
                            int fuse) {
    float acc[TAIL_MAX];
    const int tw = N - n_main;
    const float *tail = Wp + (size_t)n_main * K;
    if (tw <= NARROW_TAIL) {
        /* Narrow tail (e.g. the 10 -> 1 output layer): a vectorized loop over
         * just tw columns pays its setup cost on every k, which made this
         * 7x slower than the naive kernel (found with kprof). One dot product
         * per column, with 4 independent partial sums, avoids that. */
        for (int j = 0; j < tw; j++) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            const float *wc = tail + j;
            int k = 0;
            for (; k + 4 <= K; k += 4) {
                s0 += xrow[k + 0] * wc[(size_t)(k + 0) * tw];
                s1 += xrow[k + 1] * wc[(size_t)(k + 1) * tw];
                s2 += xrow[k + 2] * wc[(size_t)(k + 2) * tw];
                s3 += xrow[k + 3] * wc[(size_t)(k + 3) * tw];
            }
            for (; k < K; k++)
                s0 += xrow[k] * wc[(size_t)k * tw];
            float sum = (s0 + s1) + (s2 + s3);
            int n = n_main + j;
            yrow[n] = fuse ? act_scalar(sum + bias[n], act) : sum;
        }
        return;
    }
    for (int base = 0; base < tw; base += TAIL_MAX) {
        int w = tw - base < TAIL_MAX ? tw - base : TAIL_MAX;
        for (int j = 0; j < w; j++)
            acc[j] = 0.0f;
        for (int k = 0; k < K; k++) {
            float xv = xrow[k];
            const float *wr = tail + (size_t)k * tw + base;
            for (int j = 0; j < w; j++)
                acc[j] += xv * wr[j];
        }
        for (int j = 0; j < w; j++) {
            int n = n_main + base + j;
            yrow[n] = fuse ? act_scalar(acc[j] + bias[n], act) : acc[j];
        }
    }
}

#ifdef VW
/* Fused epilogue: bias add and activation run on the accumulator registers. */
static inline void store_vec(float *dst, vec acc, const float *bias, act_t act,
                             int fuse) {
    if (fuse)
        acc = vact(VADD(acc, VLOAD(bias)), act);
    VSTORE(dst, acc);
}

/* 4 rows x NB columns: 8 accumulators, each weight load reused 4 times. */
static inline void kernel_4x2(const float *x, const float *panel, const float *bias,
                              float *y, int K, int N, int b0, int n0, act_t act,
                              int fuse) {
    vec c00 = VZERO(), c01 = VZERO(), c10 = VZERO(), c11 = VZERO();
    vec c20 = VZERO(), c21 = VZERO(), c30 = VZERO(), c31 = VZERO();
    const float *x0 = x + (size_t)(b0 + 0) * K;
    const float *x1 = x + (size_t)(b0 + 1) * K;
    const float *x2 = x + (size_t)(b0 + 2) * K;
    const float *x3 = x + (size_t)(b0 + 3) * K;
    const float *w = panel;
    for (int k = 0; k < K; k++, w += NB) {
        vec w0 = VLOAD(w), w1 = VLOAD(w + VW);
        vec a;
        a = VSET1(x0[k]); c00 = VFMA(a, w0, c00); c01 = VFMA(a, w1, c01);
        a = VSET1(x1[k]); c10 = VFMA(a, w0, c10); c11 = VFMA(a, w1, c11);
        a = VSET1(x2[k]); c20 = VFMA(a, w0, c20); c21 = VFMA(a, w1, c21);
        a = VSET1(x3[k]); c30 = VFMA(a, w0, c30); c31 = VFMA(a, w1, c31);
    }
    float *y0 = y + (size_t)(b0 + 0) * N + n0;
    float *y1 = y + (size_t)(b0 + 1) * N + n0;
    float *y2 = y + (size_t)(b0 + 2) * N + n0;
    float *y3 = y + (size_t)(b0 + 3) * N + n0;
    const float *bb = bias + n0;
    store_vec(y0, c00, bb, act, fuse); store_vec(y0 + VW, c01, bb + VW, act, fuse);
    store_vec(y1, c10, bb, act, fuse); store_vec(y1 + VW, c11, bb + VW, act, fuse);
    store_vec(y2, c20, bb, act, fuse); store_vec(y2 + VW, c21, bb + VW, act, fuse);
    store_vec(y3, c30, bb, act, fuse); store_vec(y3 + VW, c31, bb + VW, act, fuse);
}

#if VW == 16
/* 8 rows x NB columns (AVX-512 only): 16 accumulators fit easily in the 32
 * zmm registers and halve the weight loads per FMA compared with 4 rows. */
static inline void kernel_8x2(const float *x, const float *panel, const float *bias,
                              float *y, int K, int N, int b0, int n0, act_t act,
                              int fuse) {
    vec c[8][2];
    for (int r = 0; r < 8; r++)
        c[r][0] = c[r][1] = VZERO();
    const float *xr[8];
    for (int r = 0; r < 8; r++)
        xr[r] = x + (size_t)(b0 + r) * K;
    const float *w = panel;
    for (int k = 0; k < K; k++, w += NB) {
        vec w0 = VLOAD(w), w1 = VLOAD(w + VW);
#pragma GCC unroll 8
        for (int r = 0; r < 8; r++) {
            vec a = VSET1(xr[r][k]);
            c[r][0] = VFMA(a, w0, c[r][0]);
            c[r][1] = VFMA(a, w1, c[r][1]);
        }
    }
    const float *bb = bias + n0;
    for (int r = 0; r < 8; r++) {
        float *yr = y + (size_t)(b0 + r) * N + n0;
        store_vec(yr, c[r][0], bb, act, fuse);
        store_vec(yr + VW, c[r][1], bb + VW, act, fuse);
    }
}
#endif

/* 1 row x NB columns, for leftover batch rows. */
static inline void kernel_1x2(const float *x, const float *panel, const float *bias,
                              float *y, int K, int N, int b, int n0, act_t act,
                              int fuse) {
    vec c0 = VZERO(), c1 = VZERO();
    const float *xr = x + (size_t)b * K;
    const float *w = panel;
    for (int k = 0; k < K; k++, w += NB) {
        vec a = VSET1(xr[k]);
        c0 = VFMA(a, VLOAD(w), c0);
        c1 = VFMA(a, VLOAD(w + VW), c1);
    }
    float *yr = y + (size_t)b * N + n0;
    store_vec(yr, c0, bias + n0, act, fuse);
    store_vec(yr + VW, c1, bias + n0 + VW, act, fuse);
}
#endif

/* Compute rows [b_begin, b_end) of the output. Panels are the outer loop so
 * one panel (K * NB floats) stays in L2 while every row of the batch uses it. */
static void linear_simd_rows(const float *x, const float *Wp, const float *bias,
                             float *y, int b_begin, int b_end, int K, int N,
                             act_t act, int fuse) {
#ifdef VW
    const int n_main = N - N % NB;
    for (int n0 = 0; n0 < n_main; n0 += NB) {
        const float *panel = Wp + (size_t)n0 * K;
        int b = b_begin;
#if VW == 16
        for (; b + 8 <= b_end; b += 8)
            kernel_8x2(x, panel, bias, y, K, N, b, n0, act, fuse);
#endif
        for (; b + 4 <= b_end; b += 4)
            kernel_4x2(x, panel, bias, y, K, N, b, n0, act, fuse);
        for (; b < b_end; b++)
            kernel_1x2(x, panel, bias, y, K, N, b, n0, act, fuse);
    }
#else
    const int n_main = 0;
#endif
    if (n_main < N)
        for (int b = b_begin; b < b_end; b++)
            row_tail(x + (size_t)b * K, Wp, bias, y + (size_t)b * N, K, N, n_main,
                     act, fuse);
}

void linear_simd(const float *x, const float *Wt, const float *bias, float *y,
                 int B, int K, int N, act_t act, int fuse) {
    linear_simd_rows(x, Wt, bias, y, 0, B, K, N, act, fuse);
}

void linear_simd_omp(const float *x, const float *Wt, const float *bias, float *y,
                     int B, int K, int N, act_t act) {
#ifdef _OPENMP
    /* Waking a thread team costs a few microseconds, more than a tiny layer's
     * entire runtime (the 10 -> 1 layer took 7.9 us threaded vs 2.6 us on one
     * thread), so small problems stay single-threaded. */
    if ((double)B * K * N < OMP_MIN_MACS) {
        linear_simd_rows(x, Wt, bias, y, 0, B, K, N, act, 1);
        return;
    }
#pragma omp parallel
    {
        int t = omp_get_thread_num(), nt = omp_get_num_threads();
        /* Split the batch into contiguous chunks that are multiples of 4 rows
         * so every thread can use the 4-row micro-kernel. */
        int blocks = (B + 3) / 4;
        int per = (blocks + nt - 1) / nt;
        int b_begin = t * per * 4;
        int b_end = b_begin + per * 4;
        if (b_end > B)
            b_end = B;
        if (b_begin < b_end)
            linear_simd_rows(x, Wt, bias, y, b_begin, b_end, K, N, act, 1);
    }
#else
    linear_simd_rows(x, Wt, bias, y, 0, B, K, N, act, 1);
#endif
}

/* ------------------------------------------------------------------ */
/* Peak throughput probe                                               */
/* ------------------------------------------------------------------ */

/* Runs `iters` rounds of 12 independent FMA chains (enough to hide FMA
 * latency) and returns the number of floating-point operations performed.
 * The caller times it to get the machine's achievable FMA peak, which kprof
 * uses as the roofline ceiling. */
double fma_probe(long iters) {
#ifdef VW
    /* Distinct starting values, otherwise the compiler proves the chains are
     * identical and merges them. */
    vec a0 = VSET1(1.00f), a1 = VSET1(1.01f), a2 = VSET1(1.02f), a3 = VSET1(1.03f);
    vec a4 = VSET1(1.04f), a5 = VSET1(1.05f), a6 = VSET1(1.06f), a7 = VSET1(1.07f);
    vec a8 = VSET1(1.08f), a9 = VSET1(1.09f), a10 = VSET1(1.10f), a11 = VSET1(1.11f);
    const vec m = VSET1(0.999999f), c = VSET1(1e-7f);
    for (long i = 0; i < iters; i++) {
        a0 = VFMA(a0, m, c); a1 = VFMA(a1, m, c); a2 = VFMA(a2, m, c);
        a3 = VFMA(a3, m, c); a4 = VFMA(a4, m, c); a5 = VFMA(a5, m, c);
        a6 = VFMA(a6, m, c); a7 = VFMA(a7, m, c); a8 = VFMA(a8, m, c);
        a9 = VFMA(a9, m, c); a10 = VFMA(a10, m, c); a11 = VFMA(a11, m, c);
    }
    vec s = VADD(VADD(VADD(a0, a1), VADD(a2, a3)), VADD(VADD(a4, a5), VADD(a6, a7)));
    s = VADD(s, VADD(VADD(a8, a9), VADD(a10, a11)));
    float out[VW];
    VSTORE(out, s);
    volatile float sink = out[0];
    (void)sink;
    return (double)iters * 12.0 * VW * 2.0;
#else
    float a[12];
    for (int j = 0; j < 12; j++) a[j] = 1.0f + 0.01f * j;
    for (long i = 0; i < iters; i++)
        for (int j = 0; j < 12; j++) a[j] = a[j] * 0.999999f + 1e-7f;
    volatile float sink = a[0];
    (void)sink;
    return (double)iters * 12.0 * 2.0;
#endif
}
