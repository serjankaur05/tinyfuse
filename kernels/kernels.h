/*
 * tinyfuse kernels: dense (Linear) layer kernels for CPU.
 *
 * Shapes follow PyTorch's nn.Linear:
 *   x    [B][K]   input activations, row-major
 *   W    [N][K]   weights (PyTorch layout, out_features x in_features)
 *   bias [N]
 *   y    [B][N]   output
 *
 * The optimized kernels take weights packed by pack_weights() into
 * panel-major order, so the innermost loop reads memory sequentially and maps
 * onto SIMD lanes.
 */
#ifndef TINYFUSE_KERNELS_H
#define TINYFUSE_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ACT_NONE = 0,
    ACT_RELU = 1,
    ACT_CELU = 2,    /* alpha = 1: x > 0 ? x : exp(x) - 1 */
    ACT_SIGMOID = 3
} act_t;

/* Reference: one dot product per output, scalar, no blocking. */
void linear_naive(const float *x, const float *W, const float *bias, float *y,
                  int B, int K, int N);

/* Separate (unfused) elementwise passes over y. */
void bias_add(float *y, const float *bias, int B, int N);
void activation(float *y, int B, int N, act_t act);

/* Repack W [N][K] into panel-major order (K * N floats). Each panel holds
 * panel_width() output columns as [K][panel_width]; leftover columns follow
 * as [K][N % panel_width]. */
void pack_weights(const float *W, float *Wp, int K, int N);
int panel_width(void);

/* Register-tiled SIMD kernel (AVX-512 / AVX2+FMA / scalar fallback).
 * If fuse != 0, bias and activation are applied in the epilogue while the
 * tile is still in registers; otherwise only the matmul is computed and the
 * caller runs bias_add() and activation() as separate passes. */
void linear_simd(const float *x, const float *Wt, const float *bias, float *y,
                 int B, int K, int N, act_t act, int fuse);

/* Same as linear_simd(fuse=1) but splits the batch across OpenMP threads. */
void linear_simd_omp(const float *x, const float *Wt, const float *bias, float *y,
                     int B, int K, int N, act_t act);

/* Runs a register-only FMA loop; returns the FLOPs executed (time it to get
 * the achievable peak GFLOP/s). */
double fma_probe(long iters);

/* Name of the SIMD instruction set the library was compiled for. */
const char *simd_isa(void);

#ifdef __cplusplus
}
#endif

#endif
