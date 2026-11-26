/*
 * bench: runs every layer of the EEG P300 network through each kernel variant,
 * checks the result against the naive reference, and writes one Chrome/Perfetto
 * trace event per kernel launch so kprof (the Go profiler) can analyze it.
 *
 * usage: bench [batch] [repeats] [trace.json]
 */
#define _POSIX_C_SOURCE 199309L
#include "../kernels/kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    int K, N;
    act_t act;
    const char *act_name;
} layer_t;

/* nn.Sequential from Machine_Learning_to_Analyze_EEG_Data.ipynb */
static const layer_t LAYERS[] = {
    {226, 500, ACT_CELU, "celu"},
    {500, 1000, ACT_RELU, "relu"},
    {1000, 100, ACT_RELU, "relu"},
    {100, 10, ACT_RELU, "relu"},
    {10, 1, ACT_SIGMOID, "sigmoid"},
};
#define NLAYERS ((int)(sizeof(LAYERS) / sizeof(LAYERS[0])))

enum { V_NAIVE, V_SIMD_UNFUSED, V_SIMD_FUSED, V_SIMD_OMP, NVARIANTS };
static const char *VARIANT_NAMES[] = {"naive", "simd_unfused", "simd_fused",
                                      "simd_fused_omp"};

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static float *alloc_rand(size_t n, float scale, unsigned *seed) {
    float *p = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        p[i] = ((float)rand_r(seed) / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    return p;
}

static void run_variant(int v, const float *x, const float *W, const float *Wt,
                        const float *bias, float *y, int B, const layer_t *L) {
    switch (v) {
    case V_NAIVE:
        linear_naive(x, W, bias, y, B, L->K, L->N);
        activation(y, B, L->N, L->act);
        break;
    case V_SIMD_UNFUSED:
        linear_simd(x, Wt, bias, y, B, L->K, L->N, L->act, 0);
        bias_add(y, bias, B, L->N);
        activation(y, B, L->N, L->act);
        break;
    case V_SIMD_FUSED:
        linear_simd(x, Wt, bias, y, B, L->K, L->N, L->act, 1);
        break;
    case V_SIMD_OMP:
        linear_simd_omp(x, Wt, bias, y, B, L->K, L->N, L->act);
        break;
    }
}

int main(int argc, char **argv) {
    int B = argc > 1 ? atoi(argv[1]) : 256;
    int repeats = argc > 2 ? atoi(argv[2]) : 50;
    const char *trace_path = argc > 3 ? argv[3] : "trace.json";
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    /* Achievable FMA peak, single thread and all threads. */
    /* Best of 3 runs, since a shared VM's clock speed varies. */
    double peak_1t = 0.0, peak_mt = 0.0;
    for (int rep = 0; rep < 3; rep++) {
        double pk = now_us();
        double pflops = fma_probe(100000000L);
        double g = pflops / ((now_us() - pk) * 1e3);
        if (g > peak_1t)
            peak_1t = g;
#ifdef _OPENMP
        pk = now_us();
        double total = 0.0;
#pragma omp parallel reduction(+ : total)
        total += fma_probe(100000000L);
        g = total / ((now_us() - pk) * 1e3);
#endif
        if (g > peak_mt)
            peak_mt = g;
    }

    /* Memory bandwidth: STREAM-style triad a = b + s*c over 3 x 32 MB arrays,
     * far larger than cache. Best of 5. */
    double bw = 0.0;
    {
        size_t n = 8u << 20;
        float *A = malloc(n * sizeof(float)), *Bv = malloc(n * sizeof(float)),
              *C = malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) { A[i] = 0.0f; Bv[i] = 1.0f; C[i] = 2.0f; }
        for (int rep = 0; rep < 5; rep++) {
            double s = now_us();
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; i++)
                A[i] = Bv[i] + 0.5f * C[i];
            double g = 3.0 * n * sizeof(float) / ((now_us() - s) * 1e3);
            if (g > bw)
                bw = g;
        }
        volatile float sink = A[n / 2];
        (void)sink;
        free(A); free(Bv); free(C);
    }

    FILE *tf = fopen(trace_path, "w");
    if (!tf) {
        perror(trace_path);
        return 1;
    }
    fprintf(tf, "{\"metadata\":{\"isa\":\"%s\",\"threads\":%d,\"batch\":%d,"
                "\"peak_gflops_1t\":%.2f,\"peak_gflops_mt\":%.2f,\"bw_gbps\":%.2f},"
                "\"traceEvents\":[\n",
            simd_isa(), threads, B, peak_1t, peak_mt, bw);

    unsigned seed = 42;
    int first_event = 1, failures = 0;
    double t0 = now_us();

    printf("tinyfuse bench | isa=%s threads=%d batch=%d repeats=%d\n", simd_isa(),
           threads, B, repeats);
    printf("measured FMA peak: %.1f GFLOP/s (1 thread), %.1f GFLOP/s (%d threads); "
           "memory bandwidth %.1f GB/s\n", peak_1t, peak_mt, threads, bw);
    printf("%-6s %-12s %-16s %10s %10s %11s\n", "layer", "shape", "variant",
           "median_us", "GFLOP/s", "max_err");

    for (int l = 0; l < NLAYERS; l++) {
        const layer_t *L = &LAYERS[l];
        float *x = alloc_rand((size_t)B * L->K, 1.0f, &seed);
        float *W = alloc_rand((size_t)L->N * L->K, 1.0f / sqrtf((float)L->K), &seed);
        float *bias = alloc_rand(L->N, 0.1f, &seed);
        float *Wt = malloc((size_t)L->K * L->N * sizeof(float));
        float *ref = malloc((size_t)B * L->N * sizeof(float));
        float *y = malloc((size_t)B * L->N * sizeof(float));
        pack_weights(W, Wt, L->K, L->N);
        run_variant(V_NAIVE, x, W, Wt, bias, ref, B, L);

        double flops = 2.0 * B * L->K * L->N;
        double bytes = 4.0 * ((double)B * L->K + (double)L->K * L->N +
                              (double)B * L->N + L->N);

        for (int v = 0; v < NVARIANTS; v++) {
            double *times = malloc(repeats * sizeof(double));
            for (int w = 0; w < 3; w++) /* warm-up */
                run_variant(v, x, W, Wt, bias, y, B, L);
            for (int r = 0; r < repeats; r++) {
                double s = now_us();
                run_variant(v, x, W, Wt, bias, y, B, L);
                double e = now_us();
                times[r] = e - s;
                fprintf(tf,
                        "%s{\"name\":\"L%d linear %dx%d+%s\",\"cat\":\"%s\","
                        "\"ph\":\"X\",\"pid\":1,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,"
                        "\"args\":{\"layer\":%d,\"variant\":\"%s\",\"B\":%d,\"K\":%d,"
                        "\"N\":%d,\"flops\":%.0f,\"bytes\":%.0f}}\n",
                        first_event ? "" : ",", l + 1, L->K, L->N, L->act_name,
                        VARIANT_NAMES[v], v + 1, s - t0, e - s, l + 1,
                        VARIANT_NAMES[v], B, L->K, L->N, flops, bytes);
                first_event = 0;
            }

            float max_err = 0.0f;
            for (size_t i = 0; i < (size_t)B * L->N; i++) {
                float d = fabsf(y[i] - ref[i]);
                if (d > max_err)
                    max_err = d;
            }
            if (max_err > 1e-3f) {
                failures++;
                fprintf(stderr, "MISMATCH layer %d variant %s err=%g\n", l + 1,
                        VARIANT_NAMES[v], max_err);
            }

            /* median for the console summary (insertion sort; repeats is small) */
            for (int i = 1; i < repeats; i++) {
                double key = times[i];
                int j = i - 1;
                while (j >= 0 && times[j] > key) {
                    times[j + 1] = times[j];
                    j--;
                }
                times[j + 1] = key;
            }
            double med = times[repeats / 2];
            char shape[32];
            snprintf(shape, sizeof shape, "%dx%d", L->K, L->N);
            printf("L%-5d %-12s %-16s %10.1f %10.2f %11.2e\n", l + 1, shape,
                   VARIANT_NAMES[v], med, flops / (med * 1e3), max_err);
            free(times);
        }
        free(x); free(W); free(bias); free(Wt); free(ref); free(y);
    }

    fprintf(tf, "]}\n");
    fclose(tf);
    if (failures) {
        fprintf(stderr, "%d correctness failure(s)\n", failures);
        return 1;
    }
    printf("all variants match the naive reference (max_err <= 1e-3)\n");
    return 0;
}
