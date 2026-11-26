# tinyfuse

A small ML performance stack, built end to end around one real model: the
P300 "surprise" detector from my
[EEG project](https://github.com/serjankaur05/ML-for-EEG-Data)
(`nn.Sequential`: 226 → 500 → 1000 → 100 → 10 → 1 with CELU / ReLU / sigmoid).

| Part | Language | What it does |
|---|---|---|
| `kernels/` | C | Dense-layer kernels: naive reference, register-tiled **AVX-512 / AVX2+FMA** micro-kernels, panel-packed weights, vectorized `exp`, fused bias + activation epilogues, **OpenMP** threading |
| `compiler/` | Python → **LLVM IR** | `tinyfuse`, an ahead-of-time compiler: graph IR, decomposition, **epilogue fusion**, liveness-based **memory planning**, LLVM IR codegen with row tiling, built with `clang -O3 -fveclib=libmvec` |
| `profiler/` | **Go** | `kprof`, a trace profiler: per-kernel p50/p95 latency, GFLOP/s, GB/s, arithmetic intensity, % of measured peak, roofline (compute vs memory bound), layer-by-layer comparison, HTML report |
| `bench/` | C, Python | Measures FMA peak and memory bandwidth, benchmarks every backend, verifies every output, writes Chrome/Perfetto traces |

Every result is checked against a reference: kernels against the naive C
version, and every compiled model and backend against a float64 NumPy forward
pass (max abs error ≤ 1e-7).

## Results

Measured on a 2-vCPU Intel Xeon cloud VM (AVX-512), batch 256, 100 repeats,
median. Shared VMs are noisy, so expect ±10–20% run to run.
`make profile` reproduces everything.

**Machine ceilings (measured by `bench`):** FMA peak 240 GFLOP/s on 1 thread,
461 GFLOP/s on 2 threads; memory bandwidth 35.6 GB/s.

### Whole network, one forward pass (365.6 MFLOP)

| Backend | Latency | GFLOP/s | vs naive |
|---|---:|---:|---:|
| C naive (scalar dot products) | 78.5 ms | 4.7 | 1.0× |
| tinyfuse compiler, fused, untiled | 12.2 ms | 29.9 | 6.4× |
| tinyfuse compiler, unfused, 4-row tiles | 10.7 ms | 34.3 | 7.4× |
| tinyfuse compiler, fused, 4-row tiles | 10.2 ms | 35.9 | 7.7× |
| NumPy (OpenBLAS) | 2.81 ms | 130.3 | 28.0× |
| C SIMD kernels, 1 thread | 3.14 ms | 116.3 | 25.0× |
| **C SIMD kernels, 2 threads** | **1.87 ms** | **195.8** | **42.1×** |

The hand-written kernels beat NumPy's BLAS path by about 1.5×. That gap
comes mostly from fusing bias and activation into the matmul and avoiding
per-op Python overhead, not from a faster GEMM than OpenBLAS in general.

### How the main layer (500 × 1000, ReLU) got fast, single thread

| Step | GFLOP/s | What changed |
|---|---:|---|
| Naive dot products | ~4 | Scalar; the compiler cannot vectorize a float reduction without fast-math |
| 4×2 SIMD register tile | 49 | 8 vector accumulators, each weight load reused across 4 rows |
| **Panel-packed weights** | 96 | Weights repacked as `[N/32][K][32]`, so the K loop reads sequentially instead of jumping 4 KB (a new page) every step |
| **8×2 tile on AVX-512** | 143 | 16 accumulators (of 32 zmm registers) halve weight loads per FMA: **59% of measured FMA peak** |

### Fixes found by measuring

- **CELU was slower than the matmul.** Layer 1 spent more time in 128k scalar
  `expf()` calls than in its FLOPs, and calling `expf()` mid-tile also spilled
  every accumulator register. A vectorized `exp` (range reduction +
  degree-5 polynomial + `scalef`) cut layer 1 from **1,124 µs to 440 µs**.
- **kprof flagged the 10 → 1 output layer at 7× slower than naive** (12.0 µs
  vs 1.1 µs). A 1-column tail made the vectorized loop pay its setup cost on
  every k. A dedicated narrow-tail dot product brought it to **2.6 µs**.
- **Threading tiny layers hurt:** a 2.6 µs layer took 7.9 µs with OpenMP.
  Layers under 100k MACs now stay single-threaded.
- **The first peak measurement was wrong.** The FMA probe reported 365 GFLOP/s
  because the compiler proved its 12 identical FMA chains equal and merged
  them (the assembly had 3 FMAs, not 12). Distinct seeds fixed it.

### Compiler passes on the EEG model

```
frontend:  10 nodes  linear[226x500] -> act[celu] -> ... -> linear[10x1] -> act[sigmoid]
decompose: 15 nodes
fuse:      15 -> 5 nodes  dense[226x500] act=celu -> dense[500x1000] act=relu -> ...
plan:      arena 1500 floats/row (slot0 500 + slot1 1000)
```

- **Fusion** turns MatMul → BiasAdd → Act into one Dense node whose epilogue
  runs on rows still in L1: 15 nodes become 5, and 10 full passes over
  intermediate tensors disappear.
- **Memory planning** makes elementwise ops in place and ping-pongs between
  two arena slots sized per slot. The unfused graph needs **1,500 instead of
  4,830 floats per row (−69%)**, and the last layer writes straight into the
  caller's buffer.
- **Row tiling** in codegen loads each weight row once per 4 batch rows:
  12.2 → 10.2 ms. A sweep found 4–8 rows best (2 rows: 12.4 ms, 16: 12.1 ms).
- Fusion's speed gain is small at batch 256 (under 5%), because every
  intermediate tensor fits in L2. At batch 4,096 the fused C kernel is 12%
  faster on layer 1. The memory saving holds at any size.
- Generated code reaches about 36 GFLOP/s versus 143 for the hand-written
  kernel. The gap is register blocking: LLVM keeps the accumulators in memory.
  Emitting register tiles in the IR is the obvious next step.

## Run it

Needs gcc or clang with OpenMP, clang/LLVM, Go ≥ 1.22, Python 3 + NumPy.

```bash
make build            # C bench + kernel library + kprof
make test             # C correctness, 9 Python tests (passes, codegen, kernels), Go tests
make profile          # benchmarks + kprof text and HTML reports in results/
make test MARCH=x86-64-v3   # AVX2 build; MARCH=x86-64 for the scalar fallback

python3 compiler/tinyfuse.py model/eeg_p300_mlp.json -o build/eeg --print-ir
./build/kprof -trace results/kernel_trace.json -compare naive,simd_fused_omp -html report.html
```

Traces use the Chrome trace event format, so `results/*_trace.json` also opens
in [Perfetto](https://ui.perfetto.dev) as a timeline.

## Layout

```
kernels/kernels.c      naive, SIMD (AVX-512/AVX2/scalar), packing, vexp, OpenMP, FMA probe
bench/bench.c          per-layer kernel benchmark + peak/bandwidth probes + trace
bench/bench_model.py   end-to-end benchmark across all backends + per-node trace
compiler/tinyfuse.py   frontend, passes, LLVM IR codegen, clang driver
compiler/runtime.py    ctypes loaders, parameter packing, NumPy reference
profiler/              kprof (Go): trace parsing, stats, roofline, HTML report, tests
tests/                 unittest suite
model/                 the EEG network spec
```

## Limits

This is CPU work: SIMD lanes and cache blocking, not GPU or accelerator
kernels. The ideas carry over directly (tiling to a memory hierarchy, fusing
epilogues, measuring against a roofline), but there is no CUDA, Triton or
NKI code here. Only inference (forward pass) is implemented.
