#!/usr/bin/env python3
"""End-to-end benchmark of the EEG P300 network across every backend.

Compiles the model with tinyfuse (fused / unfused / untiled), runs the
hand-written C kernels and NumPy, checks every output against a float64
reference, prints a summary, and writes:
  results/model_results.json   median latency per backend
  results/model_trace.json     per-node Chrome/Perfetto trace for kprof

usage: python3 bench/bench_model.py [--batch 256] [--repeats 50]
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "compiler"))
import runtime as rt          # noqa: E402
import tinyfuse               # noqa: E402

MODEL = os.path.join(ROOT, "model", "eeg_p300_mlp.json")
BUILD = os.path.join(ROOT, "build")
RESULTS = os.path.join(ROOT, "results")


def timeit(fn, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e6)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    spec = rt.load_json(MODEL)
    layers = rt.model_layers(spec)
    params = rt.init_params(layers, a.seed)
    x = np.random.default_rng(a.seed + 1).uniform(0, 1, (a.batch, spec["input_features"])).astype(np.float32)
    ref = rt.reference_forward(x, layers, params)
    flops = sum(2 * L["K"] * L["N"] for L in layers) * a.batch

    builds = {
        "tinyfuse_fused": dict(fuse=True, tile_rows=4),
        "tinyfuse_unfused": dict(fuse=False, tile_rows=4),
        "tinyfuse_fused_untiled": dict(fuse=True, tile_rows=1),
    }
    models = {}
    for name, opts in builds.items():
        tinyfuse.compile_model(MODEL, os.path.join(BUILD, name), verbose=False, **opts)
        models[name] = rt.CompiledModel(os.path.join(BUILD, name), params)
    ck = rt.CKernels(os.path.join(BUILD, "libtfkernels.so"), layers, params)

    backends = {
        "c_naive": lambda: ck.forward(x, "naive"),
        "numpy_blas": lambda: rt.numpy_forward(x, layers, params),
        "tinyfuse_unfused": lambda: models["tinyfuse_unfused"](x),
        "tinyfuse_fused_untiled": lambda: models["tinyfuse_fused_untiled"](x),
        "tinyfuse_fused": lambda: models["tinyfuse_fused"](x),
        "c_simd_fused": lambda: ck.forward(x, "simd"),
        "c_simd_fused_omp": lambda: ck.forward(x, "simd_omp"),
    }

    rows = []
    print(f"EEG P300 MLP | batch={a.batch} | {flops / 1e6:.1f} MFLOP per forward | isa={ck.isa}")
    print(f"{'backend':<24}{'median_us':>11}{'GFLOP/s':>10}{'vs naive':>10}{'max_err':>11}")
    base = None
    for name, fn in backends.items():
        out = np.asarray(fn(), dtype=np.float64).reshape(ref.shape)
        err = float(np.max(np.abs(out - ref)))
        reps = a.repeats if name != "c_naive" else max(5, a.repeats // 10)
        med = statistics.median(timeit(fn, reps))
        base = base or med
        rows.append({"backend": name, "median_us": med, "gflops": flops / (med * 1e3),
                     "speedup_vs_naive": base / med, "max_abs_err": err})
        print(f"{name:<24}{med:>11.1f}{flops / (med * 1e3):>10.2f}{base / med:>9.1f}x{err:>11.2e}")
        assert err < 1e-4, f"{name} disagrees with the reference (max err {err})"

    # Per-node trace of the fused and unfused compiled models for kprof.
    events, t0 = [], time.perf_counter()
    for name in ("tinyfuse_unfused", "tinyfuse_fused"):
        m = models[name]
        y = np.empty((a.batch, 1), dtype=np.float32)
        bufs = m.buffers(x, y)
        for _ in range(3):
            m(x, y)
        for r in range(a.repeats):
            for j, node in enumerate(m.info["nodes"]):
                s = time.perf_counter()
                m.run_node(j, bufs, a.batch)
                e = time.perf_counter()
                events.append({
                    "name": f"{node['fn']} {node['desc']}", "cat": name, "ph": "X",
                    "pid": 2, "tid": 1 if name == "tinyfuse_unfused" else 2,
                    "ts": (s - t0) * 1e6, "dur": (e - s) * 1e6,
                    "args": {"variant": name, "node": j, "op": node["op"],
                             "flops": node["flops_per_row"] * a.batch,
                             "bytes": 4 * a.batch * (node["K"] + node["N"]) + 4 * node["K"] * node["N"]}})

    with open(os.path.join(RESULTS, "model_results.json"), "w") as fh:
        json.dump({"batch": a.batch, "isa": ck.isa, "mflop_per_forward": flops / 1e6,
                   "passes": models["tinyfuse_fused"].info["passes"],
                   "plan_fused": models["tinyfuse_fused"].info["plan"],
                   "plan_unfused": models["tinyfuse_unfused"].info["plan"],
                   "results": rows}, fh, indent=2)
    with open(os.path.join(RESULTS, "model_trace.json"), "w") as fh:
        json.dump({"metadata": {"batch": a.batch, "isa": ck.isa}, "traceEvents": events}, fh)
    print("all backends match the float64 reference (max abs err < 1e-4)")


if __name__ == "__main__":
    main()
