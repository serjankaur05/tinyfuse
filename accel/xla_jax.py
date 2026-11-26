"""The EEG network in JAX: what XLA does with it, through StableHLO (MLIR).

  1. Lowers the jitted forward pass to StableHLO, an MLIR dialect, and saves
     it (results/eeg.stablehlo.mlir). Counts the ops XLA starts from.
  2. Compiles it and saves the optimized HLO (results/eeg.optimized.hlo).
     Counts the fusion computations and the kernels XLA will launch, and lists
     what each fusion contains, so XLA's fusion decisions can be compared with
     tinyfuse's dense-epilogue fusion.
  3. Times op-by-op execution (jax.disable_jit) against jax.jit, and checks
     the output against the float64 NumPy reference.

Runs on whatever JAX finds (GPU on a Colab T4 runtime, otherwise CPU).
usage: python accel/xla_jax.py [--batch 256]
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "compiler"))
import runtime as rt  # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def act(x, kind):
    if kind == "relu":
        return jax.nn.relu(x)
    if kind == "celu":
        return jax.nn.celu(x, alpha=1.0)
    if kind == "sigmoid":
        return jax.nn.sigmoid(x)
    return x


def make_forward(layers):
    kinds = [L["act"] for L in layers]

    def forward(params, x):
        for (W, b), kind in zip(params, kinds):
            x = act(x @ W.T + b, kind)
        return x
    return forward


def time_us(fn, repeats=200):
    for _ in range(10):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append((time.perf_counter() - t) * 1e6)
    return statistics.median(ts)


def summarize_hlo(text: str):
    """Count fusion computations and the instructions inside each one, plus the
    kernel-launching instructions (fusions, library calls, dots) in ENTRY."""
    fusions, current = {}, None
    for line in text.splitlines():
        head = re.match(r"\s*(%?[\w.\-]+)\s+\(.*\)\s*->.*\{\s*$", line)
        if head and not line.lstrip().startswith("ENTRY"):
            name = head.group(1).lstrip("%")
            current = name if "fus" in name else None
            if current:
                fusions[current] = collections.Counter()
            continue
        if line.strip() == "}":
            current = None
            continue
        if current:
            op = re.search(r"=\s*[\w\[\]{},:\s.]*?\s([a-z][a-z\-]*)\(", line)
            if op:
                fusions[current][op.group(1)] += 1
    entry = text[text.find("ENTRY"):]
    launches = len(re.findall(r"\s(fusion|custom-call|convolution|dot)\(", entry))
    return fusions, launches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    backend = jax.default_backend()
    print(f"jax={jax.__version__}  backend={backend}  devices={jax.devices()}")

    spec = rt.load_json(os.path.join(ROOT, "model", "eeg_p300_mlp.json"))
    layers = rt.model_layers(spec)
    np_params = rt.init_params(layers, seed=0)
    params = [(jnp.asarray(W), jnp.asarray(b)) for W, b in np_params]
    x_np = np.random.default_rng(1).uniform(0, 1, (a.batch, spec["input_features"])).astype(np.float32)
    x = jnp.asarray(x_np)

    forward = make_forward(layers)
    jitted = jax.jit(forward)

    # 1) StableHLO (MLIR) before XLA optimizations
    lowered = jitted.lower(params, x)
    shlo = lowered.as_text()                           # StableHLO MLIR text
    with open(os.path.join(RESULTS, "eeg.stablehlo.mlir"), "w") as fh:
        fh.write(shlo)
    shlo_ops = collections.Counter(re.findall(r"\b(stablehlo\.[a-z_]+)", shlo))

    # 2) optimized HLO after XLA's passes (fusion, layout assignment, ...)
    compiled = lowered.compile()
    hlo = compiled.as_text()
    with open(os.path.join(RESULTS, "eeg.optimized.hlo"), "w") as fh:
        fh.write(hlo)
    fusions, launches = summarize_hlo(hlo)

    # 3) correctness and timing
    out = np.asarray(jitted(params, x))
    ref = rt.reference_forward(x_np, layers, np_params)
    err = float(np.abs(out - ref).max())
    jit_us = time_us(lambda: jitted(params, x))
    with jax.disable_jit():
        eager_us = time_us(lambda: forward(params, x), repeats=50)

    print(f"\nStableHLO (MLIR) ops before optimization: {sum(shlo_ops.values())}")
    for op, n in shlo_ops.most_common():
        print(f"  {n:>3}  {op}")
    print(f"\nXLA optimized HLO: {len(fusions)} fusion computations, "
          f"~{launches} kernel launches in the entry computation")
    for name, ops in fusions.items():
        print(f"  {name}: " + ", ".join(f"{k} x{v}" for k, v in ops.most_common()))
    print(f"\nop-by-op: {eager_us:.1f} us   jit: {jit_us:.1f} us   "
          f"speedup {eager_us / jit_us:.1f}x   max_err {err:.1e}")
    assert err < 1e-4, f"JAX output differs from reference by {err}"

    with open(os.path.join(RESULTS, "xla_results.json"), "w") as fh:
        json.dump({"jax": jax.__version__, "backend": backend, "batch": a.batch,
                   "stablehlo_ops": dict(shlo_ops), "stablehlo_op_total": sum(shlo_ops.values()),
                   "xla_fusions": {k: dict(v) for k, v in fusions.items()},
                   "xla_kernel_launches": launches, "eager_us": eager_us, "jit_us": jit_us,
                   "jit_speedup": eager_us / jit_us, "max_abs_err": err}, fh, indent=2)
    print("saved results/eeg.stablehlo.mlir and results/eeg.optimized.hlo")


if __name__ == "__main__":
    main()
