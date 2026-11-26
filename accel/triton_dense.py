"""Fused dense layer (matmul + bias + activation) as a Triton GPU kernel.

Runs the EEG P300 network on an NVIDIA GPU (tested target: Colab T4) and
compares, at fp16 with fp32 accumulation:

  torch_eager        nn.Linear (cuBLAS addmm) + a separate activation kernel
  torch_compile      torch.compile(model)              (Inductor)
  torch_cudagraphs   torch.compile(mode="reduce-overhead") (Inductor + CUDA graphs)
  triton_fused       one Triton kernel per layer: bias + activation fused into
                     the matmul epilogue, autotuned tile sizes
  triton_cudagraph   the same five Triton kernels replayed from one CUDA graph

The Triton kernel is registered as a PyTorch custom op (torch.library), so it
also composes with torch.compile. Every backend is checked against an fp32
reference. Results go to results/gpu_results.json and a per-layer trace
(results/gpu_trace.json) that kprof can read.

usage: python accel/triton_dense.py [--batches 256 4096]
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import triton
import triton.language as tl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "compiler"))
import runtime as rt  # noqa: E402  (model spec + PyTorch-style init, NumPy only)

ACTS = {"none": 0, "relu": 1, "celu": 2, "sigmoid": 3}
RESULTS = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _configs():
    cfgs = []
    for bm, bn, bk, warps, stages in [
        (64, 64, 32, 4, 2), (64, 128, 32, 4, 2), (128, 64, 32, 4, 2),
        (128, 128, 32, 8, 2), (32, 64, 64, 4, 2), (16, 32, 64, 2, 2),
        (64, 32, 64, 4, 3), (128, 32, 32, 4, 2),
    ]:
        cfgs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                                  num_warps=warps, num_stages=stages))
    return cfgs


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def fused_dense_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
                       stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
                       ACT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
    """y[M, N] = act(x[M, K] @ w[N, K]^T + b[N]).

    Each program owns a BLOCK_M x BLOCK_N output tile, accumulates it in fp32
    registers over K (tl.dot maps onto tensor cores for fp16 inputs), then adds
    the bias and applies the activation before the single store to HBM. w is
    read in PyTorch's [N, K] layout through strides, so no transpose copy."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(w_ptr + k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                    mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    if ACT == 1:
        acc = tl.maximum(acc, 0.0)
    elif ACT == 2:  # CELU(alpha=1) = max(x,0) + exp(min(x,0)) - 1
        acc = tl.maximum(acc, 0.0) + tl.exp(tl.minimum(acc, 0.0)) - 1.0
    elif ACT == 3:
        acc = 1.0 / (1.0 + tl.exp(-acc))
    y = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y, acc.to(y_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _launch(x, w, b, y, act):
    M, K = x.shape
    N = w.shape[0]

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    return grid, (x, w, b, y, M, N, K, x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                  y.stride(0), y.stride(1), act)


# ---------------------------------------------------------------------------
# PyTorch custom op
# ---------------------------------------------------------------------------
try:  # torch >= 2.6: Triton-aware custom op that torch.compile can trace through
    from torch.library import triton_op, wrap_triton

    @triton_op("tinyfuse::fused_dense", mutates_args={})
    def fused_dense(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, act: int) -> torch.Tensor:
        y = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)
        grid, args = _launch(x, w, b, y, act)
        wrap_triton(fused_dense_kernel)[grid](*args)
        return y
    CUSTOM_OP = "torch.library.triton_op"
except ImportError:  # older torch: plain Python wrapper
    def fused_dense(x, w, b, act):
        y = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)
        grid, args = _launch(x, w, b, y, act)
        fused_dense_kernel[grid](*args)
        return y
    CUSTOM_OP = "python wrapper (torch.library.triton_op unavailable)"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_torch_model(layers, params, device, dtype):
    mods = []
    for L, (W, b) in zip(layers, params):
        lin = nn.Linear(L["K"], L["N"])
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(W))
            lin.bias.copy_(torch.from_numpy(b))
        mods.append(lin)
        mods.append({"relu": nn.ReLU(), "celu": nn.CELU(), "sigmoid": nn.Sigmoid(),
                     "none": nn.Identity()}[L["act"]])
    return nn.Sequential(*mods).to(device=device, dtype=dtype).eval()


class TritonMLP(nn.Module):
    """Same network, every Linear + activation is one fused Triton kernel."""

    def __init__(self, torch_model, layers):
        super().__init__()
        lins = [m for m in torch_model if isinstance(m, nn.Linear)]
        self.ws = [lin.weight.detach().contiguous() for lin in lins]
        self.bs = [lin.bias.detach().contiguous() for lin in lins]
        self.acts = [ACTS[L["act"]] for L in layers]

    def forward(self, x):
        for w, b, a in zip(self.ws, self.bs, self.acts):
            x = fused_dense(x, w, b, a)
        return x


class GraphReplay:
    """Capture a callable on a static input into one CUDA graph and replay it,
    removing per-kernel CPU launch overhead."""

    def __init__(self, fn, example):
        self.static_x = example.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):          # warm up (also finishes Triton autotuning)
                fn(self.static_x)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_y = fn(self.static_x)

    def __call__(self, x):
        self.static_x.copy_(x)
        self.graph.replay()
        return self.static_y


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def bench_ms(fn):
    """Median ms using triton.testing.do_bench (flushes L2 between runs)."""
    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="median")


def count_kernels(fn):
    """CUDA kernels launched by one call, from torch.profiler."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    n = 0
    for e in prof.events():
        dev = getattr(e, "device_type", None)
        if dev is not None and str(dev).endswith("CUDA"):
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[256, 4096])
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU (Colab: Runtime > Change runtime type > T4 GPU)"
    os.makedirs(RESULTS, exist_ok=True)
    dev, dt = "cuda", torch.float16
    gpu = torch.cuda.get_device_name()
    print(f"gpu={gpu}  torch={torch.__version__}  triton={triton.__version__}  custom op: {CUSTOM_OP}")

    spec = rt.load_json(os.path.join(ROOT, "model", "eeg_p300_mlp.json"))
    layers = rt.model_layers(spec)
    params = rt.init_params(layers, seed=0)
    ref_fp32 = build_torch_model(layers, params, dev, torch.float32)
    model = build_torch_model(layers, params, dev, dt)
    tmodel = TritonMLP(model, layers)

    results = {"gpu": gpu, "torch": torch.__version__, "triton": triton.__version__,
               "custom_op": CUSTOM_OP, "dtype": "fp16 inputs, fp32 accumulation", "batches": {}}
    events, ts = [], 0.0

    for B in a.batches:
        x = torch.rand(B, spec["input_features"], device=dev, dtype=dt)
        with torch.inference_mode():
            ref = ref_fp32(x.float())
        flops = sum(2 * L["K"] * L["N"] for L in layers) * B

        compiled = torch.compile(model)
        cudagraphs = torch.compile(model, mode="reduce-overhead")
        backends = {
            "torch_eager": lambda: model(x),
            "torch_compile": lambda: compiled(x),
            "torch_cudagraphs": lambda: cudagraphs(x),
            "triton_fused": lambda: tmodel(x),
        }
        with torch.inference_mode():
            backends["triton_fused"]()                       # autotune before capture
            replay = GraphReplay(tmodel, x)
        backends["triton_cudagraph"] = lambda: replay(x)

        rows = {}
        print(f"\nbatch={B}  {flops / 1e6:.1f} MFLOP per forward")
        print(f"{'backend':<20}{'median_us':>11}{'TFLOP/s':>9}{'kernels':>9}{'max_err':>10}")
        with torch.inference_mode():
            for name, fn in backends.items():
                out = fn().float()
                err = (out - ref).abs().max().item()
                assert err < 2e-2, f"{name} differs from fp32 reference by {err}"
                ms = bench_ms(fn)
                kern = count_kernels(fn) if name in ("torch_eager", "triton_fused") else None
                rows[name] = {"median_us": ms * 1e3, "tflops": flops / (ms * 1e9),
                              "kernels_per_forward": kern, "max_abs_err": err}
                print(f"{name:<20}{ms * 1e3:>11.1f}{flops / (ms * 1e9):>9.2f}"
                      f"{kern if kern is not None else '-':>9}{err:>10.1e}")

            # Per layer: Triton fused kernel vs torch Linear + activation.
            per_layer = []
            h = x
            for i, (L, w, b) in enumerate(zip(layers, tmodel.ws, tmodel.bs)):
                act_mod = model[2 * i + 1]
                lin = model[2 * i]
                t_ms = bench_ms(lambda: fused_dense(h, w, b, ACTS[L["act"]]))
                e_ms = bench_ms(lambda: act_mod(lin(h)))
                f = 2 * B * L["K"] * L["N"]
                per_layer.append({"layer": i + 1, "K": L["K"], "N": L["N"], "act": L["act"],
                                  "triton_us": t_ms * 1e3, "torch_us": e_ms * 1e3,
                                  "speedup": e_ms / t_ms})
                for variant, us in (("torch_linear_act", e_ms * 1e3), ("triton_fused", t_ms * 1e3)):
                    events.append({"name": f"L{i + 1} dense {L['K']}x{L['N']}+{L['act']}",
                                   "cat": variant, "ph": "X", "pid": 3, "tid": B, "ts": ts, "dur": us,
                                   "args": {"layer": i + 1, "variant": f"{variant}_b{B}", "flops": f,
                                            "bytes": 2 * (B * L["K"] + L["K"] * L["N"] + B * L["N"])}})
                    ts += us
                h = fused_dense(h, w, b, ACTS[L["act"]])
        print(f"{'layer':<8}{'shape':<12}{'torch_us':>10}{'triton_us':>11}{'speedup':>9}")
        for r in per_layer:
            print(f"L{r['layer']:<7}{str(r['K']) + 'x' + str(r['N']):<12}{r['torch_us']:>10.1f}"
                  f"{r['triton_us']:>11.1f}{r['speedup']:>8.2f}x")
        results["batches"][str(B)] = {"mflop": flops / 1e6, "backends": rows, "per_layer": per_layer}

    with open(os.path.join(RESULTS, "gpu_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    with open(os.path.join(RESULTS, "gpu_trace.json"), "w") as fh:
        json.dump({"metadata": {"device": gpu}, "traceEvents": events}, fh)
    print("\nall GPU backends match the fp32 reference (max abs err < 2e-2)")


if __name__ == "__main__":
    main()
