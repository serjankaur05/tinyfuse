"""Fused dense layer as an AWS NKI kernel (Neuron Kernel Interface, Trainium).

The kernel computes y = act(x @ W^T + b) the way Trainium's NeuronCore wants it:

  * HBM -> SBUF tiles via nl.load. The partition dimension (axis 0, max 128
    lanes) carries the contraction dim K, so x is passed transposed (xT[K, M])
    and W as w[K, N].
  * nl.matmul(xT_tile, w_tile, transpose_x=True) runs on the Tensor Engine:
    a 128(K) x 128(M) stationary tile against a 128(K) x 512(N) moving tile,
    accumulating fp32 partial sums in PSUM across the K loop.
  * Epilogue on the Vector/Scalar engines: bias add and activation on the PSUM
    result, then one nl.store back to HBM, so the pre-activation tensor never
    round-trips through HBM.

There is no Trainium here, so correctness is checked with nki.simulate_kernel
on CPU, layer by layer and for the whole EEG network against NumPy. The host
pads every dimension to a tile multiple; the script reports how much of each
tile is useful work, which is the main inefficiency for tiny layers on a
128 x 512 systolic tile.

Install into an isolated venv (current neuronx-cc wheels are cp311-cp314):
    pip install uv && uv venv /content/nki -p 3.12
    uv pip install -p /content/nki numpy neuronx-cc \
        --extra-index-url https://pip.repos.neuron.amazonaws.com \
        --index-strategy unsafe-best-match
    /content/nki/bin/python accel/nki_dense.py
"""
import json
import os
import sys

import numpy as np

try:                                   # Neuron SDK layout: neuronxcc.nki
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
except ImportError:                    # standalone nki package in newer releases
    import nki
    import nki.language as nl


def simulate(kernel, *args):
    """Run a kernel on CPU in the NKI simulator (API name differs by release)."""
    if hasattr(nki, "simulate_kernel"):
        return nki.simulate_kernel(kernel, *args)
    return nki.simulate(kernel)(*args)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "compiler"))
import runtime as rt  # noqa: E402  (NumPy-only helpers: spec, init, reference)

TK = 128   # partition dim: contraction tile (nl.tile_size.pmax)
TM = 128   # stationary free dim: rows of the output tile (gemm_stationary_fmax)
TN = 512   # moving free dim: columns of the output tile (gemm_moving_fmax)


def make_kernel(act: str):
    """One specialized kernel per activation, so the epilogue is fixed at trace time."""

    @nki.jit
    def fused_dense(xT, w, bias_rep):
        # xT: [K, M], w: [K, N], bias_rep: [TM, N] (bias repeated on all partitions)
        K, M = xT.shape
        N = w.shape[1]
        y = nl.ndarray((M, N), dtype=xT.dtype, buffer=nl.shared_hbm)
        for m in nl.affine_range(M // TM):
            for n in nl.affine_range(N // TN):
                acc = nl.zeros((TM, TN), nl.float32, buffer=nl.psum)
                for k in nl.affine_range(K // TK):
                    x_t = nl.load(xT[nl.ds(k * TK, TK), nl.ds(m * TM, TM)])
                    w_t = nl.load(w[nl.ds(k * TK, TK), nl.ds(n * TN, TN)])
                    acc += nl.matmul(x_t, w_t, transpose_x=True)
                b_t = nl.load(bias_rep[nl.ds(0, TM), nl.ds(n * TN, TN)])
                out = nl.add(acc, b_t)
                if act == "relu":
                    out = nl.maximum(out, 0.0)
                elif act == "celu":   # max(x,0) + exp(min(x,0)) - 1
                    out = nl.add(nl.maximum(out, 0.0),
                                 nl.subtract(nl.exp(nl.minimum(out, 0.0)), 1.0))
                elif act == "sigmoid":
                    out = nl.sigmoid(out)
                nl.store(y[nl.ds(m * TM, TM), nl.ds(n * TN, TN)], value=out)
        return y

    return fused_dense


KERNELS = {a: make_kernel(a) for a in ("none", "relu", "celu", "sigmoid")}


def pad_to(n: int, t: int) -> int:
    return (n + t - 1) // t * t


def run_layer(x: np.ndarray, W: np.ndarray, b: np.ndarray, act: str):
    """Pad to tile multiples, run the kernel in the simulator, unpad.
    Returns (y, tile_utilization)."""
    M, K = x.shape
    N = W.shape[0]
    Mp, Kp, Np = pad_to(M, TM), pad_to(K, TK), pad_to(N, TN)
    xT = np.zeros((Kp, Mp), np.float32)
    xT[:K, :M] = x.T
    w = np.zeros((Kp, Np), np.float32)
    w[:K, :N] = W.T
    bias_rep = np.zeros((TM, Np), np.float32)
    bias_rep[:, :N] = b
    y = simulate(KERNELS[act], xT, w, bias_rep)
    util = (M * K * N) / (Mp * Kp * Np)
    return np.asarray(y)[:M, :N], util


def main():
    spec = rt.load_json(os.path.join(ROOT, "model", "eeg_p300_mlp.json"))
    layers = rt.model_layers(spec)
    params = rt.init_params(layers, seed=0)
    rng = np.random.default_rng(1)
    try:
        from importlib.metadata import version
        nki_version = version("neuronx-cc")
    except Exception:
        nki_version = getattr(nki, "__version__", "unknown")
    print(f"neuronx-cc {nki_version}")
    report = {"neuronx_cc": nki_version, "layers": [], "network": {}}

    # 1) every activation on an awkward shape (exercises padding on all dims)
    for act in KERNELS:
        x = rng.uniform(-1, 1, (37, 70)).astype(np.float32)
        W = rng.uniform(-0.1, 0.1, (45, 70)).astype(np.float32)
        b = rng.uniform(-0.1, 0.1, 45).astype(np.float32)
        y, _ = run_layer(x, W, b, act)
        ref = rt.act_np(x.astype(np.float64) @ W.T + b, act)
        err = float(np.abs(y - ref).max())
        print(f"unit act={act:<8} shape 37x70 -> 45   max_err={err:.2e}")
        assert err < 1e-4, f"activation {act}: error {err}"

    # 2) the EEG network, layer by layer, batch 256
    B = 256
    x = rng.uniform(0, 1, (B, spec["input_features"])).astype(np.float32)
    h = x
    print(f"\nEEG P300 network in the NKI simulator, batch {B}")
    print(f"{'layer':<7}{'shape':<12}{'act':<9}{'tile_util':>10}{'max_err':>10}")
    for i, (L, (W, b)) in enumerate(zip(layers, params)):
        y, util = run_layer(h, W, b, L["act"])
        ref = rt.act_np(h.astype(np.float64) @ W.T.astype(np.float64) + b, L["act"])
        err = float(np.abs(y - ref).max())
        report["layers"].append({"layer": i + 1, "K": L["K"], "N": L["N"], "act": L["act"],
                                 "tile_utilization": util, "max_abs_err": err})
        print(f"L{i + 1:<6}{str(L['K']) + 'x' + str(L['N']):<12}{L['act']:<9}{util:>9.1%}{err:>10.1e}")
        assert err < 1e-4, f"layer {i + 1}: error {err}"
        h = y.astype(np.float32)

    ref = rt.reference_forward(x, layers, params)
    err = float(np.abs(h - ref).max())
    useful = sum(2 * B * L["K"] * L["N"] for L in layers)
    padded = sum(2 * pad_to(B, TM) * pad_to(L["K"], TK) * pad_to(L["N"], TN) for L in layers)
    report["network"] = {"batch": B, "max_abs_err": err, "useful_mflop": useful / 1e6,
                         "padded_mflop": padded / 1e6, "utilization": useful / padded}
    print(f"\nwhole network vs float64 reference: max_err={err:.2e}")
    print(f"tile utilization: {useful / padded:.1%} of padded Tensor Engine work is useful")
    assert err < 1e-4

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "nki_results.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("NKI kernel matches NumPy for every activation and every EEG layer")


if __name__ == "__main__":
    main()
