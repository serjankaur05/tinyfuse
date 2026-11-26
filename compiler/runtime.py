"""Runtime helpers: build parameters, load tinyfuse-compiled models and the C
kernel library with ctypes, and run a NumPy reference."""
from __future__ import annotations

import ctypes
import json
import os

import numpy as np

ACT_IDS = {"none": 0, "relu": 1, "celu": 2, "sigmoid": 3}
FP = ctypes.POINTER(ctypes.c_float)


def load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def ptr(a: np.ndarray):
    return a.ctypes.data_as(FP)


def act_np(x: np.ndarray, act: str) -> np.ndarray:
    if act == "relu":
        return np.maximum(x, 0)
    if act == "celu":
        return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    return x


def model_layers(spec: dict) -> list[dict]:
    """[(K, N, act)] for each Linear in the spec, with its following activation."""
    layers, width = [], spec["input_features"]
    for layer in spec["layers"]:
        if layer["type"] == "linear":
            layers.append({"K": width, "N": layer["out_features"], "act": "none"})
            width = layer["out_features"]
        else:
            layers[-1]["act"] = layer["type"]
    return layers


def init_params(layers: list[dict], seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Same distribution as torch.nn.Linear's default init: U(-1/sqrt(K), 1/sqrt(K))."""
    rng = np.random.default_rng(seed)
    out = []
    for L in layers:
        bound = 1.0 / np.sqrt(L["K"])
        W = rng.uniform(-bound, bound, (L["N"], L["K"])).astype(np.float32)
        b = rng.uniform(-bound, bound, L["N"]).astype(np.float32)
        out.append((W, b))
    return out


def reference_forward(x: np.ndarray, layers, params) -> np.ndarray:
    """float64 NumPy reference."""
    h = x.astype(np.float64)
    for L, (W, b) in zip(layers, params):
        h = act_np(h @ W.T.astype(np.float64) + b, L["act"])
    return h


def numpy_forward(x: np.ndarray, layers, params) -> np.ndarray:
    """float32 NumPy (BLAS-backed) baseline."""
    h = x
    for L, (W, b) in zip(layers, params):
        h = act_np(h @ W.T + b, L["act"]).astype(np.float32, copy=False)
    return h


class CompiledModel:
    """A model produced by compiler/tinyfuse.py."""

    def __init__(self, prefix: str, params):
        self.info = load_json(prefix + ".json")
        self.lib = ctypes.CDLL(os.path.abspath(prefix + ".so"))
        flat = np.zeros(self.info["param_floats"], dtype=np.float32)
        for lay, (W, b) in zip(self.info["params"], params):
            K, N = lay["K"], lay["N"]
            flat[lay["w_offset"]:lay["w_offset"] + K * N] = W.T.reshape(-1)   # Wt [K][N]
            flat[lay["b_offset"]:lay["b_offset"] + N] = b
        self.params = flat
        self.fwd = self.lib.tf_forward
        self.fwd.argtypes = [FP, FP, FP, ctypes.c_int64, FP]
        self.fwd.restype = None
        self.arena = None
        self.node_fns = []
        for m in self.info["nodes"]:
            fn = getattr(self.lib, m["fn"])
            fn.restype = None
            self.node_fns.append(fn)

    def _arena(self, B):
        need = B * self.info["plan"]["arena_floats_per_row"]
        if self.arena is None or self.arena.size < need:
            self.arena = np.zeros(max(need, 1), dtype=np.float32)
        return self.arena

    def __call__(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        B = x.shape[0]
        if y is None:
            y = np.empty((B, self.info["output_features"]), dtype=np.float32)
        self.fwd(ptr(self.params), ptr(x), ptr(y), B, ptr(self._arena(B)))
        return y

    def buffers(self, x, y):
        B = x.shape[0]
        arena = self._arena(B)
        s0 = self.info["plan"]["slot0_width"] * B
        return {"input": x, "output": y, "slot0": arena[:s0], "slot1": arena[s0:]}

    def run_node(self, j: int, bufs: dict, B: int):
        m, fn = self.info["nodes"][j], self.node_fns[j]
        if m["signature"] == "params,in,out,B":
            fn(ptr(self.params), ptr(bufs[m["in"]]), ptr(bufs[m["out"]]), ctypes.c_int64(B))
        elif m["signature"] == "params,buf,B":
            fn(ptr(self.params), ptr(bufs[m["out"]]), ctypes.c_int64(B))
        else:
            fn(ptr(bufs[m["out"]]), ctypes.c_int64(B))


class CKernels:
    """The hand-written kernels in kernels/kernels.c, built as a shared library."""

    def __init__(self, path: str, layers, params):
        self.lib = ctypes.CDLL(os.path.abspath(path))
        L = self.lib
        L.linear_naive.argtypes = [FP, FP, FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.activation.argtypes = [FP, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.pack_weights.argtypes = [FP, FP, ctypes.c_int, ctypes.c_int]
        L.linear_simd.argtypes = [FP, FP, FP, FP] + [ctypes.c_int] * 5
        L.linear_simd_omp.argtypes = [FP, FP, FP, FP] + [ctypes.c_int] * 4
        L.simd_isa.restype = ctypes.c_char_p
        self.isa = L.simd_isa().decode()
        self.layers, self.params = layers, params
        self.packed = []
        for Ld, (W, b) in zip(layers, params):
            Wp = np.empty(Ld["K"] * Ld["N"], dtype=np.float32)
            L.pack_weights(ptr(np.ascontiguousarray(W)), ptr(Wp), Ld["K"], Ld["N"])
            self.packed.append(Wp)
        self.bufs = {}

    def _buf(self, i, B, N):
        key = (i % 2, B)
        need = B * max(Ld["N"] for Ld in self.layers)
        if key not in self.bufs:
            self.bufs[key] = np.empty(need, dtype=np.float32)
        return self.bufs[key][:B * N]

    def forward(self, x: np.ndarray, variant: str) -> np.ndarray:
        B, h = x.shape[0], x
        for i, (Ld, (W, b), Wp) in enumerate(zip(self.layers, self.params, self.packed)):
            K, N, act = Ld["K"], Ld["N"], ACT_IDS[Ld["act"]]
            y = self._buf(i, B, N)
            if variant == "naive":
                self.lib.linear_naive(ptr(h), ptr(W), ptr(b), ptr(y), B, K, N)
                self.lib.activation(ptr(y), B, N, act)
            elif variant == "simd":
                self.lib.linear_simd(ptr(h), ptr(Wp), ptr(b), ptr(y), B, K, N, act, 1)
            elif variant == "simd_omp":
                self.lib.linear_simd_omp(ptr(h), ptr(Wp), ptr(b), ptr(y), B, K, N, act)
            else:
                raise ValueError(variant)
            h = y
        return h.reshape(B, self.layers[-1]["N"])
