#!/usr/bin/env python3
"""tinyfuse: a tiny ahead-of-time compiler for MLP models, targeting LLVM IR.

Pipeline
    model JSON (nn.Sequential-style)
      -> frontend: graph IR of Linear / activation nodes
      -> pass 1  decompose:  Linear -> MatMul + BiasAdd
      -> pass 2  fuse:       MatMul -> BiasAdd -> Act  ==>  Dense(act)  (epilogue fusion)
      -> pass 3  plan:       liveness-based buffer assignment (in-place elementwise
                             ops, two ping-pong arena slots)
      -> codegen:            one LLVM IR function per node + tf_forward(), with
                             row tiling and constant weight offsets baked in
      -> clang -O3:          LLVM's loop vectorizer widens the inner loops and
                             -fveclib=libmvec maps exp() to SIMD math routines

Usage
    python3 compiler/tinyfuse.py model/eeg_p300_mlp.json -o build/eeg \
        [--no-fuse] [--tile-rows 4] [--emit-only] [--print-ir]

Outputs build/eeg.ll (LLVM IR), build/eeg.so (shared library) and
build/eeg.json (node list, parameter layout and memory plan for the runtime).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field

ACTIVATIONS = ("relu", "celu", "sigmoid")


def load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Graph IR
# --------------------------------------------------------------------------
@dataclass
class Value:
    name: str
    width: int                     # features per row (rows = batch size at runtime)
    buffer: str = ""               # filled in by the memory planner


@dataclass
class Node:
    op: str                        # linear | matmul | bias_add | act | dense
    inputs: list[Value]
    output: Value
    attrs: dict = field(default_factory=dict)

    def describe(self) -> str:
        a = self.attrs
        if self.op in ("linear", "matmul", "dense"):
            extra = f" act={a['act']}" if self.op == "dense" else ""
            return f"{self.op}[{a['K']}x{a['N']}]{extra}"
        if self.op == "act":
            return f"act[{a['act']}]"
        return self.op


@dataclass
class Graph:
    name: str
    nodes: list[Node]
    input: Value
    output: Value
    params: list[dict]             # one entry per Linear: K, N


def frontend(spec: dict) -> Graph:
    """Turn the JSON layer list into a graph of Linear and activation nodes."""
    x = Value("x", spec["input_features"])
    cur, nodes, params = x, [], []
    for i, layer in enumerate(spec["layers"]):
        t = layer["type"]
        if t == "linear":
            K, N = cur.width, layer["out_features"]
            out = Value(f"t{i}", N)
            nodes.append(Node("linear", [cur], out, {"K": K, "N": N, "param": len(params)}))
            params.append({"K": K, "N": N})
        elif t in ACTIVATIONS:
            out = Value(f"t{i}", cur.width)
            nodes.append(Node("act", [cur], out, {"act": t, "N": cur.width}))
        else:
            raise ValueError(f"unsupported layer type: {t}")
        cur = out
    return Graph(spec["name"], nodes, x, cur, params)


# --------------------------------------------------------------------------
# Passes
# --------------------------------------------------------------------------
def decompose_linear(g: Graph) -> Graph:
    """Linear(x) = MatMul(x) + bias. Splitting it exposes BiasAdd to fusion."""
    out = []
    for n in g.nodes:
        if n.op == "linear":
            mm = Value(n.output.name + "_mm", n.attrs["N"])
            out.append(Node("matmul", n.inputs, mm, dict(n.attrs)))
            out.append(Node("bias_add", [mm], n.output, dict(n.attrs)))
        else:
            out.append(n)
    g.nodes = out
    return g


def fuse_epilogues(g: Graph) -> Graph:
    """MatMul -> BiasAdd -> Act  ==>  Dense(act).

    The bias add and activation become the epilogue of the matmul, applied to
    each output row while it is still in L1, so the two extra full passes over
    the activation tensor (and their intermediate tensors) disappear."""
    out, i, nodes = [], 0, g.nodes
    while i < len(nodes):
        n = nodes[i]
        if (n.op == "matmul" and i + 1 < len(nodes) and nodes[i + 1].op == "bias_add"
                and nodes[i + 1].inputs[0] is n.output):
            act, end, last = "none", i + 2, nodes[i + 1]
            if end < len(nodes) and nodes[end].op == "act" and nodes[end].inputs[0] is last.output:
                act, last, end = nodes[end].attrs["act"], nodes[end], end + 1
            attrs = dict(n.attrs, act=act)
            out.append(Node("dense", n.inputs, last.output, attrs))
            i = end
        else:
            out.append(n)
            i += 1
    g.nodes = out
    return g


def plan_memory(g: Graph) -> dict:
    """Assign every tensor to a buffer.

    1. Walk back from the graph output through in-place ops: those tensors are
       written straight into the caller's output buffer.
    2. Walk forward: elementwise ops (bias_add, act) run in place; every other
       op writes to one of two arena slots, never the slot holding its input.
    3. Size each slot by the widest tensor it ever holds.
    Returns statistics comparing the plan with one buffer per tensor."""
    in_place = ("bias_add", "act")
    producer = {id(n.output): n for n in g.nodes}
    to_output, v = set(), g.output
    while id(v) in producer:
        to_output.add(id(v))
        n = producer[id(v)]
        if n.op not in in_place:
            break
        v = n.inputs[0]

    g.input.buffer = "input"
    widths = {"slot0": 0, "slot1": 0}
    for n in g.nodes:
        if id(n.output) in to_output:
            n.output.buffer = "output"
        elif n.op in in_place:
            n.output.buffer = n.inputs[0].buffer
        else:
            n.output.buffer = "slot1" if n.inputs[0].buffer == "slot0" else "slot0"
            widths[n.output.buffer] = max(widths[n.output.buffer], n.output.width)

    naive = sum(n.output.width for n in g.nodes if id(n.output) not in to_output)
    return {"slot0_width": widths["slot0"], "slot1_width": widths["slot1"],
            "arena_floats_per_row": widths["slot0"] + widths["slot1"],
            "naive_floats_per_row": naive}


def param_layout(g: Graph) -> list[dict]:
    """Weights are stored transposed (Wt[K][N]) so codegen's inner loop over N
    reads contiguously; each bias follows its weight block."""
    off, layout = 0, []
    for p in g.params:
        w_off, b_off = off, off + p["K"] * p["N"]
        layout.append({"K": p["K"], "N": p["N"], "w_offset": w_off, "b_offset": b_off,
                       "w_layout": "transposed [K][N]"})
        off = b_off + p["N"]
    return layout


# --------------------------------------------------------------------------
# LLVM IR codegen
# --------------------------------------------------------------------------
def f32(x: float) -> str:
    """LLVM float literal: a double with float precision, written in hex."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", float(struct.unpack("<f", struct.pack("<f", x))[0])))[0]


class Fn:
    """Tiny builder for one LLVM function. Loop counters live in allocas, which
    mem2reg turns into SSA phis at -O1 and above."""

    def __init__(self, name: str, args: list[str]):
        self.name, self.args = name, args
        self.allocas: list[str] = []
        self.body: list[str] = []
        self.n = 0

    def tmp(self, prefix="t") -> str:
        self.n += 1
        return f"%{prefix}{self.n}"

    def label(self, prefix) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def emit(self, line: str):
        self.body.append("  " + line)

    def block(self, label: str):
        self.body.append(f"{label}:")

    def gep(self, base: str, idx: str) -> str:
        r = self.tmp("p")
        self.emit(f"{r} = getelementptr inbounds float, ptr {base}, i64 {idx}")
        return r

    def load(self, ptr: str) -> str:
        r = self.tmp("v")
        self.emit(f"{r} = load float, ptr {ptr}, align 4")
        return r

    def store(self, val: str, ptr: str):
        self.emit(f"store float {val}, ptr {ptr}, align 4")

    def binop(self, op: str, ty: str, a: str, b: str) -> str:
        r = self.tmp()
        self.emit(f"{r} = {op} {ty} {a}, {b}")
        return r

    def loop(self, start: str, end: str, body, step: int = 1) -> None:
        """for (i = start; i < end; i += step) body(i)"""
        ctr = self.tmp("i")
        self.allocas.append(f"  {ctr}.addr = alloca i64, align 8")
        self.emit(f"store i64 {start}, ptr {ctr}.addr, align 8")
        cond, bod, done = self.label("cond"), self.label("body"), self.label("done")
        self.emit(f"br label %{cond}")
        self.block(cond)
        iv = self.tmp("iv")
        self.emit(f"{iv} = load i64, ptr {ctr}.addr, align 8")
        if step == 1:
            c = self.binop("icmp slt", "i64", iv, end)
        else:  # iv + step <= end
            nxt = self.binop("add nsw", "i64", iv, str(step))
            c = self.binop("icmp sle", "i64", nxt, end)
        self.emit(f"br i1 {c}, label %{bod}, label %{done}")
        self.block(bod)
        body(iv)
        inc = self.binop("add nsw", "i64", iv, str(step))
        self.emit(f"store i64 {inc}, ptr {ctr}.addr, align 8")
        self.emit(f"br label %{cond}")
        self.block(done)
        return ctr

    def render(self, attrs="") -> str:
        head = f"define void @{self.name}({', '.join(self.args)}) {attrs}{{"
        return "\n".join([head, "entry:", *self.allocas, *self.body, "  ret void", "}"])


def emit_act(f: Fn, v: str, act: str) -> str:
    zero, one = f32(0.0), f32(1.0)
    if act == "none":
        return v
    if act == "relu":
        r = f.tmp()
        f.emit(f"{r} = call float @llvm.maxnum.f32(float {v}, float {zero})")
        return r
    if act == "celu":   # max(x,0) + exp(min(x,0)) - 1, branch-free so it vectorizes
        neg, pos, e = f.tmp(), f.tmp(), f.tmp()
        f.emit(f"{neg} = call float @llvm.minnum.f32(float {v}, float {zero})")
        f.emit(f"{pos} = call float @llvm.maxnum.f32(float {v}, float {zero})")
        f.emit(f"{e} = call float @llvm.exp.f32(float {neg})")
        em1 = f.binop("fsub", "float", e, one)
        return f.binop("fadd", "float", pos, em1)
    if act == "sigmoid":
        nv, e = f.tmp(), f.tmp()
        f.emit(f"{nv} = fneg float {v}")
        f.emit(f"{e} = call float @llvm.exp.f32(float {nv})")
        d = f.binop("fadd", "float", one, e)
        return f.binop("fdiv", "float", one, d)
    raise ValueError(act)


def gen_matmul_like(name: str, n: Node, lay: dict, tile_rows: int, fused: bool) -> Fn:
    """out[b][:] = (bias if fused else 0) + sum_k in[b][k] * Wt[k][:], then act.

    Rows are processed `tile_rows` at a time: every weight row Wt[k][:] is
    loaded once and used for all rows in the tile (register/L1 blocking).
    The n loop is innermost and contiguous, so LLVM vectorizes it."""
    K, N = n.attrs["K"], n.attrs["N"]
    act = n.attrs.get("act", "none") if fused else "none"
    f = Fn(name, ["ptr noalias readonly %params", "ptr noalias readonly %in",
                  "ptr noalias %out", "i64 %B"])
    wt = f.gep("%params", str(lay["w_offset"]))
    bias = f.gep("%params", str(lay["b_offset"]))

    def tile(rows):
        def run(b0):
            xr, yr = [], []
            for r in range(rows):
                b = f.binop("add nsw", "i64", b0, str(r)) if r else b0
                xr.append(f.gep("%in", f.binop("mul nsw", "i64", b, str(K))))
                yr.append(f.gep("%out", f.binop("mul nsw", "i64", b, str(N))))

            def init(nn):        # y[r][n] = bias[n] (fused) or 0
                v = f.load(f.gep(bias, nn)) if fused else f32(0.0)
                for r in range(rows):
                    f.store(v, f.gep(yr[r], nn))
            f.loop("0", str(N), init)

            def kstep(k):
                xs = [f.load(f.gep(xr[r], k)) for r in range(rows)]
                wrow = f.gep(wt, f.binop("mul nsw", "i64", k, str(N)))

                def nstep(nn):
                    w = f.load(f.gep(wrow, nn))
                    for r in range(rows):
                        p = f.gep(yr[r], nn)
                        acc = f.load(p)
                        s = f.tmp()
                        f.emit(f"{s} = call float @llvm.fmuladd.f32(float {xs[r]}, float {w}, float {acc})")
                        f.store(s, p)
                f.loop("0", str(N), nstep)
            f.loop("0", str(K), kstep)

            if act != "none":     # epilogue on rows that are still in L1
                def epi(nn):
                    for r in range(rows):
                        p = f.gep(yr[r], nn)
                        f.store(emit_act(f, f.load(p), act), p)
                f.loop("0", str(N), epi)
        return run

    if tile_rows > 1:
        # main loop over full tiles, then a 1-row remainder loop
        ctr = f.loop("0", "%B", tile(tile_rows), step=tile_rows)
        start = f.tmp("iv")
        f.emit(f"{start} = load i64, ptr {ctr}.addr, align 8")
        f.loop(start, "%B", tile(1))
    else:
        f.loop("0", "%B", tile(1))
    return f


def gen_bias_add(name: str, n: Node, lay: dict) -> Fn:
    N = n.attrs["N"]
    f = Fn(name, ["ptr noalias readonly %params", "ptr noalias %buf", "i64 %B"])
    bias = f.gep("%params", str(lay["b_offset"]))

    def row(b):
        yr = f.gep("%buf", f.binop("mul nsw", "i64", b, str(N)))

        def col(nn):
            p = f.gep(yr, nn)
            f.store(f.binop("fadd", "float", f.load(p), f.load(f.gep(bias, nn))), p)
        f.loop("0", str(N), col)
    f.loop("0", "%B", row)
    return f


def gen_act(name: str, n: Node) -> Fn:
    N = n.attrs["N"]
    f = Fn(name, ["ptr noalias %buf", "i64 %B"])
    total = f.binop("mul nsw", "i64", "%B", str(N))

    def el(i):
        p = f.gep("%buf", i)
        f.store(emit_act(f, f.load(p), n.attrs["act"]), p)
    f.loop("0", total, el)
    return f


def codegen(g: Graph, layout: list[dict], plan: dict, tile_rows: int) -> tuple[str, list[dict]]:
    fns, meta = [], []
    for j, n in enumerate(g.nodes):
        name = f"tf_node{j}_{n.op}"
        lay = layout[n.attrs["param"]] if "param" in n.attrs else None
        if n.op in ("matmul", "dense"):
            fn = gen_matmul_like(name, n, lay, tile_rows, fused=(n.op == "dense"))
            sig = "params,in,out,B"
        elif n.op == "bias_add":
            fn, sig = gen_bias_add(name, n, lay), "params,buf,B"
        elif n.op == "act":
            fn, sig = gen_act(name, n), "buf,B"
        else:
            raise ValueError(n.op)
        fns.append(fn.render())
        K = n.attrs.get("K", 0)
        N = n.attrs["N"]
        flops = 2 * K * N if n.op in ("matmul", "dense") else N
        meta.append({"index": j, "fn": name, "op": n.op, "desc": n.describe(),
                     "signature": sig, "in": n.inputs[0].buffer, "out": n.output.buffer,
                     "K": K, "N": N, "act": n.attrs.get("act", "none"),
                     "flops_per_row": flops})

    # tf_forward(params, x, y, B, arena): resolves buffers and calls each node.
    f = Fn("tf_forward", ["ptr %params", "ptr %x", "ptr %y", "i64 %B", "ptr %arena"])
    slot = f.binop("mul nsw", "i64", "%B", str(plan["slot0_width"]))
    bufs = {"input": "%x", "output": "%y", "slot0": "%arena", "slot1": f.gep("%arena", slot)}
    for m in meta:
        if m["signature"] == "params,in,out,B":
            f.emit(f"call void @{m['fn']}(ptr %params, ptr {bufs[m['in']]}, ptr {bufs[m['out']]}, i64 %B)")
        elif m["signature"] == "params,buf,B":
            f.emit(f"call void @{m['fn']}(ptr %params, ptr {bufs[m['out']]}, i64 %B)")
        else:
            f.emit(f"call void @{m['fn']}(ptr {bufs[m['out']]}, i64 %B)")
    fns.append(f.render())

    header = [f"; tinyfuse output for model '{g.name}'",
              "declare float @llvm.fmuladd.f32(float, float, float)",
              "declare float @llvm.maxnum.f32(float, float)",
              "declare float @llvm.minnum.f32(float, float)",
              "declare float @llvm.exp.f32(float)", ""]
    return "\n".join(header) + "\n\n".join(fns) + "\n", meta


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def compile_model(spec_path: str, out_prefix: str, fuse: bool = True, tile_rows: int = 4,
                  emit_only: bool = False, verbose: bool = True) -> dict:
    spec = load_json(spec_path)
    g = frontend(spec)
    log = [f"frontend:  {len(g.nodes)} nodes  " + " -> ".join(n.describe() for n in g.nodes)]
    g = decompose_linear(g)
    log.append(f"decompose: {len(g.nodes)} nodes")
    if fuse:
        before = len(g.nodes)
        g = fuse_epilogues(g)
        log.append(f"fuse:      {before} -> {len(g.nodes)} nodes  "
                   + " -> ".join(n.describe() for n in g.nodes))
    plan = plan_memory(g)
    log.append(f"plan:      arena {plan['arena_floats_per_row']} floats/row "
               f"(slot0 {plan['slot0_width']} + slot1 {plan['slot1_width']}); "
               f"one buffer per tensor would need {plan['naive_floats_per_row']}")
    layout = param_layout(g)
    ir, meta = codegen(g, layout, plan, tile_rows)

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    with open(out_prefix + ".ll", "w") as fh:
        fh.write(ir)
    info = {"model": g.name, "fused": fuse, "tile_rows": tile_rows,
            "input_features": g.input.width, "output_features": g.output.width,
            "param_floats": layout[-1]["b_offset"] + layout[-1]["N"],
            "params": layout, "plan": plan, "nodes": meta, "passes": log}
    with open(out_prefix + ".json", "w") as fh:
        json.dump(info, fh, indent=2)

    if not emit_only:
        clang = shutil.which("clang") or "clang"
        march = os.environ.get("TINYFUSE_MARCH", "native")
        cmd = [clang, "-O3", f"-march={march}", "-fveclib=libmvec", "-fno-math-errno",
               "-Wno-override-module", "-Wno-invalid-feature-combination",
               "-shared", "-fPIC", out_prefix + ".ll", "-o", out_prefix + ".so", "-lmvec", "-lm"]
        subprocess.run(cmd, check=True)
        log.append("codegen:   " + " ".join(os.path.basename(c) for c in cmd))
    if verbose:
        print("\n".join(log))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("-o", "--out", default="build/model")
    ap.add_argument("--no-fuse", action="store_true", help="skip the epilogue fusion pass")
    ap.add_argument("--tile-rows", type=int, default=4, help="batch rows per tile (1 = no tiling)")
    ap.add_argument("--emit-only", action="store_true", help="write LLVM IR but do not run clang")
    ap.add_argument("--print-ir", action="store_true")
    a = ap.parse_args(argv)
    compile_model(a.model, a.out, fuse=not a.no_fuse, tile_rows=a.tile_rows, emit_only=a.emit_only)
    if a.print_ir:
        with open(a.out + ".ll") as fh:
            print(fh.read())


if __name__ == "__main__":
    sys.exit(main())
