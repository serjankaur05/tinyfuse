"""Tests for the tinyfuse compiler passes, generated code and C kernels.

Run with:  python3 -m unittest discover -s tests -v   (after `make build`)
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "compiler"))
import runtime as rt          # noqa: E402
import tinyfuse as tf         # noqa: E402

MODEL = os.path.join(ROOT, "model", "eeg_p300_mlp.json")
KERNEL_LIB = os.path.join(ROOT, "build", "libtfkernels.so")
# Odd batch sizes exercise the 1-row remainder paths after 4- and 8-row tiles.
BATCHES = (1, 3, 5, 8, 13, 64, 257)


class PassTests(unittest.TestCase):
    def setUp(self):
        self.spec = rt.load_json(MODEL)

    def test_frontend_mirrors_sequential(self):
        g = tf.frontend(self.spec)
        self.assertEqual([n.op for n in g.nodes], ["linear", "act"] * 5)
        self.assertEqual(g.output.width, 1)

    def test_decompose_then_fuse(self):
        g = tf.decompose_linear(tf.frontend(self.spec))
        self.assertEqual(len(g.nodes), 15)
        g = tf.fuse_epilogues(g)
        self.assertEqual([n.op for n in g.nodes], ["dense"] * 5)
        self.assertEqual([n.attrs["act"] for n in g.nodes],
                         ["celu", "relu", "relu", "relu", "sigmoid"])

    def test_fusion_without_trailing_activation(self):
        spec = {"name": "t", "input_features": 4,
                "layers": [{"type": "linear", "out_features": 3}]}
        g = tf.fuse_epilogues(tf.decompose_linear(tf.frontend(spec)))
        self.assertEqual(len(g.nodes), 1)
        self.assertEqual(g.nodes[0].attrs["act"], "none")

    def test_memory_plan_unfused(self):
        g = tf.decompose_linear(tf.frontend(self.spec))
        plan = tf.plan_memory(g)
        # The last matmul and its in-place ops write straight to the output.
        self.assertEqual([n.output.buffer for n in g.nodes[-3:]], ["output"] * 3)
        # Never read and write the same arena slot in a non-elementwise op.
        for n in g.nodes:
            if n.op == "matmul":
                self.assertNotEqual(n.inputs[0].buffer, n.output.buffer)
        self.assertEqual(plan["arena_floats_per_row"], 1500)
        self.assertEqual(plan["naive_floats_per_row"], 4830)

    def test_param_layout_is_contiguous(self):
        layout = tf.param_layout(tf.frontend(self.spec))
        off = 0
        for lay in layout:
            self.assertEqual(lay["w_offset"], off)
            off = lay["b_offset"] + lay["N"]
        self.assertEqual(off, sum(K * N + N for K, N in
                                  [(226, 500), (500, 1000), (1000, 100), (100, 10), (10, 1)]))


class GeneratedCodeTests(unittest.TestCase):
    """Compile real LLVM IR with clang and compare against NumPy."""

    @classmethod
    def setUpClass(cls):
        cls.spec = rt.load_json(MODEL)
        cls.layers = rt.model_layers(cls.spec)
        cls.params = rt.init_params(cls.layers, seed=3)
        cls.tmp = tempfile.mkdtemp()
        cls.models = {}
        for name, fuse, tile in [("fused", True, 4), ("unfused", False, 4),
                                 ("fused_t1", True, 1), ("fused_t8", True, 8)]:
            prefix = os.path.join(cls.tmp, name)
            tf.compile_model(MODEL, prefix, fuse=fuse, tile_rows=tile, verbose=False)
            cls.models[name] = rt.CompiledModel(prefix, cls.params)

    def test_all_builds_match_reference(self):
        rng = np.random.default_rng(7)
        for B in BATCHES:
            x = rng.uniform(0, 1, (B, 226)).astype(np.float32)
            ref = rt.reference_forward(x, self.layers, self.params)
            for name, m in self.models.items():
                with self.subTest(build=name, batch=B):
                    np.testing.assert_allclose(m(x), ref, atol=1e-5)

    def test_node_by_node_equals_forward(self):
        m = self.models["unfused"]
        x = np.random.default_rng(1).uniform(0, 1, (9, 226)).astype(np.float32)
        y = np.empty((9, 1), dtype=np.float32)
        bufs = m.buffers(x, y)
        for j in range(len(m.info["nodes"])):
            m.run_node(j, bufs, 9)
        np.testing.assert_allclose(y, m(x), rtol=0, atol=0)

    def test_ir_has_expected_structure(self):
        with open(os.path.join(self.tmp, "fused.ll")) as fh:
            ir = fh.read()
        self.assertEqual(ir.count("define void @tf_node"), 5)
        self.assertIn("@llvm.fmuladd.f32", ir)
        self.assertIn("@llvm.exp.f32", ir)          # CELU and sigmoid epilogues


@unittest.skipUnless(os.path.exists(KERNEL_LIB), "run `make build` first")
class KernelTests(unittest.TestCase):
    def test_every_variant_and_activation(self):
        rng = np.random.default_rng(11)
        # Shapes hit full panels, column tails, narrow tails and row remainders.
        for B, K, N in [(1, 7, 1), (5, 10, 1), (13, 33, 37), (64, 226, 500), (9, 100, 10)]:
            for act in ("none", "relu", "celu", "sigmoid"):
                layers = [{"K": K, "N": N, "act": act}]
                params = rt.init_params(layers, seed=B + K)
                x = rng.uniform(-1, 1, (B, K)).astype(np.float32)
                ref = rt.reference_forward(x, layers, params)
                ck = rt.CKernels(KERNEL_LIB, layers, params)
                for variant in ("naive", "simd", "simd_omp"):
                    with self.subTest(shape=(B, K, N), act=act, variant=variant):
                        np.testing.assert_allclose(ck.forward(x, variant), ref, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
