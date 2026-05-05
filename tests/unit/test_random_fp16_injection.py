"""
Unit test for create_random_fault_injection with FP16 tensors.

Directly builds a minimal ONNX graph from the injection nodes (no graph.py
needed), runs it on CPU, and checks:
  1. Exactly one element changes.
  2. It changes at rand_idx.
  3. It changes to the expected FP16 random_value.
"""
import sys, os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inject_ops import create_random_fault_injection
from graph import bin2fp16          # same helper used by delta_init


SHAPE = (2, 8)                       # arbitrary FP16 tensor shape
INPUT_NAME = "X"
OUTPUT_NAME = "X_faulty"
IDX_NAME   = "rand_idx"


def build_injection_model(random_value: float) -> onnx.ModelProto:
    """Wrap create_random_fault_injection in a self-contained ONNX model."""
    inj_nodes = create_random_fault_injection(
        INPUT_NAME, random_value, fp16=True, rand_idx_name=IDX_NAME)

    # Graph inputs / outputs
    X   = helper.make_tensor_value_info(INPUT_NAME,  TensorProto.FLOAT16, list(SHAPE))
    idx = helper.make_tensor_value_info(IDX_NAME,    TensorProto.INT64,   [])
    Y   = helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT16, list(SHAPE))

    graph = helper.make_graph(inj_nodes, "random_inj",
                              inputs=[X, idx], outputs=[Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    return model


def run(model: onnx.ModelProto, x_fp16: np.ndarray, rand_idx: int) -> np.ndarray:
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        INPUT_NAME: x_fp16,
        IDX_NAME:   np.array(rand_idx, dtype=np.int64),
    })[0]                               # shape SHAPE, dtype float16
    return out.astype(np.float32)       # upcast for comparison


def test_random_fault_injection_fp16():
    np.random.seed(42)
    # Build a known FP16 input: values well away from zero so NaN fault is detectable
    x_fp32  = np.random.uniform(0.5, 1.5, SHAPE).astype(np.float32)
    x_fp16  = x_fp32.astype(np.float16)

    # Generate a random FP16 fault value (same logic as delta_init)
    np.random.seed(7)
    one_bin = ''.join(str(np.random.randint(0, 2)) for _ in range(16))
    fault_val_fp16 = np.float16(bin2fp16(one_bin))
    fault_val_fp32 = float(fault_val_fp16)

    print(f"[info]  fault_val = {fault_val_fp32:.6f}  "
          f"({'NaN' if np.isnan(fault_val_fp32) else 'finite'})")

    model = build_injection_model(fault_val_fp32)
    clean_fp32 = x_fp16.astype(np.float32).flatten()
    N = clean_fp32.size

    for rand_idx in [0, 5, N - 1]:
        faulty_fp32 = run(model, x_fp16, rand_idx).flatten()

        # Which elements differ?
        if np.isnan(fault_val_fp32):
            diff = np.isnan(faulty_fp32) & ~np.isnan(clean_fp32)
        else:
            diff = faulty_fp32 != clean_fp32

        changed = np.where(diff)[0].tolist()
        print(f"  rand_idx={rand_idx:2d} → changed={changed}, "
              f"faulty[{rand_idx}]={faulty_fp32[rand_idx]:.6f}  "
              f"(expected {fault_val_fp32:.6f})")

        assert len(changed) == 1, \
            f"Expected 1 changed element, got {len(changed)}: {changed}"
        assert changed[0] == rand_idx, \
            f"Changed at wrong position {changed[0]}, expected {rand_idx}"

        if np.isnan(fault_val_fp32):
            assert np.isnan(faulty_fp32[rand_idx]), \
                f"Expected NaN at {rand_idx}, got {faulty_fp32[rand_idx]}"
        else:
            assert abs(faulty_fp32[rand_idx] - fault_val_fp32) < 1e-4, \
                f"Value mismatch at {rand_idx}: got {faulty_fp32[rand_idx]}, expected {fault_val_fp32}"

    print("\n[PASS] create_random_fault_injection FP16 is correct.")


if __name__ == "__main__":
    test_random_fault_injection_fp16()
