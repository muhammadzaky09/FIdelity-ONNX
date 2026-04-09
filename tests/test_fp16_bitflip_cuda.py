"""
Runtime test for create_random_bitflip_injection using llama/onnx_bitflip.so on CUDA.

Checks:
  1. Exactly one element changes between original and faulty output.
  2. The changed element is a valid FP16 bit-flip at the specified bit_position.
  3. Shape is preserved.

Run with:
    conda run -n fidelity-onnx python testing/test_fp16_bitflip_cuda.py
"""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from inject_ops import create_random_bitflip_injection

REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SO      = os.path.join(REPO, 'llama', 'onnx_bitflip.so')
BIT_POS = 7


def fp16_bits(x: np.float16) -> int:
    return struct.unpack('<H', struct.pack('<e', x))[0]


def build_model():
    A        = helper.make_tensor_value_info("A",               TensorProto.FLOAT16, [4, 4])
    B        = helper.make_tensor_value_info("B",               TensorProto.FLOAT16, [4, 8])
    ri       = helper.make_tensor_value_info("rand_idx_inject", TensorProto.INT64,   [])
    bp       = helper.make_tensor_value_info("bit_pos_inject",  TensorProto.INT32,   [])
    Y_vi     = helper.make_tensor_value_info("Y",               TensorProto.FLOAT16, [4, 8])
    Y_faulty = helper.make_tensor_value_info("Y_faulty",        TensorProto.FLOAT16, [4, 8])

    inj = create_random_bitflip_injection("Y", fp16=True,
                                          rand_idx_name="rand_idx_inject",
                                          bit_pos_name="bit_pos_inject")

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "B"], ["Y"])] + inj,
        "test_rbf_cuda",
        [A, B, ri, bp],
        [Y_faulty],
        value_info=[Y_vi],
    )
    return helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 18),
        helper.make_opsetid("custom.bitflip", 1),
    ])


def run_test():
    print(f"Library : {SO}")
    if not os.path.exists(SO):
        print(f"SKIP: library not found at {SO}")
        return

    opts = ort.SessionOptions()
    opts.register_custom_ops_library(SO)

    model = build_model()

    # Try CUDA first, fall back to CPU
    try:
        sess = ort.InferenceSession(
            model.SerializeToString(), opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        ep = sess.get_providers()[0]
    except Exception as e:
        print(f"SKIP: could not create session: {e}")
        return

    print(f"Execution provider : {ep}")
    if "CUDA" not in ep:
        print("SKIP: CUDA EP not active — BitFlip op requires GPU (CUDA 12 + cuDNN 9).")
        print("      Run this test on the GPU server.")
        return

    A  = np.random.randn(4, 4).astype(np.float16)
    B  = np.random.randn(4, 8).astype(np.float16)
    ri = np.array(3, dtype=np.int64)   # inject at flat index 3 — change this to test any element

    orig   = (A @ B).astype(np.float16)
    faulty = sess.run(None, {"A": A, "B": B,
                             "rand_idx_inject": ri,
                             "bit_pos_inject": np.array(BIT_POS, dtype=np.int32)})[0]

    flat_orig   = orig.flatten()
    flat_faulty = faulty.flatten()
    changed_idx = np.where(flat_orig != flat_faulty)[0]

    print(f"\nShape preserved  : {faulty.shape == orig.shape}")
    print(f"Changed elements : {len(changed_idx)}  (expected 1)")
    assert faulty.shape == orig.shape, "Shape changed!"
    assert len(changed_idx) == 1, f"Expected 1 changed element, got {len(changed_idx)}: {changed_idx}"

    idx    = changed_idx[0]
    o_bits = fp16_bits(flat_orig[idx])
    f_bits = fp16_bits(flat_faulty[idx])
    xor    = o_bits ^ f_bits

    print(f"Changed index    : {idx}")
    print(f"Original bits    : {o_bits:016b}  ({flat_orig[idx]})")
    print(f"Faulty   bits    : {f_bits:016b}  ({flat_faulty[idx]})")
    print(f"XOR              : {xor:016b}  (should be a single set bit at bit {BIT_POS})")

    is_single_bit = (xor != 0) and ((xor & (xor - 1)) == 0)
    flipped_bit   = xor.bit_length() - 1

    assert is_single_bit, f"Not a single bit flip! XOR={xor:016b}"
    assert flipped_bit == BIT_POS, \
        f"Wrong bit flipped: got bit {flipped_bit}, expected bit {BIT_POS}"

    print("\nPASS")


if __name__ == "__main__":
    run_test()
