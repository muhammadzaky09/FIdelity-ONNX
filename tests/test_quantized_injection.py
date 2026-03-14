"""
Unit test for create_quantized_fault_injection.

The function outputs a float *delta* tensor:
    delta[rand_idx] = float((int8(x[rand_idx]) XOR (1 << bit_pos)) - int8(x[rand_idx]))
    delta[i]        = 0.0  for all i != rand_idx

Verification:
  1. Exactly one element of delta is non-zero (at rand_idx).
  2. That element equals the expected bit-flip delta.
  3. Shape is preserved.

No custom ops required — runs on CPU.
"""
import sys, os
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from inject_ops import create_quantized_fault_injection


def build_model(input_shape, bit_position, rand_idx_name, fp16=False, is_signed=True):
    prec  = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    x_vi  = helper.make_tensor_value_info("x",              prec,            input_shape)
    ri_vi = helper.make_tensor_value_info(rand_idx_name,    TensorProto.INT64, [])
    out_vi = helper.make_tensor_value_info("delta",         prec,            input_shape)

    inj_nodes = create_quantized_fault_injection(
        input_name="x",
        output_name="delta",
        bit_position=bit_position,
        fp16=fp16,
        is_signed=is_signed,
        rand_idx_name=rand_idx_name,
    )

    graph = helper.make_graph(
        inj_nodes,
        "test_qfi",
        [x_vi, ri_vi],
        [out_vi],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def run_case(label, x_np, rand_idx, bit_position, fp16=False, is_signed=True):
    model = build_model(
        input_shape=list(x_np.shape),
        bit_position=bit_position,
        rand_idx_name="rand_idx_inject",
        fp16=fp16,
        is_signed=is_signed,
    )

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), opts,
                                providers=["CPUExecutionProvider"])

    delta = sess.run(None, {
        "x":               x_np,
        "rand_idx_inject": np.array(rand_idx, dtype=np.int64),
    })[0]

    flat_x     = x_np.flatten()
    flat_delta = delta.flatten().astype(np.float32)

    # ── assertions ────────────────────────────────────────────────────────────
    nonzero_idx = np.where(flat_delta != 0.0)[0]
    assert len(nonzero_idx) == 1, \
        f"[{label}] Expected exactly 1 non-zero delta, got {len(nonzero_idx)}: {nonzero_idx}"
    assert nonzero_idx[0] == rand_idx, \
        f"[{label}] Non-zero at wrong index: {nonzero_idx[0]} (expected {rand_idx})"
    assert delta.shape == x_np.shape, \
        f"[{label}] Shape mismatch: {delta.shape} vs {x_np.shape}"

    # Compute expected delta
    orig_int = int(np.int8(flat_x[rand_idx]) if is_signed else np.uint8(flat_x[rand_idx]))
    bitmask  = 1 << bit_position
    raw_xor = (orig_int & 0xFF) ^ (bitmask & 0xFF)   # unsigned XOR in Python int
    if is_signed:
        flipped_int = int(np.frombuffer(bytes([raw_xor]), dtype=np.int8)[0])
    else:
        flipped_int = raw_xor
    expected_delta = float(flipped_int - orig_int)

    actual_delta = float(flat_delta[rand_idx])
    assert actual_delta == expected_delta, \
        f"[{label}] Delta mismatch at [{rand_idx}]: got {actual_delta}, expected {expected_delta}"

    print(f"[{label}]")
    print(f"  rand_idx={rand_idx}  bit_pos={bit_position}  fp16={fp16}  signed={is_signed}")
    print(f"  x[{rand_idx}]     = {float(flat_x[rand_idx]):.1f}  "
          f"(int repr = {orig_int} = {orig_int & 0xFF:08b}b)")
    print(f"  flipped int  = {flipped_int} = {flipped_int & 0xFF:08b}b")
    print(f"  delta        = {actual_delta:.1f}  (expected {expected_delta:.1f})")
    print(f"  shape OK     : {delta.shape}")
    print(f"  PASS\n")


if __name__ == "__main__":
    # ── Case 1: fp32, signed, 2-D tensor, flip bit 3 ─────────────────────────
    # x[2] = 7.0 → int8=7 (0000_0111) XOR 8 (0000_1000) = 15 → delta = +8
    x1 = np.array([[-5.0, 3.0, 7.0],
                   [-1.0, 12.0, 2.0]], dtype=np.float32)
    run_case("fp32 2D bit3", x1, rand_idx=2, bit_position=3)

    # ── Case 2: fp32, signed, flip sign bit (bit 7) ───────────────────────────
    # x[1] = 3.0 → int8=3 (0000_0011) XOR 128 (1000_0000) = -125 → delta = -128
    x2 = np.array([10.0, 3.0, 5.0, -2.0], dtype=np.float32)
    run_case("fp32 1D bit7 (sign)", x2, rand_idx=1, bit_position=7)

    # ── Case 3: fp16 input, signed, 2-D tensor, flip bit 2 ───────────────────
    # x[3] = -8.0 → int8=-8 (1111_1000) XOR 4 (0000_0100) = int8(-4) → delta = +4
    x3 = np.array([[1.0, -3.0, 6.0, -8.0]], dtype=np.float16)
    run_case("fp16 2D bit2", x3, rand_idx=3, bit_position=2, fp16=True)

    # ── Case 4: unsigned (int4 mode), flip bit 0 ──────────────────────────────
    # x[0] = 6.0 → uint8=6 (0000_0110) XOR 1 (0000_0001) = 7 → delta = +1
    x4 = np.array([6.0, 2.0, 9.0], dtype=np.float32)
    run_case("fp32 1D unsigned bit0", x4, rand_idx=0, bit_position=0, is_signed=False)

    print("All cases PASSED.")
