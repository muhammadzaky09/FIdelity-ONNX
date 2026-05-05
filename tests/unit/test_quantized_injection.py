"""
Unit test for create_quantized_fault_injection.

The function outputs an INT32 *delta* tensor:
    delta[rand_idx] = (int8(x[rand_idx]) XOR (1 << bit_pos)) - int8(x[rand_idx])
    delta[i]        = 0  for all i != rand_idx

Verification:
  1. Exactly one element of delta is non-zero (at rand_idx).
  2. That element equals the expected bit-flip delta.
  3. Shape is preserved.

No custom ops required — runs on CPU.
"""
import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto
import onnxruntime as ort

from inject_ops import create_quantized_fault_injection


def build_model(input_shape, bit_position, rand_idx_name, fp16=False, is_signed=True):
    prec   = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    x_vi   = helper.make_tensor_value_info("x",           prec,              input_shape)
    ri_vi  = helper.make_tensor_value_info(rand_idx_name, TensorProto.INT64, [])
    bp_vi  = helper.make_tensor_value_info("bit_pos_inject", TensorProto.INT32, [])
    out_vi = helper.make_tensor_value_info("delta",        TensorProto.INT32, input_shape)

    inj_nodes = create_quantized_fault_injection(
        input_name="x",
        output_name="delta",
        fp16=fp16,
        is_signed=is_signed,
        rand_idx_name=rand_idx_name,
        bit_pos_name="bit_pos_inject",
    )

    graph = helper.make_graph(
        inj_nodes,
        "test_qfi",
        [x_vi, ri_vi, bp_vi],
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
        "rand_idx_inject": np.array(rand_idx,    dtype=np.int64),
        "bit_pos_inject":  np.array(bit_position, dtype=np.int32),
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


@pytest.mark.parametrize(
    ("label", "x_np", "rand_idx", "bit_position", "fp16", "is_signed"),
    [
        (
            "fp32 2D bit3",
            np.array([[-5.0, 3.0, 7.0], [-1.0, 12.0, 2.0]], dtype=np.float32),
            2,
            3,
            False,
            True,
        ),
        (
            "fp32 1D bit7 sign",
            np.array([10.0, 3.0, 5.0, -2.0], dtype=np.float32),
            1,
            7,
            False,
            True,
        ),
        (
            "fp16 2D bit2",
            np.array([[1.0, -3.0, 6.0, -8.0]], dtype=np.float16),
            3,
            2,
            True,
            True,
        ),
        (
            "fp32 1D unsigned bit0",
            np.array([6.0, 2.0, 9.0], dtype=np.float32),
            0,
            0,
            False,
            False,
        ),
    ],
)
def test_quantized_fault_injection_delta(label, x_np, rand_idx, bit_position, fp16, is_signed):
    run_case(label, x_np, rand_idx, bit_position, fp16=fp16, is_signed=is_signed)
