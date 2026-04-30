"""
Correctness tests for the FIdelity Table II Conv INPUT16/WEIGHT16 masks.

For NVDLA Conv layers:
  - INPUT16 affects 16 neurons at one 2D output position across
    consecutive output channels.
  - WEIGHT16 affects all or a subset of 16 consecutive neurons in one row
    of one output channel.
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inject_ops import create_conv_input16_mask, create_conv_weight16_mask


PROVIDERS = ["CPUExecutionProvider"]


def build_model(nodes, shape, fp16, name):
    dtype = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    graph = helper.make_graph(
        nodes,
        name,
        inputs=[helper.make_tensor_value_info("y", dtype, list(shape))],
        outputs=[helper.make_tensor_value_info("y_masked", dtype, list(shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    return model


def run_model(model, x):
    sess = ort.InferenceSession(model.SerializeToString(), providers=PROVIDERS)
    return sess.run(None, {"y": x})[0]


def assert_masked_same_shape(out: np.ndarray, x: np.ndarray) -> None:
    assert out.shape == x.shape, (
        "Masked output must match input layout.\n"
        f"  input shape NCHW:  {tuple(x.shape)}\n"
        f"  output shape NCHW: {tuple(out.shape)}"
    )


def nonzero_coords(out):
    return np.argwhere(out.astype(np.float32) != 0.0)


def _geom_header(which: str, out: np.ndarray, coords: np.ndarray, expected_nnz: int) -> str:
    return (
        f"{which}: expected exactly {expected_nnz} nonzero values "
        f"(INPUT16/WEIGHT16 block vs channels or width).\n"
        f"  output shape NCHW: {tuple(out.shape)}\n"
        f"  nonzero count: {len(coords)} (expected {expected_nnz})\n"
        f"  nonzero indices as rows [n, c, h, w]:\n{coords}"
    )


def assert_consecutive(values, *, label: str):
    sorted_vals = sorted(int(v) for v in values)
    start = sorted_vals[0]
    expected = list(range(start, start + len(sorted_vals)))
    assert sorted_vals == expected, (
        f"{label}: channels or widths must be one consecutive block.\n"
        f"  got indices (sorted): {sorted_vals}\n"
        f"  expected range: {expected}"
    )


def assert_conv_input16_geometry(out, block_length):
    coords = nonzero_coords(out)
    expected = min(block_length, out.shape[1])
    hdr = _geom_header("Conv INPUT16 mask", out, coords, expected)

    assert len(coords) == expected, hdr

    n_values = set(coords[:, 0])
    c_values = set(coords[:, 1])
    h_values = set(coords[:, 2])
    w_values = set(coords[:, 3])

    assert len(n_values) == 1, (
        f"{hdr}\n"
        f"  INPUT16 keeps one batch row: unique N indices should be 1, got {sorted(n_values)}."
    )
    assert len(h_values) == 1, (
        f"{hdr}\n"
        f"  INPUT16 keeps one spatial (H) row: unique H indices should be 1, got {sorted(h_values)}."
    )
    assert len(w_values) == 1, (
        f"{hdr}\n"
        f"  INPUT16 keeps one spatial (W) column: unique W indices should be 1, got {sorted(w_values)}."
    )
    assert len(c_values) == expected, (
        f"{hdr}\n"
        f"  INPUT16 spans {expected} output channels at that pixel; "
        f"unique C indices should be {expected}, got {len(c_values)}: {sorted(c_values)}."
    )
    assert_consecutive(c_values, label="INPUT16 channel indices")

    n = int(coords[0, 0])
    h = int(coords[0, 2])
    w = int(coords[0, 3])
    c_sorted = sorted(int(x) for x in coords[:, 1])
    print(
        "  PASS INPUT16 geometry: "
        f"NCHW={tuple(out.shape)} dtype={out.dtype}  "
        f"nnz={len(coords)} (block<= {block_length})  "
        f"pixel (n,h,w)=({n},{h},{w})  channels C={c_sorted[0]}..{c_sorted[-1]} "
        f"(contiguous {len(c_sorted)})",
        flush=True,
    )
    print(f"  nonzero indices [n c h w]:\n{coords}", flush=True)


def assert_conv_weight16_geometry(out, block_length):
    coords = nonzero_coords(out)
    expected = min(block_length, out.shape[3])
    hdr = _geom_header("Conv WEIGHT16 mask", out, coords, expected)

    assert len(coords) == expected, hdr

    n_values = set(coords[:, 0])
    c_values = set(coords[:, 1])
    h_values = set(coords[:, 2])
    w_values = set(coords[:, 3])

    assert len(n_values) == 1, (
        f"{hdr}\n"
        f"  WEIGHT16 keeps one batch row: unique N indices should be 1, got {sorted(n_values)}."
    )
    assert len(c_values) == 1, (
        f"{hdr}\n"
        f"  WEIGHT16 keeps one output channel: unique C indices should be 1, got {sorted(c_values)}."
    )
    assert len(h_values) == 1, (
        f"{hdr}\n"
        f"  WEIGHT16 keeps one kernel row (H): unique H indices should be 1, got {sorted(h_values)}."
    )
    assert len(w_values) == expected, (
        f"{hdr}\n"
        f"  WEIGHT16 spans {expected} consecutive input widths; "
        f"unique W indices should be {expected}, got {len(w_values)}: {sorted(w_values)}."
    )
    assert_consecutive(w_values, label="WEIGHT16 width indices")

    n = int(coords[0, 0])
    c = int(coords[0, 1])
    h = int(coords[0, 2])
    w_sorted = sorted(int(x) for x in coords[:, 3])
    print(
        "  PASS WEIGHT16 geometry: "
        f"NCHW={tuple(out.shape)} dtype={out.dtype}  "
        f"nnz={len(coords)} (block<= {block_length})  "
        f"stripe (n,c,h)=({n},{c},{h})  widths W={w_sorted[0]}..{w_sorted[-1]} "
        f"(contiguous {len(w_sorted)})",
        flush=True,
    )
    print(f"  nonzero indices [n c h w]:\n{coords}", flush=True)


def test_conv_input16_keeps_one_spatial_position_across_16_channels_fp32():
    print(
        "\n=== INPUT16 fp32: one spatial (h,w), up to 16 consecutive C ===",
        flush=True,
    )
    x = np.ones((2, 32, 5, 7), dtype=np.float32)
    nodes = create_conv_input16_mask("y", "y_masked", block_length=16, fp16=False)
    out = run_model(build_model(nodes, x.shape, fp16=False, name="conv_input16_fp32"), x)

    assert_masked_same_shape(out, x)
    assert_conv_input16_geometry(out, block_length=16)


def test_conv_input16_small_channel_count_keeps_one_spatial_position():
    print(
        "\n=== INPUT16 fp16: C < block_length (expect all C kept) ===",
        flush=True,
    )
    x = np.ones((1, 10, 4, 6), dtype=np.float16)
    nodes = create_conv_input16_mask("y", "y_masked", block_length=16, fp16=True)
    out = run_model(build_model(nodes, x.shape, fp16=True, name="conv_input16_small_c"), x)

    assert_masked_same_shape(out, x)
    assert_conv_input16_geometry(out, block_length=16)


def test_conv_weight16_keeps_one_channel_row_with_16_consecutive_columns_fp32():
    print(
        "\n=== WEIGHT16 fp32: one (n,c,h), up to 16 consecutive W ===",
        flush=True,
    )
    x = np.ones((2, 8, 4, 24), dtype=np.float32)
    nodes = create_conv_weight16_mask("y", "y_masked", block_length=16, fp16=False)
    out = run_model(build_model(nodes, x.shape, fp16=False, name="conv_weight16_fp32"), x)

    assert_masked_same_shape(out, x)
    assert_conv_weight16_geometry(out, block_length=16)


def test_conv_weight16_small_width_keeps_one_channel_row_subset():
    print(
        "\n=== WEIGHT16 fp16: W < block_length (expect all W kept) ===",
        flush=True,
    )
    x = np.ones((1, 8, 4, 10), dtype=np.float16)
    nodes = create_conv_weight16_mask("y", "y_masked", block_length=16, fp16=True)
    out = run_model(build_model(nodes, x.shape, fp16=True, name="conv_weight16_small_w"), x)

    assert_masked_same_shape(out, x)
    assert_conv_weight16_geometry(out, block_length=16)


if __name__ == "__main__":
    _TESTS = (
        test_conv_input16_keeps_one_spatial_position_across_16_channels_fp32,
        test_conv_input16_small_channel_count_keeps_one_spatial_position,
        test_conv_weight16_keeps_one_channel_row_with_16_consecutive_columns_fp32,
        test_conv_weight16_small_width_keeps_one_channel_row_subset,
    )
    for _fn in _TESTS:
        print(f"running {_fn.__name__} …", flush=True)
        _fn()
    print("Conv INPUT16/WEIGHT16 mask tests passed (all %d)." % len(_TESTS))
