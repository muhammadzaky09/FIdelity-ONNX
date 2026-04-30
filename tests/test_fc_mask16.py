"""
Correctness tests for the FIdelity Table II FC INPUT16/WEIGHT16 masks.

For NVDLA FC layers:
  - INPUT16 affects 16 consecutive output neurons.
  - WEIGHT16 affects one output neuron per selected row, for up to 16
    faulty neurons total.
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inject_ops import create_fc_input16_mask, create_fc_weight16_mask


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


def nonzero_coords(out):
    return np.argwhere(out.astype(np.float32) != 0.0)


def assert_consecutive(values):
    values = sorted(int(v) for v in values)
    assert values == list(range(values[0], values[0] + len(values))), values


def assert_fc_input16_geometry(out, block_length):
    flat = out.reshape(-1, out.shape[-1])
    coords = np.argwhere(flat.astype(np.float32) != 0.0)
    expected = min(block_length, flat.shape[1])

    assert len(coords) == expected
    row_values = set(coords[:, 0])
    feature_values = set(coords[:, 1])
    assert len(row_values) == 1
    assert len(feature_values) == expected
    assert_consecutive(feature_values)


def assert_fc_weight16_geometry(out, block_length):
    flat = out.reshape(-1, out.shape[-1])
    coords = np.argwhere(flat.astype(np.float32) != 0.0)
    expected = min(block_length, flat.shape[0])

    assert len(coords) == expected
    row_values = set(coords[:, 0])
    feature_values = set(coords[:, 1])
    assert len(row_values) == expected
    assert len(feature_values) == 1
    assert_consecutive(row_values)


def test_fc_input16_keeps_one_row_with_16_consecutive_outputs_fp32():
    x = np.ones((4, 32), dtype=np.float32)
    nodes = create_fc_input16_mask("y", "y_masked", block_length=16, fp16=False)
    out = run_model(build_model(nodes, x.shape, fp16=False, name="fc_input16_fp32"), x)

    assert out.shape == x.shape
    assert_fc_input16_geometry(out, block_length=16)


def test_fc_input16_small_output_count_keeps_one_row_outputs_fp16():
    x = np.ones((4, 10), dtype=np.float16)
    nodes = create_fc_input16_mask("y", "y_masked", block_length=16, fp16=True)
    out = run_model(build_model(nodes, x.shape, fp16=True, name="fc_input16_small_f"), x)

    assert out.shape == x.shape
    assert_fc_input16_geometry(out, block_length=16)


def test_fc_weight16_keeps_16_rows_at_one_output_feature_fp32():
    x = np.ones((24, 8), dtype=np.float32)
    nodes = create_fc_weight16_mask("y", "y_masked", block_length=16, fp16=False)
    out = run_model(build_model(nodes, x.shape, fp16=False, name="fc_weight16_fp32"), x)

    assert out.shape == x.shape
    assert_fc_weight16_geometry(out, block_length=16)


def test_fc_weight16_small_batch_keeps_all_rows_at_one_output_feature_fp16():
    x = np.ones((10, 8), dtype=np.float16)
    nodes = create_fc_weight16_mask("y", "y_masked", block_length=16, fp16=True)
    out = run_model(build_model(nodes, x.shape, fp16=True, name="fc_weight16_small_r"), x)

    assert out.shape == x.shape
    assert_fc_weight16_geometry(out, block_length=16)


def test_fc_masks_support_rank3_outputs_by_flattening_prefix_dims():
    x = np.ones((2, 12, 20), dtype=np.float32)

    in_nodes = create_fc_input16_mask("y", "y_masked", block_length=16, fp16=False)
    in_out = run_model(build_model(in_nodes, x.shape, fp16=False, name="fc_input16_rank3"), x)
    assert in_out.shape == x.shape
    assert_fc_input16_geometry(in_out, block_length=16)

    wt_nodes = create_fc_weight16_mask("y", "y_masked", block_length=16, fp16=False)
    wt_out = run_model(build_model(wt_nodes, x.shape, fp16=False, name="fc_weight16_rank3"), x)
    assert wt_out.shape == x.shape
    assert_fc_weight16_geometry(wt_out, block_length=16)


if __name__ == "__main__":
    test_fc_input16_keeps_one_row_with_16_consecutive_outputs_fp32()
    test_fc_input16_small_output_count_keeps_one_row_outputs_fp16()
    test_fc_weight16_keeps_16_rows_at_one_output_feature_fp32()
    test_fc_weight16_small_batch_keeps_all_rows_at_one_output_feature_fp16()
    test_fc_masks_support_rank3_outputs_by_flattening_prefix_dims()
    print("FC INPUT16/WEIGHT16 mask tests passed")
