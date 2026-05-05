import itertools
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import TensorProto, helper

from graph import modify_onnx_graph


INPUT_WEIGHT_FAULTS = ["INPUT", "WEIGHT", "INPUT16", "WEIGHT16"]
RANDOM_FAULTS = ["RANDOM", "RANDOM_BITFLIP"]


def make_matmul_model(path: Path, precision: str) -> None:
    tensor_type = TensorProto.FLOAT16 if precision == "float16" else TensorProto.FLOAT
    weight_values = np.arange(12, dtype=np.float32).reshape(4, 3)
    if precision == "float16":
        weight_values = weight_values.astype(np.float16)

    x = helper.make_tensor_value_info("X", tensor_type, [2, 4])
    x_id = helper.make_tensor_value_info("X_id", tensor_type, [2, 4])
    y = helper.make_tensor_value_info("Y", tensor_type, [2, 3])
    w = helper.make_tensor("W", tensor_type, [4, 3], weight_values.flatten())
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["X"], ["X_id"], name="input_identity"),
            helper.make_node("MatMul", ["X_id", "W"], ["Y"], name="target_matmul"),
        ],
        "fault_model_matrix",
        [x],
        [y],
        [w],
        value_info=[x_id],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def injected_graph_for(tmp_path, fault_model, precision):
    model_path = tmp_path / f"{fault_model}_{precision}.onnx"
    output_path = tmp_path / f"{fault_model}_{precision}_injected.onnx"
    make_matmul_model(model_path, precision)
    config = {
        "model_name": str(model_path),
        "output_path": str(output_path),
        "target_layer": "target_matmul",
        "input_tensor": "X_id",
        "weight_tensor": "W",
        "layer_type": "MatMul",
    }
    modify_onnx_graph(config, {"precision": precision}, fault_model)
    model = onnx.load(output_path)
    return model, gs.import_onnx(model)


def assert_runtime_inputs(graph, fault_model):
    input_names = {inp.name for inp in graph.inputs}
    assert "rand_idx_inject" in input_names
    if fault_model == "RANDOM":
        assert "bit_pos_inject" not in input_names
    else:
        assert "bit_pos_inject" in input_names


def assert_expected_custom_opset(model, fault_model, precision):
    opsets = {opset.domain for opset in model.opset_import}
    if precision == "float16" and fault_model != "RANDOM":
        assert "custom.bitflip" in opsets
    if precision == "float32" and fault_model == "RANDOM_BITFLIP":
        assert "ai.onnx.contrib" in opsets


def assert_injection_subgraph_present(graph, fault_model):
    tensor_names = set(graph.tensors().keys())
    if fault_model in RANDOM_FAULTS:
        assert any(name.endswith("_faulty") for name in tensor_names)
    else:
        assert any("_fault_injected" in name for name in tensor_names)


def input_weight_cases():
    return itertools.product(INPUT_WEIGHT_FAULTS, ["int8", "int4", "float16"])


def random_cases():
    return itertools.product(RANDOM_FAULTS, ["float32", "float16"])


def test_all_input_weight_fault_models_across_precisions(tmp_path):
    for fault_model, precision in input_weight_cases():
        model, graph = injected_graph_for(tmp_path, fault_model, precision)
        assert_runtime_inputs(graph, fault_model)
        assert_expected_custom_opset(model, fault_model, precision)
        assert_injection_subgraph_present(graph, fault_model)


def test_all_random_fault_models_across_float_precisions(tmp_path):
    for fault_model, precision in random_cases():
        model, graph = injected_graph_for(tmp_path, fault_model, precision)
        assert_runtime_inputs(graph, fault_model)
        assert_expected_custom_opset(model, fault_model, precision)
        assert_injection_subgraph_present(graph, fault_model)
