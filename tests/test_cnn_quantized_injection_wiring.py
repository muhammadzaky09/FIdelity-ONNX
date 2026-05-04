import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graph import modify_onnx_graph
from parser import parse_target_nodes


def _make_quantized_conv_model(model_path: str) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 5, 5])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 3, 3])

    w_init = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [1, 1, 3, 3],
        np.arange(9, dtype=np.float32),
    )
    scale_init = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.25])

    nodes = [
        helper.make_node("Round", ["X"], ["X_q"], name="input_round"),
        helper.make_node("Mul", ["X_q", "scale"], ["X_dq"], name="input_dequant"),
        helper.make_node("Round", ["W"], ["W_q"], name="weight_round"),
        helper.make_node("Mul", ["W_q", "scale"], ["W_dq"], name="weight_dequant"),
        helper.make_node("Conv", ["X_dq", "W_dq"], ["Y"], name="test_conv"),
    ]

    graph = helper.make_graph(
        nodes,
        "quantized_conv",
        [x],
        [y],
        [w_init, scale_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def _single_quantized_injection_input(model_path: str) -> str:
    model = onnx.load(model_path)
    matching = [
        node for node in model.graph.node
        if node.op_type == "Cast" and node.output and node.output[0] == "int_val_inject"
    ]
    assert len(matching) == 1
    return matching[0].input[0]


def _conv2d_nchw(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    n, _, h, width = x.shape
    out_channels, _, kh, kw = w.shape
    out = np.zeros((n, out_channels, h - kh + 1, width - kw + 1), dtype=np.float32)
    for batch in range(n):
        for channel in range(out_channels):
            for y in range(out.shape[2]):
                for x_pos in range(out.shape[3]):
                    window = x[batch, :, y:y + kh, x_pos:x_pos + kw]
                    out[batch, channel, y, x_pos] = np.sum(window * w[channel])
    return out


def _flip_uint8(value: np.uint8, bit_position: int) -> np.uint8:
    return np.uint8(int(value) ^ (1 << bit_position))


def _flip_int8(value: np.int8, bit_position: int) -> np.int8:
    raw = np.array(value, dtype=np.int8).view(np.uint8)
    flipped = np.uint8(int(raw) ^ (1 << bit_position))
    return np.array(flipped, dtype=np.uint8).view(np.int8)


def _run_model(model_path: str, feeds: dict[str, np.ndarray]) -> np.ndarray:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)[0]


def _make_qdq_activation_cast_conv_model(model_path: str) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 2])
    w_init = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [1, 1, 2, 2],
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )

    nodes = [
        helper.make_node("QuantizeLinear", ["X", "scale", "zp"], ["X_quant"], name="input_quant"),
        helper.make_node("Cast", ["X_quant"], ["X_cast"], name="input_cast", to=TensorProto.UINT8),
        helper.make_node("DequantizeLinear", ["X_cast", "scale", "zp"], ["X_dq"], name="input_dq"),
        helper.make_node("Conv", ["X_dq", "W"], ["Y"], name="test_conv"),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq_activation_cast_conv",
        [x],
        [y],
        [
            w_init,
            helper.make_tensor("scale", TensorProto.FLOAT, [], [0.5]),
            helper.make_tensor("zp", TensorProto.UINT8, [], [10]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def _make_qdq_activation_value_model(model_path: str) -> None:
    x_q = helper.make_tensor_value_info("X_q", TensorProto.UINT8, [1, 1, 3, 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 2])
    w_init = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [1, 1, 2, 2],
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )

    nodes = [
        helper.make_node("DequantizeLinear", ["X_q", "scale", "zp"], ["X_dq"], name="input_dq"),
        helper.make_node("Conv", ["X_dq", "W"], ["Y"], name="test_conv"),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq_activation_value",
        [x_q],
        [y],
        [
            w_init,
            helper.make_tensor("scale", TensorProto.FLOAT, [], [0.5]),
            helper.make_tensor("zp", TensorProto.UINT8, [], [10]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def _make_qdq_weight_value_model(model_path: str) -> None:
    x_raw = helper.make_tensor_value_info("X_raw", TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 2, 2])

    w_q = np.array(
        [[[[1, 2], [3, 4]]],
         [[[5, 6], [7, 8]]]],
        dtype=np.int8,
    )
    nodes = [
        helper.make_node("Identity", ["X_raw"], ["X"], name="input_identity"),
        helper.make_node("DequantizeLinear", ["W_q", "w_scale", "w_zp"], ["W_dq"], name="weight_dq", axis=0),
        helper.make_node("Conv", ["X", "W_dq"], ["Y"], name="test_conv"),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq_weight_value",
        [x_raw],
        [y],
        [
            helper.make_tensor("W_q", TensorProto.INT8, list(w_q.shape), w_q.flatten()),
            helper.make_tensor("w_scale", TensorProto.FLOAT, [2], [0.25, 0.5]),
            helper.make_tensor("w_zp", TensorProto.INT8, [2], [0, 0]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def test_parser_resolves_conv_activation_and_weight_to_round_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "quantized_conv.onnx")
        _make_quantized_conv_model(model_path)

        configs = parse_target_nodes(model_path, ["Conv"])

    assert len(configs) == 1
    assert configs[0]["target_layer"] == "test_conv"
    assert configs[0]["layer_type"] == "Conv"
    assert configs[0]["input_tensor"] == "X_q"
    assert configs[0]["weight_tensor"] == "W_q"


def test_conv_input_injection_reads_quantized_activation_tensor():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "quantized_conv.onnx")
        output_path = os.path.join(tmpdir, "input_injected.onnx")
        _make_quantized_conv_model(model_path)

        config = parse_target_nodes(model_path, ["Conv"])[0]
        config["output_path"] = output_path
        modify_onnx_graph(config, {"precision": "int8"}, "INPUT")

        assert _single_quantized_injection_input(output_path) == "X_q"


def test_conv_weight_injection_reads_quantized_weight_tensor():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "quantized_conv.onnx")
        output_path = os.path.join(tmpdir, "weight_injected.onnx")
        _make_quantized_conv_model(model_path)

        config = parse_target_nodes(model_path, ["Conv"])[0]
        config["output_path"] = output_path
        modify_onnx_graph(config, {"precision": "int8"}, "WEIGHT")

        assert _single_quantized_injection_input(output_path) == "W_q"


def test_qdq_conv_input_injection_reads_dequantize_integer_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "qdq_activation_cast.onnx")
        output_path = os.path.join(tmpdir, "input_injected.onnx")
        _make_qdq_activation_cast_conv_model(model_path)

        config = parse_target_nodes(model_path, ["Conv"])[0]
        config["output_path"] = output_path
        assert config["input_tensor"] == "X_dq"
        assert config["weight_tensor"] == "W"
        assert config["layer_type"] == "Conv"

        modify_onnx_graph(config, {"precision": "int8"}, "INPUT")

        assert _single_quantized_injection_input(output_path) == "X_cast"
        onnx.checker.check_model(onnx.load(output_path))


def test_qdq_conv_input_injection_matches_dequantized_delta_reference():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "qdq_activation_value.onnx")
        output_path = os.path.join(tmpdir, "input_injected.onnx")
        _make_qdq_activation_value_model(model_path)

        config = parse_target_nodes(model_path, ["Conv"])[0]
        config["output_path"] = output_path
        modify_onnx_graph(config, {"precision": "int8"}, "INPUT")

        x_q = np.array(
            [[[[10, 11, 12],
               [13, 14, 15],
               [16, 17, 18]]]],
            dtype=np.uint8,
        )
        scale = np.float32(0.5)
        zp = np.uint8(10)
        w = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        rand_idx = 4
        bit_position = 0

        x = (x_q.astype(np.float32) - np.float32(zp)) * scale
        x_faulty_q = x_q.copy()
        x_faulty_q.flat[rand_idx] = _flip_uint8(x_faulty_q.flat[rand_idx], bit_position)
        x_faulty = (x_faulty_q.astype(np.float32) - np.float32(zp)) * scale
        expected = _conv2d_nchw(x, w) + _conv2d_nchw(x_faulty - x, w)

        actual = _run_model(output_path, {
            "X_q": x_q,
            "rand_idx_inject": np.array(rand_idx, dtype=np.int64),
            "bit_pos_inject": np.array(bit_position, dtype=np.int32),
        })

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_qdq_conv_weight_injection_matches_dequantized_delta_reference():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "qdq_weight_value.onnx")
        output_path = os.path.join(tmpdir, "weight_injected.onnx")
        _make_qdq_weight_value_model(model_path)

        config = parse_target_nodes(model_path, ["Conv"])[0]
        config["output_path"] = output_path
        modify_onnx_graph(config, {"precision": "int8"}, "WEIGHT")

        assert _single_quantized_injection_input(output_path) == "W_q"
        onnx.checker.check_model(onnx.load(output_path))

        x = np.array(
            [[[[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0],
               [7.0, 8.0, 9.0]]]],
            dtype=np.float32,
        )
        w_q = np.array(
            [[[[1, 2], [3, 4]]],
             [[[5, 6], [7, 8]]]],
            dtype=np.int8,
        )
        scale = np.array([0.25, 0.5], dtype=np.float32).reshape(2, 1, 1, 1)
        rand_idx = 4
        bit_position = 1

        w = w_q.astype(np.float32) * scale
        w_faulty_q = w_q.copy()
        w_faulty_q.flat[rand_idx] = _flip_int8(w_faulty_q.flat[rand_idx], bit_position)
        w_faulty = w_faulty_q.astype(np.float32) * scale
        expected = _conv2d_nchw(x, w) + _conv2d_nchw(x, w_faulty - w)

        actual = _run_model(output_path, {
            "X_raw": x,
            "rand_idx_inject": np.array(rand_idx, dtype=np.int64),
            "bit_pos_inject": np.array(bit_position, dtype=np.int32),
        })

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
