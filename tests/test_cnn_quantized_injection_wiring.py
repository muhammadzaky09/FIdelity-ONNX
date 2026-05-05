import os
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime_extensions import get_library_path
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
    b_init = helper.make_tensor("B", TensorProto.FLOAT, [1], [2.5])

    nodes = [
        helper.make_node("DequantizeLinear", ["X_q", "scale", "zp"], ["X_dq"], name="input_dq"),
        helper.make_node("Conv", ["X_dq", "W", "B"], ["Y"], name="test_conv"),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq_activation_value",
        [x_q],
        [y],
        [
            w_init,
            b_init,
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
        helper.make_node("Conv", ["X", "W_dq", "B"], ["Y"], name="test_conv"),
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
            helper.make_tensor("B", TensorProto.FLOAT, [2], [1.0, -3.0]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def _make_conv_output_consumer_model(model_path: str) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3])
    z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 1, 2, 2])
    w_init = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [1, 1, 2, 2],
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
    bias_init = helper.make_tensor("Bias", TensorProto.FLOAT, [1, 1, 1, 1], [10.0])

    nodes = [
        helper.make_node("Conv", ["X", "W"], ["Y"], name="test_conv"),
        helper.make_node("Add", ["Y", "Bias"], ["Z"], name="consumer_add"),
    ]
    graph = helper.make_graph(nodes, "conv_output_consumer", [x], [z], [w_init, bias_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def _make_gemm_output_consumer_model(model_path: str) -> None:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 3])
    w_init = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [4, 3],
        np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0],
             [7.0, 8.0, 9.0],
             [10.0, 11.0, 12.0]],
            dtype=np.float32,
        ).flatten(),
    )
    c_init = helper.make_tensor("C", TensorProto.FLOAT, [1, 3], [0.5, 1.5, 2.5])

    nodes = [
        helper.make_node("Gemm", ["X", "W"], ["Y"], name="test_gemm"),
        helper.make_node("Add", ["Y", "C"], ["Z"], name="consumer_add"),
    ]
    graph = helper.make_graph(nodes, "gemm_output_consumer", [x], [z], [w_init, c_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def _run_model_with_extensions(model_path: str, feeds: dict[str, np.ndarray]) -> np.ndarray:
    opts = ort.SessionOptions()
    opts.register_custom_ops_library(get_library_path())
    sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)[0]


def _toggle_fp32_bit(value: np.float32, bit_position: int) -> np.float32:
    raw = np.array(value, dtype=np.float32).view(np.uint32)
    toggled = np.uint32(int(raw) ^ (1 << bit_position))
    return np.array(toggled, dtype=np.uint32).view(np.float32)


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
        expected = _conv2d_nchw(x, w) + np.float32(2.5) + _conv2d_nchw(x_faulty - x, w)

        actual = _run_model(output_path, {
            "X_q": x_q,
            "rand_idx_inject": np.array(rand_idx, dtype=np.int64),
            "bit_pos_inject": np.array(bit_position, dtype=np.int32),
        })

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_conv_random_fault_replaces_output_tensor_before_consumers():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "conv_output_consumer.onnx")
        output_path = os.path.join(tmpdir, "conv_random.onnx")
        _make_conv_output_consumer_model(model_path)

        config = parse_target_nodes(model_path, ["Conv"])[0]
        config["output_path"] = output_path
        np.random.seed(7)
        modify_onnx_graph(config, {"precision": "float32"}, "RANDOM")

        x = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        rand_idx = 2
        clean = _run_model(model_path, {"X": x})
        faulty = _run_model(output_path, {
            "X": x,
            "rand_idx_inject": np.array(rand_idx, dtype=np.int64),
        })

        changed = np.flatnonzero(faulty.flatten() != clean.flatten())
        assert changed.tolist() == [rand_idx]
        assert "bit_pos_inject" not in {inp.name for inp in onnx.load(output_path).graph.input}


def test_conv_random_bitflip_replaces_output_tensor_before_consumers():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "conv_output_consumer.onnx")
        output_path = os.path.join(tmpdir, "conv_random_bitflip.onnx")
        _make_conv_output_consumer_model(model_path)

        config = parse_target_nodes(model_path, ["Conv"])[0]
        config["output_path"] = output_path
        modify_onnx_graph(config, {"precision": "float32"}, "RANDOM_BITFLIP")

        x = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        rand_idx = 2
        bit_position = 3
        clean = _run_model(model_path, {"X": x})
        expected = clean.copy()
        expected.flat[rand_idx] = _toggle_fp32_bit(
            np.float32(expected.flat[rand_idx] - np.float32(10.0)),
            bit_position,
        ) + np.float32(10.0)

        actual = _run_model_with_extensions(output_path, {
            "X": x,
            "rand_idx_inject": np.array(rand_idx, dtype=np.int64),
            "bit_pos_inject": np.array(bit_position, dtype=np.int32),
        })

        np.testing.assert_array_equal(actual, expected)


def test_gemm_random_fault_replaces_fc_output_tensor_before_consumers():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "gemm_output_consumer.onnx")
        output_path = os.path.join(tmpdir, "gemm_random.onnx")
        _make_gemm_output_consumer_model(model_path)

        config = parse_target_nodes(model_path, ["Gemm"])[0]
        config["output_path"] = output_path
        np.random.seed(11)
        modify_onnx_graph(config, {"precision": "float32"}, "RANDOM")

        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        rand_idx = 1
        clean = _run_model(model_path, {"X": x})
        faulty = _run_model(output_path, {
            "X": x,
            "rand_idx_inject": np.array(rand_idx, dtype=np.int64),
        })

        changed = np.flatnonzero(faulty.flatten() != clean.flatten())
        assert changed.tolist() == [rand_idx]


def test_gemm_random_bitflip_replaces_fc_output_tensor_before_consumers():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "gemm_output_consumer.onnx")
        output_path = os.path.join(tmpdir, "gemm_random_bitflip.onnx")
        _make_gemm_output_consumer_model(model_path)

        config = parse_target_nodes(model_path, ["Gemm"])[0]
        config["output_path"] = output_path
        modify_onnx_graph(config, {"precision": "float32"}, "RANDOM_BITFLIP")

        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        rand_idx = 1
        bit_position = 4
        clean = _run_model(model_path, {"X": x})
        expected = clean.copy()
        expected.flat[rand_idx] = _toggle_fp32_bit(
            np.float32(expected.flat[rand_idx] - np.float32(1.5)),
            bit_position,
        ) + np.float32(1.5)

        actual = _run_model_with_extensions(output_path, {
            "X": x,
            "rand_idx_inject": np.array(rand_idx, dtype=np.int64),
            "bit_pos_inject": np.array(bit_position, dtype=np.int32),
        })

        np.testing.assert_array_equal(actual, expected)


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
        bias = np.array([1.0, -3.0], dtype=np.float32).reshape(1, 2, 1, 1)
        expected = _conv2d_nchw(x, w) + bias + _conv2d_nchw(x, w_faulty - w)

        actual = _run_model(output_path, {
            "X_raw": x,
            "rand_idx_inject": np.array(rand_idx, dtype=np.int64),
            "bit_pos_inject": np.array(bit_position, dtype=np.int32),
        })

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
