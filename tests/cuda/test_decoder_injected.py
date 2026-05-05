import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from inject_ops import create_random_bitflip_injection


pytestmark = [pytest.mark.cuda, pytest.mark.slow]

RAND_IDX = 5
BIT_POS = 7


def fp16_bits(value: np.float16) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def make_weight(name, values):
    array = np.asarray(values, dtype=np.float16)
    return helper.make_tensor(name, TensorProto.FLOAT16, list(array.shape), array.flatten())


def build_decoder_like_model(injected: bool):
    hidden_in = helper.make_tensor_value_info("hidden_in", TensorProto.FLOAT16, [1, 4])
    hidden_out = helper.make_tensor_value_info("hidden_out", TensorProto.FLOAT16, [1, 4])

    initializers = [
        make_weight(
            "W_proj",
            [
                [0.5, -1.0, 1.5, 0.25],
                [1.0, 0.75, -0.5, 2.0],
                [-1.5, 1.25, 0.5, -0.75],
                [0.25, -0.5, 1.0, 1.5],
            ],
        ),
        make_weight(
            "W_out",
            [
                [1.0, 0.0, 0.5, -0.5],
                [-0.75, 1.25, 0.0, 0.5],
                [0.25, -1.0, 1.5, 0.75],
                [1.0, 0.5, -0.25, 1.25],
            ],
        ),
        make_weight("Bias", [[0.125, -0.25, 0.375, -0.5]]),
    ]

    nodes = [
        helper.make_node("MatMul", ["hidden_in", "W_proj"], ["proj"], name="proj"),
    ]
    inputs = [hidden_in]
    value_info = [helper.make_tensor_value_info("proj", TensorProto.FLOAT16, [1, 4])]
    opsets = [helper.make_opsetid("", 18)]

    decoder_input = "proj"
    if injected:
        rand_idx = helper.make_tensor_value_info("rand_idx_inject", TensorProto.INT64, [])
        bit_pos = helper.make_tensor_value_info("bit_pos_inject", TensorProto.INT32, [])
        inputs.extend([rand_idx, bit_pos])
        nodes.extend(create_random_bitflip_injection(
            "proj",
            fp16=True,
            rand_idx_name="rand_idx_inject",
            bit_pos_name="bit_pos_inject",
        ))
        decoder_input = "proj_faulty"
        value_info.append(helper.make_tensor_value_info("proj_faulty", TensorProto.FLOAT16, [1, 4]))
        opsets.append(helper.make_opsetid("custom.bitflip", 1))

    nodes.extend([
        helper.make_node("MatMul", [decoder_input, "W_out"], ["decoded"], name="decode"),
        helper.make_node("Add", ["decoded", "Bias"], ["hidden_out"], name="residual_bias"),
    ])

    graph = helper.make_graph(
        nodes,
        "generated_decoder_like_bitflip",
        inputs,
        [hidden_out],
        initializers,
        value_info=value_info,
    )
    return helper.make_model(graph, opset_imports=opsets)


def test_generated_decoder_random_bitflip_changes_output_on_cuda():
    repo_root = Path(__file__).resolve().parents[2]
    bitflip_library = repo_root / "llama" / "onnx_bitflip.so"
    if not bitflip_library.exists():
        pytest.skip(f"custom op library not found: {bitflip_library}")

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session_options.register_custom_ops_library(str(bitflip_library))

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        golden_session = ort.InferenceSession(
            build_decoder_like_model(injected=False).SerializeToString(),
            session_options,
            providers=providers,
        )
        injected_session = ort.InferenceSession(
            build_decoder_like_model(injected=True).SerializeToString(),
            session_options,
            providers=providers,
        )
    except Exception as exc:
        pytest.skip(f"could not create generated decoder sessions: {exc}")

    if "CUDA" not in golden_session.get_providers()[0]:
        pytest.skip("CUDAExecutionProvider is not active")

    hidden = np.array([[0.5, -1.25, 2.0, 0.75]], dtype=np.float16)
    golden = golden_session.run(None, {"hidden_in": hidden})[0]
    injected = injected_session.run(None, {
        "hidden_in": hidden,
        "rand_idx_inject": np.array(RAND_IDX, dtype=np.int64),
        "bit_pos_inject": np.array(BIT_POS, dtype=np.int32),
    })[0]

    assert golden.shape == injected.shape
    assert np.any(golden != injected)

    projection = (hidden @ np.asarray([
        [0.5, -1.0, 1.5, 0.25],
        [1.0, 0.75, -0.5, 2.0],
        [-1.5, 1.25, 0.5, -0.75],
        [0.25, -0.5, 1.0, 1.5],
    ], dtype=np.float16)).astype(np.float16)
    flat_projection = projection.flatten()
    target_idx = RAND_IDX % flat_projection.size
    original_bits = fp16_bits(flat_projection[target_idx])
    expected_faulty_bits = original_bits ^ (1 << BIT_POS)
    assert expected_faulty_bits != original_bits
