import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from inject_ops import create_random_bitflip_injection


pytestmark = pytest.mark.cuda

BIT_POS = 7


def fp16_bits(x: np.float16) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def build_model():
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT16, [4, 4])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT16, [4, 8])
    rand_idx = helper.make_tensor_value_info("rand_idx_inject", TensorProto.INT64, [])
    bit_pos = helper.make_tensor_value_info("bit_pos_inject", TensorProto.INT32, [])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [4, 8])
    y_faulty = helper.make_tensor_value_info("Y_faulty", TensorProto.FLOAT16, [4, 8])

    injection_nodes = create_random_bitflip_injection(
        "Y",
        fp16=True,
        rand_idx_name="rand_idx_inject",
        bit_pos_name="bit_pos_inject",
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["A", "B"], ["Y"])] + injection_nodes,
        "test_rbf_cuda",
        [a, b, rand_idx, bit_pos],
        [y_faulty],
        value_info=[y],
    )
    return helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 18),
        helper.make_opsetid("custom.bitflip", 1),
    ])


def test_fp16_random_bitflip_custom_op_runtime_on_cuda():
    repo_root = Path(__file__).resolve().parents[2]
    library_path = repo_root / "llama" / "onnx_bitflip.so"
    if not library_path.exists():
        pytest.skip(f"custom op library not found: {library_path}")

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(str(library_path))

    try:
        session = ort.InferenceSession(
            build_model().SerializeToString(),
            session_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception as exc:
        pytest.skip(f"could not create CUDA custom-op session: {exc}")

    if "CUDA" not in session.get_providers()[0]:
        pytest.skip("CUDAExecutionProvider is not active")

    a = np.random.randn(4, 4).astype(np.float16)
    b = np.random.randn(4, 8).astype(np.float16)
    rand_idx = np.array(3, dtype=np.int64)

    original = (a @ b).astype(np.float16)
    faulty = session.run(None, {
        "A": a,
        "B": b,
        "rand_idx_inject": rand_idx,
        "bit_pos_inject": np.array(BIT_POS, dtype=np.int32),
    })[0]

    flat_original = original.flatten()
    flat_faulty = faulty.flatten()
    changed_idx = np.where(flat_original != flat_faulty)[0]

    assert faulty.shape == original.shape
    assert len(changed_idx) == 1

    original_bits = fp16_bits(flat_original[changed_idx[0]])
    faulty_bits = fp16_bits(flat_faulty[changed_idx[0]])
    xor = original_bits ^ faulty_bits

    assert xor != 0 and ((xor & (xor - 1)) == 0)
    assert xor.bit_length() - 1 == BIT_POS
