"""
Test create_random_bitflip_injection against the actual onnx_bitflip.so.

Expected outcome: session creation fails with a "No such operator" error because
create_random_bitflip_injection emits op='BitFlip' domain='custom.bitflip', but
onnx_bitflip.so only registers op='Perturb' domain='custom.perturb'.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

from inject_ops import create_random_bitflip_injection

SO_PATH = os.path.join(os.path.dirname(__file__), '..', 'onnx-cuda-bitflip', 'build', 'onnx_bitflip.so')

def build_model(rand_idx_name="rand_idx_inject", bit_pos_name="bit_pos_inject"):
    """MatMul(A, B) → Y, then inject RANDOM_BITFLIP at Y."""
    A  = helper.make_tensor_value_info("A",  TensorProto.FLOAT16, [2, 4])
    B  = helper.make_tensor_value_info("B",  TensorProto.FLOAT16, [4, 3])
    Y  = helper.make_tensor_value_info("Y",  TensorProto.FLOAT16, [2, 3])
    ri = helper.make_tensor_value_info(rand_idx_name, TensorProto.INT64, [])
    bp = helper.make_tensor_value_info(bit_pos_name,  TensorProto.INT32, [])

    matmul = helper.make_node("MatMul", ["A", "B"], ["Y"])

    inj_nodes = create_random_bitflip_injection("Y",
                                                fp16=True,
                                                rand_idx_name=rand_idx_name,
                                                bit_pos_name=bit_pos_name)

    Y_faulty = helper.make_tensor_value_info("Y_faulty", TensorProto.FLOAT16, [2, 3])

    graph = helper.make_graph(
        [matmul] + inj_nodes,
        "test_rbf",
        [A, B, ri, bp],
        [Y_faulty],
    )
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 18),
        helper.make_opsetid("custom.bitflip", 1),   # what the function expects
    ])
    onnx.checker.check_model(model, full_check=False)
    return model


def test_create_random_bitflip_injection():
    print("Building ONNX model with create_random_bitflip_injection ...")
    model = build_model()
    model_bytes = model.SerializeToString()

    print(f"Loading custom op library: {SO_PATH}")
    sess_opts = ort.SessionOptions()
    sess_opts.register_custom_ops_library(os.path.realpath(SO_PATH))

    print("Creating inference session ...")
    try:
        sess = ort.InferenceSession(model_bytes, sess_opts,
                                    providers=["CPUExecutionProvider"])
        print("Session created — running inference ...")

        A  = np.random.randn(2, 4).astype(np.float16)
        B  = np.random.randn(4, 3).astype(np.float16)
        ri = np.int64(3)

        Y_orig   = (A @ B)
        Y_faulty = sess.run(None, {"A": A, "B": B,
                                   "rand_idx_inject": ri,
                                   "bit_pos_inject": np.array(7, dtype=np.int32)})[0]

        diff = (Y_faulty != Y_orig)
        changed = np.argwhere(diff.flatten())
        print(f"Changed elements: {changed.flatten().tolist()}")
        assert len(changed) == 1, f"Expected 1 changed element, got {len(changed)}"
        print("PASS: exactly one element changed.")

    except Exception as e:
        print(f"\nFAIL: {type(e).__name__}: {e}")
        print("\nNote: the op is registered as  op='Perturb'  domain='custom.perturb'")
        print("      but the function emits     op='BitFlip'  domain='custom.bitflip'")


if __name__ == "__main__":
    test_create_random_bitflip_injection()
