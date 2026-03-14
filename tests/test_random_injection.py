"""
Test RANDOM fault injection on a tiny MatMul model.
Verifies:
  1. The injected model runs without errors.
  2. The MatMul output is consumed by the injection nodes (not some other tensor).
  3. Exactly one element is changed compared to the clean run.
  4. The changed element is the one requested via rand_idx_inject.
"""
import sys, os, tempfile
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph import modify_onnx_graph

# ── build a tiny float32 MatMul model ──────────────────────────────────────
def build_matmul_model(path):
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 8])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 8])
    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["Y"],
                            name="test_matmul")
    graph = helper.make_graph([node], "g", inputs=[A, B], outputs=[Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path

# ── inject ──────────────────────────────────────────────────────────────────
def inject(src_path):
    out_path = src_path.replace(".onnx", "_random.onnx")
    config = {
        "model_name":    src_path,
        "output_path":   out_path,
        "target_layer":  "test_matmul",
        "input_tensor":  "",
        "weight_tensor": "",
    }
    llama_config = {"precision": "float32"}
    modify_onnx_graph(config, llama_config, "RANDOM", bit_position=0)
    return out_path

# ── helpers ─────────────────────────────────────────────────────────────────
def run_clean(model_path, A, B):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {"A": A, "B": B})[0]

def run_faulty(model_path, A, B, rand_idx):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {
        "A": A, "B": B,
        "rand_idx_inject": np.array(rand_idx, dtype=np.int64),
    })[0]

# ── test ─────────────────────────────────────────────────────────────────────
def test_random():
    np.random.seed(42)
    A = np.random.randn(2, 4).astype(np.float32)
    B = np.random.randn(4, 8).astype(np.float32)
    N = 2 * 8  # total elements in MatMul output

    with tempfile.TemporaryDirectory() as tmp:
        src  = os.path.join(tmp, "matmul.onnx")
        inj  = os.path.join(tmp, "matmul_random.onnx")

        build_matmul_model(src)
        print("[build] clean model saved")

        inject(src)
        print("[inject] injected model saved")

        # Inspect injected model inputs
        m = onnx.load(inj)
        input_names = [inp.name for inp in m.graph.input]
        print(f"[check] injected model inputs: {input_names}")
        assert "rand_idx_inject" in input_names, "rand_idx_inject not in model inputs!"

        clean_out = run_clean(src, A, B)
        print(f"[clean] MatMul output shape: {clean_out.shape}")

        for rand_idx in [0, 5, N-1, 7]:
            faulty_out = run_faulty(inj, A, B, rand_idx)

            diff = (faulty_out.flatten() != clean_out.flatten())
            changed_positions = np.where(diff)[0].tolist()

            print(f"  rand_idx={rand_idx:2d} → changed positions: {changed_positions}")

            assert len(changed_positions) == 1, \
                f"Expected exactly 1 changed element, got {len(changed_positions)}: {changed_positions}"
            assert changed_positions[0] == rand_idx, \
                f"Expected change at {rand_idx}, got {changed_positions[0]}"

        print("\n[PASS] RANDOM fault injection is correct.")

if __name__ == "__main__":
    test_random()
