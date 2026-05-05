import onnx
import onnx_graphsurgeon as gs

from graph import modify_onnx_graph

def create_simple_test_model():
    """Create a simple ONNX model for testing.

    Graph:  X → Relu → X_relu → MatMul(X_relu, W) → Y

    X_relu is produced by a node, so analyze_paths_gs can trace a path from it
    to test_matmul for the INPUT fault model.
    """
    import onnx.helper as helper
    from onnx import TensorProto
    import numpy as np

    X      = helper.make_tensor_value_info('X',      TensorProto.FLOAT, [1, 10])
    Y      = helper.make_tensor_value_info('Y',      TensorProto.FLOAT, [1, 20])
    X_relu = helper.make_tensor_value_info('X_relu', TensorProto.FLOAT, [1, 10])

    # Weight initializer
    W_init = helper.make_tensor('W', TensorProto.FLOAT, [10, 20],
                                np.random.randn(10, 20).astype(np.float32).flatten())

    relu_node   = helper.make_node('Relu',   inputs=['X'],      outputs=['X_relu'])
    matmul_node = helper.make_node('MatMul', inputs=['X_relu', 'W'], outputs=['Y'],
                                   name='test_matmul')

    graph = helper.make_graph(
        [relu_node, matmul_node],
        'test_model',
        [X],
        [Y],
        [W_init],
    )
    model = helper.make_model(graph)
    model.opset_import[0].version = 18

    return model

def test_graph_surgeon_conversion(tmp_path):
    model = create_simple_test_model()
    model_path = tmp_path / "model.onnx"
    onnx.save(model, model_path)
    
    # Test configurations
    configs = [
        {
            "name": "RANDOM fault model",
            "config": {
                "model_name": model_path,
                "output_path": tmp_path / "random.onnx",
                "target_layer": "test_matmul",
                "input_tensor": "X_relu",
                "weight_tensor": "W",
            },
            "llama_config": {"precision": "float16"},
            "fault_model": "RANDOM",
        },
        {
            "name": "INPUT fault model (int8)",
            "config": {
                "model_name": model_path,
                "output_path": tmp_path / "input.onnx",
                "target_layer": "test_matmul",
                "input_tensor": "X_relu",
                "weight_tensor": "W",
            },
            "llama_config": {"precision": "int8"},
            "fault_model": "INPUT",
        },
    ]

    for test_cfg in configs:
        test_cfg["config"] = {k: str(v) for k, v in test_cfg["config"].items()}
        output_path = modify_onnx_graph(
            test_cfg["config"],
            test_cfg["llama_config"],
            test_cfg["fault_model"],
        )
        graph = gs.import_onnx(onnx.load(output_path))

        if test_cfg["fault_model"] == "RANDOM":
            assert any("_faulty" in name for name in graph.tensors().keys())
        elif test_cfg["fault_model"] == "INPUT":
            assert any("_fault_injected" in name for name in graph.tensors().keys())
            assert any(v.name == "bit_pos_inject" for v in graph.inputs)
            assert any(v.name == "rand_idx_inject" for v in graph.inputs)

if __name__ == "__main__":
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_graph_surgeon_conversion(Path(tmp))
