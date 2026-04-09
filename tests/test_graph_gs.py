#!/usr/bin/env python3
"""Test script to verify GraphSurgeon conversion works correctly."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import modify_onnx_graph
import onnx
import onnx_graphsurgeon as gs
import tempfile
import json

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

def test_graph_surgeon_conversion():
    """Test that the GraphSurgeon conversion maintains functionality."""
    
    print("Creating test model...")
    model = create_simple_test_model()
    
    # Save test model
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        model_path = f.name
        onnx.save(model, model_path)
    
    print(f"Test model saved to: {model_path}")
    
    # Test configurations
    configs = [
        {
            "name": "RANDOM fault model",
            "config": {
                "model_name": model_path,
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
                "target_layer": "test_matmul",
                "input_tensor": "X_relu",
                "weight_tensor": "W",
            },
            "llama_config": {"precision": "int8"},
            "fault_model": "INPUT",
        },
    ]

    # Test each configuration
    for test_cfg in configs:
        print(f"\nTesting {test_cfg['name']}...")

        try:
            output_path = modify_onnx_graph(
                test_cfg["config"],
                test_cfg["llama_config"],
                test_cfg["fault_model"],
            )
            
            # Verify output model
            modified_model = onnx.load(output_path)
            
            # Convert to GraphSurgeon to inspect
            graph = gs.import_onnx(modified_model)
            
            print(f"✓ Successfully created modified model with {len(graph.nodes)} nodes")
            print(f"  Original model had {len(model.graph.node)} nodes")
            
            # Check that injection nodes were added
            if test_cfg["fault_model"] == "RANDOM":
                faulty_found = any("_faulty" in name for name in graph.tensors().keys())
                assert faulty_found, "No faulty output tensor found for RANDOM model"
                print("  ✓ Found faulty output tensor")
            elif test_cfg["fault_model"] == "INPUT":
                injected_found = any("_fault_injected" in name for name in graph.tensors().keys())
                assert injected_found, "No fault_injected tensor found for INPUT model"
                assert any(v.name == "bit_pos_inject" for v in graph.inputs), \
                    "bit_pos_inject not in graph inputs"
                assert any(v.name == "rand_idx_inject" for v in graph.inputs), \
                    "rand_idx_inject not in graph inputs"
                print("  ✓ Found fault_injected tensor and both runtime graph inputs")
            
            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)
                
        except Exception as e:
            print(f"✗ Failed: {e}")
            raise
    
    # Clean up test model
    os.remove(model_path)
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_graph_surgeon_conversion() 