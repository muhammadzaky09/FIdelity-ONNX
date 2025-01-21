import onnx
import onnxruntime as ort
import numpy as np
from inject_ops import *
from onnx import TensorProto, helper
import torch


def test_transformer_tensor_shapes():
    """Test fault injection with transformer tensor shapes."""
    test_cases = [
        # Self-attention input
        {
            'shape': [1, 32, 512],  # [batch, seq_len, hidden_dim]
            'target_indices': [0, 16, 256],  # target in middle of sequence/hidden
            'description': 'self-attention input'
        },
        # # Feed-forward weights
        {
            'shape': [512, 2048],  # [hidden_dim, ff_dim]
            'target_indices': [256, 1024],  # target in middle
            'description': 'feed-forward weights'
        },
        # Layer norm
        {
            'shape': [1, 32, 512],  # [batch, seq_len, hidden_dim]
            'target_indices': [0, 0, 256],  # target at start of sequence
            'description': 'layer norm'
        }
    ]
    
    bit_position = 3
    
    for case in test_cases:
        print(f"\nTesting {case['description']}...")
        
        # Create random int8 input
        input_data = np.random.randint(-128, 127, size=case['shape'], dtype=np.int8)
        
        # Create and run model
        model = create_test_model_transformer(
            case['shape'],
            case['target_indices'],
            bit_position
        )
        
        onnx.checker.check_model(model)
        model_path = f"test_fault_injection_{case['description'].replace(' ', '_')}.onnx"
        onnx.save(model, model_path)
        
        # Run inference
        session = ort.InferenceSession(model_path,providers=['CUDAExecutionProvider'])
        io_binding = session.io_binding()
        io_binding.bind_cpu_input("input", input_data)
        io_binding.bind_output("output_tensor")
        session.run_with_iobinding(io_binding)
        result = io_binding.copy_outputs_to_cpu()[0]
        
        # Get values at target position
        target_idx = tuple(case['target_indices'])
        original_val = input_data[target_idx]
        result_val = result[target_idx]
        print(result.shape)
        mask = 1 << bit_position
        expected_val = original_val ^ mask
        
        print(f"At position {target_idx}:")
        print(f"Original value: {bin(original_val)[2:].zfill(8)} ({original_val})")
        print(f"Mask:          {bin(mask)[2:].zfill(8)} ({mask})")
        print(f"Result:        {bin(result_val)[2:].zfill(8)} ({result_val})")
        print(f"Expected:      {bin(expected_val)[2:].zfill(8)} ({expected_val})")
        
        # Verify bit flip at target
        assert result_val == expected_val, \
            f"Bit flip failed: expected {expected_val}, got {result_val}"
        
        # Create mask for checking unchanged values
        check_mask = np.ones(case['shape'], dtype=bool)
        check_mask[target_idx] = False
        assert np.array_equal(result[check_mask], input_data[check_mask]), \
            "Values changed at untargeted positions"

def create_test_model_transformer(input_shape, target_indices, bit_position):
    """Create ONNX model with transformer tensor shapes."""
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.INT8, input_shape
    )
    
    output_tensor = helper.make_tensor_value_info(
        "output_tensor", TensorProto.INT8, input_shape
    )
    
    nodes = create_int8_fault_injection("input", input_shape, target_indices, bit_position)
    
    graph = helper.make_graph(
        nodes=nodes,
        name="transformer_fault_test",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    
    model = helper.make_model(
        graph,
        producer_name="transformer_fault_test"
    )
    
    return model

if __name__ == "__main__":
    test_transformer_tensor_shapes()
    print("\nAll transformer tensor tests passed!")