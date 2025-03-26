import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnxruntime as ort
import struct
import os

def create_fp16_fault_injection(input_name, output_name, bit_position):
    """
    Creates an ONNX model that flips exactly one bit in the tensor.
    
    Args:
        input_name: Input tensor name
        output_name: Output tensor name (with one bit flipped)
        bit_position: Bit position to flip (0-15)
    """
    nodes = []
    suffix = "_fp16"
    
    # 1. Create a copy of the input for the output
    nodes.append(helper.make_node(
        'Identity',
        inputs=[input_name],
        outputs=['tensor_copy' + suffix]
    ))
    
    # 2. Get the shape of the input tensor
    nodes.append(helper.make_node(
        'Shape',
        inputs=[input_name],
        outputs=['shape' + suffix]
    ))
    
    # 3. Generate random values for each dimension
    nodes.append(helper.make_node(
        'RandomUniformLike',
        inputs=['shape' + suffix],
        outputs=['random_vals' + suffix],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0
    ))
    
    # 4. Cast shape to float for multiplication
    nodes.append(helper.make_node(
        'Cast',
        inputs=['shape' + suffix],
        outputs=['shape_float' + suffix],
        to=TensorProto.FLOAT
    ))
    
    # 5. Scale random values to dimension sizes
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals' + suffix, 'shape_float' + suffix],
        outputs=['scaled_random' + suffix]
    ))
    
    # 6. Floor to get integer indices
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_random' + suffix],
        outputs=['floored_indices_float' + suffix]
    ))
    
    # 7. Cast to INT64
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_indices_float' + suffix],
        outputs=['random_indices' + suffix],
        to=TensorProto.INT64
    ))
    
    # 8. Reshape indices for GatherND/ScatterND
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['random_indices' + suffix, 'axes_0' + suffix],
        outputs=['indices' + suffix]
    ))
    
    # 9. Constant for axes parameter
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['axes_0' + suffix],
        value=helper.make_tensor(
            name='axes_0_tensor' + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))
    
    # 10. Get the element at the random indices
    nodes.append(helper.make_node(
        'GatherND',
        inputs=[input_name, 'indices' + suffix],
        outputs=['element' + suffix]
    ))
    
    # 11. Create bit position constant
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['bit_pos' + suffix],
        value=helper.make_tensor(
            name='bit_pos_tensor' + suffix,
            data_type=TensorProto.INT32,
            dims=[1],
            vals=[bit_position]
        )
    ))
    
    # 12. Apply BitFlip to the element
    nodes.append(helper.make_node(
        'BitFlip',
        inputs=['element' + suffix, 'bit_pos' + suffix],
        outputs=['flipped_element' + suffix],
        domain='custom.bitflip'
    ))
    
    # 13. Put the flipped element directly back into the tensor
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=['tensor_copy' + suffix, 'indices' + suffix, 'flipped_element' + suffix],
        outputs=[output_name]
    ))
    
    return nodes

def test_fp16_fault_injection():
    """Tests the create_fp16_fault_injection function to verify it flips only one bit in one element."""
    # Parameters
    input_name = "input_tensor"
    output_name = "output_perturbation"
    bit_position = 14  # Same as previous tests
    model_file = "fp16_fault_injection_test.onnx"
    
    # Create input/output info
    input_tensor = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT16, [64, 32, 3, 3]  # 4D tensor
    )
    
    output_tensor = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT16, [64, 32, 3, 3]
    )
    
    # Get the fault injection nodes
    bit_position = np.int32(bit_position)
    fault_nodes = create_fp16_fault_injection(input_name, output_name, bit_position)
    
    # Create the graph
    graph = helper.make_graph(
        fault_nodes,
        'fp16_fault_injection_test',
        [input_tensor],
        [output_tensor]
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='bitflip_test',
        opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('custom.bitflip', 1)]
    )
    
    # Save the model
    onnx.save(model, model_file)
    print(f"Test model saved to {model_file}")
    
    # Load the model with custom op
    custom_op_lib_path = "llama/onnx_bitflip.so"  # update with your actual path
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(custom_op_lib_path)
    
    # Create InferenceSession
    sess = ort.InferenceSession(model_file, sess_options, providers=['CUDAExecutionProvider'])
    
    # Create a test input with all 1.0 values
    input_data = np.random.rand(64, 32, 3, 3)
    input_data = np.float16(input_data)
    
    # Run inference
    outputs = sess.run(None, {input_name: input_data})
    
    # Get the perturbation
    perturbation = outputs[0]
    
    # Count non-zero elements in perturbation (should be only one)
    non_zero_count = np.count_nonzero(perturbation)
    print(f"Number of non-zero elements in perturbation: {non_zero_count}")
    
    # Get indices of non-zero elements
    non_zero_indices = np.nonzero(perturbation)
    
    # Check each non-zero element
    if non_zero_count > 0:
        print("Non-zero elements in perturbation:")
        for i in range(min(5, non_zero_count)):
            # Extract the index for this non-zero element
            idx = tuple(dim[i] for dim in non_zero_indices)
            perturb_val = perturbation[idx]
            
            # The original value is 1.0
            orig_val = np.float16(1.0)
            
            # The expected flipped value
            expected_flipped = orig_val.view(np.uint16) ^ (1 << bit_position)
            expected_flipped = np.frombuffer(struct.pack('H', expected_flipped), dtype=np.float16)[0]
            
            # The actual flipped value (original + perturbation)
            actual_flipped = orig_val + perturb_val
            
            print(f"  Index {idx}: Perturbation value = {perturb_val}")
            print(f"    Original value: {orig_val}")
            print(f"    Actual flipped value: {actual_flipped}")
            print(f"    Expected flipped value: {expected_flipped}")
            
            # Binary representations
            orig_bits = format(orig_val.view(np.uint16), '016b')
            actual_bits = format(actual_flipped.view(np.uint16), '016b')
            expected_bits = format(expected_flipped.view(np.uint16), '016b')
            
            print(f"    Binary: Original={orig_bits}")
            print(f"    Binary: Actual flipped={actual_bits}")
            print(f"    Binary: Expected flipped={expected_bits}")
            
            # Count bit differences
            bit_diffs_actual = sum(o != a for o, a in zip(orig_bits, actual_bits))
            bit_diffs_expected = sum(o != e for o, e in zip(orig_bits, expected_bits))
            
            print(f"    Number of bit differences (actual): {bit_diffs_actual}")
            print(f"    Number of bit differences (expected): {bit_diffs_expected}")
            
            # Check which bit positions were flipped
            for bit_idx in range(16):
                if orig_bits[15-bit_idx] != actual_bits[15-bit_idx]:
                    print(f"    Bit flipped at position: {bit_idx}")
    
    # Clean up
    os.remove(model_file)
    print("Test completed.")
    
test_fp16_fault_injection()