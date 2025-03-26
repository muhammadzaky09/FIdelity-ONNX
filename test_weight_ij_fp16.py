import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnxruntime as ort


import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import os
import struct

def create_fp16_fault_injection_weight(input_name, output_name, bit_position):
    nodes = []
    suffix = "_fp16"
    
    # 1. Create a zero tensor for the output
    nodes.append(helper.make_node(
        'Shape',
        inputs=[input_name],
        outputs=['shape' + suffix]
    ))
    
    nodes.append(helper.make_node(
        'ConstantOfShape',
        inputs=['shape' + suffix],
        outputs=['zero_tensor' + suffix],
        value=helper.make_tensor(
            name='zero_value_tensor' + suffix,
            data_type=TensorProto.FLOAT16,
            dims=[1],
            vals=[0]
        )
    ))
    
    # 2. Generate random values for each dimension
    nodes.append(helper.make_node(
        'RandomUniformLike',
        inputs=['shape' + suffix],
        outputs=['random_vals' + suffix],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0
    ))
    
    # 3. Cast shape to float for multiplication
    nodes.append(helper.make_node(
        'Cast',
        inputs=['shape' + suffix],
        outputs=['shape_float' + suffix],
        to=TensorProto.FLOAT
    ))
    
    # 4. Scale random values to dimension sizes
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals' + suffix, 'shape_float' + suffix],
        outputs=['scaled_random' + suffix]
    ))
    
    # 5. Floor to get integer indices
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_random' + suffix],
        outputs=['floored_indices_float' + suffix]
    ))
    
    # 6. Cast to INT64
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_indices_float' + suffix],
        outputs=['random_indices' + suffix],
        to=TensorProto.INT64
    ))
    
    # 7. Reshape indices for GatherND/ScatterND
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['random_indices' + suffix, 'axes_0' + suffix],
        outputs=['indices' + suffix]
    ))
    
    # 8. Constant for axes parameter
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
    
    # 9. Get the element at the random indices
    nodes.append(helper.make_node(
        'GatherND',
        inputs=[input_name, 'indices' + suffix],
        outputs=['element' + suffix]
    ))
    
    # 10. Create bit position constant
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
    
    # 11. Create a copy of the original element for the faulty version
    nodes.append(helper.make_node(
        'Identity',
        inputs=['element' + suffix],
        outputs=['element_copy' + suffix]
    ))
    
    # 12. Apply BitFlip directly to the copied element
    nodes.append(helper.make_node(
        'BitFlip',
        inputs=['element_copy' + suffix, 'bit_pos' + suffix],
        outputs=['flipped_element' + suffix],
        domain='custom.bitflip'
    ))
    
    # 13. Calculate the perturbation value (flipped - original)
    nodes.append(helper.make_node(
        'Sub',
        inputs=['flipped_element' + suffix, 'element' + suffix],
        outputs=['perturbation_value' + suffix]
    ))
    
    # 14. Put the perturbation value into the zero tensor at the selected position
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=['zero_tensor' + suffix, 'indices' + suffix, 'perturbation_value' + suffix],
        outputs=[output_name]
    ))
    
    return nodes

def create_weight_fault_injection_test_model(bit_position: int):
    """Creates a test model for the weight fault injection function."""
    # Define the model
    input_name = "input_weight"
    output_perturbation = "output_perturbation"
    
    # Input tensor - use 4D weight-like tensor [out_channels, in_channels, kernel_h, kernel_w]
    input_tensor = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT16, [64, 32, 3, 3]
    )
    
    # Output tensor for the perturbation
    output_tensor = helper.make_tensor_value_info(
        output_perturbation, TensorProto.FLOAT16, [64, 32, 3, 3]
    )
    
    # Get the fault injection nodes
    bit_position = np.int32(bit_position)
    fault_nodes = create_fp16_fault_injection_weight(input_name, output_perturbation, bit_position)
    
    # Create the graph
    graph = helper.make_graph(
        fault_nodes,
        'weight_fault_injection_test',
        [input_tensor],
        [output_tensor]
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='bitflip_test',
        opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('custom.bitflip', 1)]
    )
    
    return model

def test_weight_fault_injection():
    # Parameters
    bit_position = 10  # Bit position to flip
    model_file = "weight_fault_injection_test.onnx"
    
    # Create and save the test model
    model = create_weight_fault_injection_test_model(bit_position)
    onnx.save(model, model_file)
    print(f"Test model saved to {model_file}")
    
    # Load the model with custom op
    custom_op_lib_path = "llama/onnx_bitflip.so"  # update with your actual path
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(custom_op_lib_path)
    
    # Create InferenceSession with the custom op registered
    sess = ort.InferenceSession(model_file, sess_options, providers=['CUDAExecutionProvider'])
    
    # Create a test input (4D weight tensor)
    input_weights = np.random.rand(64, 32, 3, 3)
    input_weights = np.float16(input_weights)
    
    # Run inference
    outputs = sess.run(None, {"input_weight": input_weights})
    
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
            
            # Get the ACTUAL original value from the input tensor
            orig_val = input_weights[idx]
            
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
            
            # Check which bits are flipped
            print(f"    Checking which bits were flipped:")
            if bit_diffs_actual == bit_diffs_expected == 1:
                print(f"    ✓ Only one bit was flipped as expected")
            else:
                for bit_idx in range(16):
                    if orig_bits[15-bit_idx] != actual_bits[15-bit_idx]:
                        print(f"    ✗ Bit flipped at position: {bit_idx}")
    
    # Clean up
    os.remove(model_file)
    print("Test completed.")
    
test_weight_fault_injection()