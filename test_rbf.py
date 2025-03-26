import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, AttributeProto, GraphProto
import os
import struct
# from inject_ops import create_random_bitflip_injection

def create_random_bitflip_injection(output_name: str, bit_position: int):
    nodes = []
    suffix = "_fp16"
    faulty_output = f"{output_name}_faulty"
    
    # 1. Create a copy of the original tensor
    nodes.append(helper.make_node(
        'Identity',
        inputs=[output_name],
        outputs=['tensor_copy' + suffix]
    ))
    
    # 2. Get the shape of the tensor
    nodes.append(helper.make_node(
        'Shape',
        inputs=[output_name],
        outputs=['shape' + suffix]
    ))
    
    # 3. Create random values for indices with shape [1]
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
        inputs=[output_name, 'indices' + suffix],
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
    
    # 13. Update the tensor with the flipped element
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=['tensor_copy' + suffix, 'indices' + suffix, 'flipped_element' + suffix],
        outputs=[faulty_output]
    ))
    
    return nodes

def create_test_model(output_name: str, bit_position: int):
    """Creates a simple test model with a random bit flip injection for 4D tensors."""
    # Create a simple model with an input and output
    input_name = "input"
    output_with_bitflip = f"{output_name}_faulty"
    
    # Create input tensor (4D: batch, channels, height, width)
    input_tensor = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT16, [None, 3, 32, 32]  # 4D tensor shape
    )
    
    # Create output tensors (4D)
    output_tensor = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT16, [None, 3, 32, 32]
    )
    faulty_output_tensor = helper.make_tensor_value_info(
        output_with_bitflip, TensorProto.FLOAT16, [None, 3, 32, 32]
    )
    
    # Create a simple node (Identity in this case)
    identity_node = helper.make_node(
        'Identity',
        inputs=[input_name],
        outputs=[output_name]
    )
    bit_position = np.int32(bit_position)
    # Get the bit flip injection nodes
    bitflip_nodes = create_random_bitflip_injection(output_name, bit_position)
    
    # Combine all nodes
    nodes = [identity_node] + bitflip_nodes
    
    # Create the graph
    graph = helper.make_graph(
        nodes,
        'test_bitflip_model',
        [input_tensor],
        [output_tensor, faulty_output_tensor]
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='bitflip_test',
        opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('custom.bitflip', 1)]
    )
    
    return model

def float16_to_binary(value):
    """Convert a float16 value to its binary representation."""
    # Convert float16 to bytes
    bytes_val = value.tobytes()
    
    # Convert bytes to uint16
    uint16_val = struct.unpack('H', bytes_val)[0]
    
    # Convert to binary string
    binary = bin(uint16_val)[2:].zfill(16)
    
    return binary

def test_bitflip_injection():
    # Parameters
    output_name = "output"
    bit_position = 13  # Bit position to flip (example)
    model_file = "test_bitflip_model.onnx"
    
    # Create and save the test model
    model = create_test_model(output_name, bit_position)
    onnx.save(model, model_file)
    print(f"Test model saved to {model_file}")
    
    # Load the model with custom op
    custom_op_lib_path = "llama/onnx_bitflip.so"  # update with your actual path
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(custom_op_lib_path)
    
    # Create InferenceSession with the custom op registered
    sess = ort.InferenceSession(model_file, sess_options, providers=['CUDAExecutionProvider'])
    
    # Create a test input (4D tensor)
    batch_size = 10
    channels = 3
    height = 32
    width = 32
    input_data = np.random.random((batch_size, channels, height, width)).astype(np.float16)
    
    # Run inference
    outputs = sess.run(None, {"input": input_data})
    
    # Check outputs
    original_output = outputs[0]
    faulty_output = outputs[1]
    
    # Verify that the outputs are different
    different = np.any(original_output != faulty_output)
    print(f"Original and faulty outputs are different: {different}")
    
    # Count the number of different elements
    num_different = np.sum(original_output != faulty_output)
    print(f"Number of different elements: {num_different}")
    
    # Verify that the bit flip caused the difference
    if num_different > 0:
        # Find the indices of the different elements
        diff_indices = np.where(original_output != faulty_output)
        
        # Print some samples of the differences
        print("Sample differences:")
        for i in range(min(5, len(diff_indices[0]))):
            # Get a multi-dimensional index in the 4D tensor
            idx = (diff_indices[0][i], diff_indices[1][i], diff_indices[2][i], diff_indices[3][i])
            orig_val = original_output[idx]
            faulty_val = faulty_output[idx]
            print(f"  Index {idx}: Original={orig_val}, Faulty={faulty_val}")
            
            # Convert to binary and check if exactly one bit is flipped
            orig_bits = float16_to_binary(orig_val)
            faulty_bits = float16_to_binary(faulty_val)
            print(f"    Binary: Original={orig_bits}, Faulty={faulty_bits}")
            
            # Count the number of bit differences
            bit_diffs = sum(o != f for o, f in zip(orig_bits, faulty_bits))
            print(f"    Number of bit differences: {bit_diffs}")
            
            # Check which bit position was flipped
            for bit_idx in range(16):
                if orig_bits[15-bit_idx] != faulty_bits[15-bit_idx]:
                    print(f"    Bit flipped at position: {bit_idx}")
    
    # Clean up
    os.remove(model_file)
    print("Test completed.")

if __name__ == "__main__":
    test_bitflip_injection()