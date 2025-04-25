import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import struct

# Define the create_random_bitflip_injection function
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
    
    # 9. Reshape indices for GatherND/ScatterND
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['random_indices' + suffix, 'axes_0' + suffix],
        outputs=['indices' + suffix]
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

# ----- Fixed BitFlip operator for testing purposes -----

def bitflip_fp16(value, bit_position):
    """Flip a specific bit in an FP16 value."""
    # Convert numpy.float16 to raw bytes
    value_bytes = np.array([value], dtype=np.float16).tobytes()
    
    # Unpack bytes to an unsigned 16-bit integer
    as_uint16 = struct.unpack('H', value_bytes)[0]  # 'H' is for unsigned short (16 bits)
    
    # Flip the specific bit
    as_uint16 ^= (1 << bit_position)
    
    # Convert back to bytes and then to FP16
    flipped_bytes = struct.pack('H', as_uint16)
    return np.frombuffer(flipped_bytes, dtype=np.float16)[0]

# ----- Testing Functions -----

def create_custom_op_domain():
    """Create a custom operator domain with the BitFlip operator."""
    # This is for demonstration only - the actual implementation would 
    # require registering the custom op with ONNX Runtime
    bitflip_domain = onnx.OperatorSetIdProto()
    bitflip_domain.domain = "custom.bitflip"
    bitflip_domain.version = 1
    return bitflip_domain

def test_bitflip_without_runtime(shape, bit_position=10):
    """
    Test the bitflip injection in a way that doesn't require the custom op registration.
    Instead of running the ONNX model, we'll show what it would do.
    """
    print(f"\nTesting BitFlip on tensor with shape {shape}")
    
    # Create a tensor with FP16 data
    data = np.ones(shape, dtype=np.float16) * 1.5
    
    # Create ONNX graph inputs and outputs
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT16, shape)
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT16, shape)
    faulty_output = helper.make_tensor_value_info('output_faulty', TensorProto.FLOAT16, shape)
    
    # Create a simple identity op to pass through the input
    identity_node = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['output']
    )
    
    # Create bitflip injection nodes
    fault_nodes = create_random_bitflip_injection('output', bit_position)
    
    # Create the graph
    graph = helper.make_graph(
        [identity_node] + fault_nodes,
        f'test_bitflip_injection_{len(shape)}d',
        [input_tensor],
        [output_tensor, faulty_output]
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 14
    
    # Add the custom domain
    custom_domain = create_custom_op_domain()
    model.opset_import.extend([custom_domain])
    
    # Check the model - may fail due to custom op
    try:
        onnx.checker.check_model(model)
        print("Model validation succeeded")
    except Exception as e:
        print(f"Model validation failed (expected with custom op): {e}")
    
    # Save the model
    model_path = f'test_bitflip_injection_{len(shape)}d.onnx'
    onnx.save(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Since we can't run the model with the custom op,
    # let's demonstrate what would happen by manually performing the operation
    
    # 1. Select a random position in the tensor
    flat_index = np.random.randint(0, np.prod(shape))
    indices = np.unravel_index(flat_index, shape)
    print(f"Random position selected: {indices}")
    
    # 2. Get the value at that position
    original_value = data[indices]
    print(f"Original value: {original_value} (FP16)")
    
    # 3. Apply the bitflip
    flipped_value = bitflip_fp16(original_value, bit_position)
    print(f"Value after flipping bit {bit_position}: {flipped_value} (FP16)")
    
    # 4. Create the modified tensor
    modified_data = data.copy()
    modified_data[indices] = flipped_value
    
    # Show the original value in binary
    original_uint16 = struct.unpack('H', np.array([original_value], dtype=np.float16).tobytes())[0]
    original_binary = bin(original_uint16)[2:].zfill(16)
    
    # Show the flipped value in binary
    flipped_uint16 = struct.unpack('H', np.array([flipped_value], dtype=np.float16).tobytes())[0]
    flipped_binary = bin(flipped_uint16)[2:].zfill(16)
    
    print(f"Original binary: {original_binary}")
    print(f"Flipped binary:  {flipped_binary}")
    
    # Show the bit position that was flipped
    bit_marker = ' ' * (15 - bit_position) + '^'
    print(f"                 {bit_marker}")
    
    return original_value, flipped_value

if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)
    
    # Test with a 3D tensor
    test_bitflip_without_runtime((3, 4, 5), bit_position=14)
    
    # Test with a 4D tensor
    test_bitflip_without_runtime((2, 3, 4, 5), bit_position=10)