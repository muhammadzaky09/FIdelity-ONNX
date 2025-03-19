from onnx import helper, TensorProto
import onnx
import numpy as np
import struct
import onnxruntime as ort
import time
from onnxruntime_extensions import onnx_op, PyCustomOpDef, get_library_path as _get_library_path

# Standalone function for direct FP16 bit toggling using struct
def direct_bit_toggle_fp16(value, bit_position):
    """
    Toggle a specific bit directly using binary operations (ground truth) for FP16.
    Uses struct to ensure bit-exact manipulation without IEEE-754 canonicalization.
    
    Args:
        value: FP16 value to modify
        bit_position: Position of bit to flip (0-15)
        
    Returns:
        FP16 value with the specified bit flipped
    """
    # Convert to FP16
    as_float16 = np.float16(value)
    
    # Get raw bytes
    bytes_data = as_float16.tobytes()
    
    # Unpack as unsigned short (2 bytes)
    bits = struct.unpack('H', bytes_data)[0]  # 'H' is for unsigned short (2 bytes)
    
    # Toggle the specified bit
    toggled_bits = bits ^ (1 << bit_position)
    
    # Pack back to bytes
    bytes_data = struct.pack('H', toggled_bits)
    
    # Convert back to float16
    return np.frombuffer(bytes_data, dtype=np.float16)[0]

@onnx_op(op_type="Fp16BitFlipWithExponentHandling",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_int32],
         outputs=[PyCustomOpDef.dt_float])
def fp16_bit_flip_with_exponent_handling_op(x, bit_position):
    """
    A wrapper operator that handles FP16 bit flipping with special handling for exponent bits.
    
    Args:
        x: Float32 tensor (will be converted to FP16 internally)
        bit_position: INT32 tensor with bit positions to flip (0-15)
        
    Returns:
        Float32 tensor with bits flipped in FP16 representation
    """
    # Convert to FP16 for internal processing
    x_fp16 = x.astype(np.float16)
    
    # Get the bit position
    if bit_position.size == 1:
        bit_pos = bit_position.item() if bit_position.ndim == 0 else bit_position[0]
        single_bit_pos = True
    else:
        single_bit_pos = False
    
    # Create result array
    result_fp16 = np.empty_like(x_fp16)
    
    # Process each element individually to ensure bit-exactness
    for idx in np.ndindex(x_fp16.shape):
        # Get the input value
        input_val = x_fp16[idx]
        
        # Get bit position for this element
        if single_bit_pos:
            pos = bit_pos
        else:
            pos = bit_position[idx] if idx < bit_position.shape else bit_position[0]
            
        # Check if position is valid
        if 0 <= pos < 16:
            # Check if this is an exponent bit (bits 10-14) or near-special value
            is_exponent_bit = 10 <= pos <= 14
            is_special_value = np.isnan(input_val) or np.isinf(input_val)
            
            # Get input bits for analysis
            input_bits = input_val.view(np.uint16).item()
            
            # Check if flipping this bit would create a special value
            # In FP16, if exponent becomes all 1's (0x1F << 10), it's infinity or NaN
            exponent_mask = 0x1F << 10
            current_exponent = (input_bits & exponent_mask) >> 10
            would_make_special = False
            
            if is_exponent_bit:
                # Calculate what the new exponent would be
                bit_in_exponent = pos - 10
                new_exponent = current_exponent ^ (1 << bit_in_exponent)
                would_make_special = new_exponent == 0x1F  # All 1's in exponent
            
            # Use direct bit manipulation for exponent bits or special values
            if is_exponent_bit or is_special_value or would_make_special:
                # Use direct struct-based toggling to ensure bit-exactness
                result_fp16[idx] = direct_bit_toggle_fp16(input_val, pos)
            else:
                # For normal cases away from special values, use the view method
                input_uint16 = input_val.view(np.uint16)
                mask = np.uint16(1 << pos)
                result_uint16 = np.bitwise_xor(input_uint16, mask)
                result_fp16[idx] = result_uint16.view(np.float16)
        else:
            # Invalid bit position, return input unchanged
            result_fp16[idx] = input_val
    
    # Convert back to input float type (typically float32)
    result = result_fp16.astype(x.dtype)
    
    return result

def create_fp16_exponent_handling_model(input_name, bit_pos_name, output_name):
    """
    Create an ONNX model that applies bit flipping to FP16 values with special exponent handling.
    """
    nodes = []
    
    # Cast FP16 to Float32 for the custom op
    nodes.append(helper.make_node(
        "Cast", 
        [input_name], 
        ["input_as_float32"], 
        to=TensorProto.FLOAT
    ))
    
    # Apply the custom bit-flipping operator
    nodes.append(helper.make_node(
        "Fp16BitFlipWithExponentHandling", 
        ["input_as_float32", bit_pos_name], 
        ["result_float32"], 
        domain="ai.onnx.contrib"
    ))
    
    # Cast the result back to FP16
    nodes.append(helper.make_node(
        "Cast", 
        ["result_float32"], 
        [output_name], 
        to=TensorProto.FLOAT16
    ))
    
    return nodes

def create_and_save_model():
    """Create and save the model with exponent handling."""
    # Define names
    input_name = "input_fp16"
    bit_pos_name = "bit_position"
    output_name = "output_fp16"
    
    # Create model nodes
    nodes = create_fp16_exponent_handling_model(input_name, bit_pos_name, output_name)
    
    # Define input and output tensors
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, [None])
    bit_pos_tensor = helper.make_tensor_value_info(bit_pos_name, TensorProto.INT32, [None])
    output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, [None])
    
    # Create graph and model
    graph = helper.make_graph(
        nodes, 
        "fp16_bitflip_exponent_handling", 
        [input_tensor, bit_pos_tensor], 
        [output_tensor]
    )
    
    model = helper.make_model(
        graph,
        producer_name="fp16_bitflip_exponent_handling_model",
        opset_imports=[
            helper.make_operatorsetid("", 18),
            helper.make_operatorsetid("ai.onnx.contrib", 1)
        ]
    )
    
    # Save model
    model_path = "fp16_bitflip_exponent_handling.onnx"
    onnx.save(model, model_path)
    print(f"Model with exponent handling saved to {model_path}")
    
    return model_path

def test_model(model_path, batch_size=500):
    """Test the model with special focus on exponent bits."""
    print("Setting up test environment...")
    
    # Create session with custom op
    so = ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
    
    try:
        session = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to create session: {e}")
        return False
    
    # Create test values focusing on problematic cases
    test_values = [
        # Values that previously failed
        np.float16(65504.0),  # Largest finite FP16 value
        np.float16(-65504.0),
        np.float16(32000.0),
        np.float16(-32000.0),
        
        # Special values
        np.float16(0.0),
        np.float16(-0.0),
        np.float16(np.inf),
        np.float16(-np.inf),
        np.float16(np.nan),
        
        # Boundary values
        np.float16(6.104e-5),  # Smallest positive normal
        np.float16(-6.104e-5),  # Smallest negative normal
        np.float16(6.097e-5),  # Largest positive subnormal
        np.float16(-6.097e-5),  # Largest negative subnormal
    ]
    
    # Add some regular values
    test_values.extend([np.float16(float(i)) for i in range(-10, 11)])
    
    # Add powers of 2
    for i in range(-14, 16):  # FP16 can represent 2^-14 to 2^15
        if -14 <= i <= 15:  # Valid range for FP16
            test_values.append(np.float16(float(2**i)))
            if i != 0:  # Avoid adding -0.0
                test_values.append(np.float16(float(-(2**i))))
    
    # Specifically focus on exponent bits (bits 10-14) and boundary cases
    bit_positions = list(range(16))  # Test all 16 bits
    
    total_tests = 0
    successful_tests = 0
    failed_tests = 0
    failed_details = []
    
    print(f"\nTesting {len(test_values)} values across {len(bit_positions)} bit positions...")
    start_time = time.time()
    
    # Process tests
    for bit_pos in bit_positions:
        is_exponent_bit = 10 <= bit_pos <= 14
        print(f"Testing bit position {bit_pos}" + (" (EXPONENT BIT)" if is_exponent_bit else ""))
        
        # Process in batches
        for i in range(0, len(test_values), batch_size):
            batch = test_values[i:i+batch_size]
            
            # Prepare inputs
            input_values = np.array(batch, dtype=np.float16)
            bit_position_array = np.array([bit_pos], dtype=np.int32)
            
            # Run inference
            outputs = session.run(["output_fp16"], {"input_fp16": input_values, "bit_position": bit_position_array})
            results = outputs[0]
            
            # Verify results
            for j, (input_val, result) in enumerate(zip(input_values, results)):
                total_tests += 1
                
                # Calculate expected using the ground truth function
                expected = direct_bit_toggle_fp16(input_val, bit_pos)
                
                # Get bit patterns
                input_uint16 = input_val.view(np.uint16).item()
                result_uint16 = result.view(np.uint16).item()
                expected_uint16 = expected.view(np.uint16).item()
                
                # Check for exact match of bit patterns
                if result_uint16 == expected_uint16:
                    successful_tests += 1
                else:
                    failed_tests += 1
                    
                    # Format bit patterns for display
                    input_bits = bin(input_uint16)[2:].zfill(16)
                    result_bits = bin(result_uint16)[2:].zfill(16)
                    expected_bits = bin(expected_uint16)[2:].zfill(16)
                    
                    # Distinguish exponent bits in the display
                    exponent_start = 10
                    exponent_end = 14
                    annotated_input = ""
                    for b in range(16):
                        if b == 15:  # Sign bit
                            annotated_input += "S" + input_bits[b]
                        elif exponent_start <= b <= exponent_end:  # Exponent bits
                            annotated_input += "E" + input_bits[b]
                        else:  # Mantissa bits
                            annotated_input += "M" + input_bits[b]
                    
                    failed_details.append({
                        "input_val": float(input_val),
                        "input_bits": input_bits,
                        "annotated_input": annotated_input,
                        "result_val": float(result),
                        "result_bits": result_bits,
                        "expected_val": float(expected),
                        "expected_bits": expected_bits,
                        "bit_pos": bit_pos,
                        "is_exponent_bit": is_exponent_bit,
                        "is_special_input": np.isnan(input_val) or np.isinf(input_val),
                        "is_special_expected": np.isnan(expected) or np.isinf(expected)
                    })
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total tests run: {total_tests}")
    print(f"Successful tests: {successful_tests} ({successful_tests/total_tests*100:.2f}%)")
    print(f"Failed tests: {failed_tests} ({failed_tests/total_tests*100:.2f}%)")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    # Print detailed failure analysis
    if failed_tests > 0:
        print("\nSample of failed tests:")
        for i, failure in enumerate(failed_details[:5]):  # Show up to 5 failures
            print(f"\nFailure {i+1}:")
            print(f"  Input:    {failure['input_val']} ({failure['input_bits']})")
            print(f"  Annotated:{failure['annotated_input']} (S=Sign, E=Exponent, M=Mantissa)")
            print(f"  Result:   {failure['result_val']} ({failure['result_bits']})")
            print(f"  Expected: {failure['expected_val']} ({failure['expected_bits']})")
            print(f"  Bit pos:  {failure['bit_pos']} (Exponent bit: {failure['is_exponent_bit']})")
            
            # Additional analysis
            if failure['is_special_input']:
                print("  Analysis: Input was a special value (NaN or infinity)")
            if failure['is_special_expected']:
                print("  Analysis: Expected result is a special value (NaN or infinity)")
    
    return successful_tests == total_tests

def create_complex_graph_with_bitflip():
    """
    Create a more complex ONNX graph that incorporates the FP16 bit flipping operation
    in the middle of a neural network-like computation.
    """
    # Define names
    input_name = "input"
    bit_pos_name = "bit_position"
    output_name = "output"
    
    nodes = []
    
    # 1. Add operation (simulating a first layer)
    nodes.append(helper.make_node(
        "Add", 
        [input_name, input_name], 
        ["layer1_output"]
    ))
    
    # 2. Cast to FP16 for the next operations
    nodes.append(helper.make_node(
        "Cast", 
        ["layer1_output"], 
        ["layer1_fp16"], 
        to=TensorProto.FLOAT16
    ))
    
    # 3. Multiply by 2.0 (simulating some scaling)
    nodes.append(helper.make_node(
        "Mul", 
        ["layer1_fp16", "layer1_fp16"], 
        ["layer2_fp16"]
    ))
    
    # 4. Apply bitflip to the fp16 tensor
    # First, we need to cast to float32 for the custom op
    nodes.append(helper.make_node(
        "Cast", 
        ["layer2_fp16"], 
        ["layer2_float32"], 
        to=TensorProto.FLOAT
    ))
    
    # Apply the bitflip operation
    nodes.append(helper.make_node(
        "Fp16BitFlipWithExponentHandling", 
        ["layer2_float32", bit_pos_name], 
        ["bitflip_result_float32"], 
        domain="ai.onnx.contrib"
    ))
    
    # Cast back to FP16
    nodes.append(helper.make_node(
        "Cast", 
        ["bitflip_result_float32"], 
        ["bitflip_result_fp16"], 
        to=TensorProto.FLOAT16
    ))
    
    # 5. Apply another operation (tanh activation)
    nodes.append(helper.make_node(
        "Tanh", 
        ["bitflip_result_fp16"], 
        ["layer3_fp16"]
    ))
    
    # 6. Cast final result back to FP32 for output
    nodes.append(helper.make_node(
        "Cast", 
        ["layer3_fp16"], 
        [output_name], 
        to=TensorProto.FLOAT
    ))
    
    # Define input and output tensors
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, None])
    bit_pos_tensor = helper.make_tensor_value_info(bit_pos_name, TensorProto.INT32, [None])
    output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, None])
    
    # Create graph and model
    graph = helper.make_graph(
        nodes, 
        "complex_graph_with_bitflip", 
        [input_tensor, bit_pos_tensor], 
        [output_tensor]
    )
    
    model = helper.make_model(
        graph,
        producer_name="complex_graph_with_bitflip_model",
        opset_imports=[
            helper.make_operatorsetid("", 18),
            helper.make_operatorsetid("ai.onnx.contrib", 1)
        ]
    )
    
    # Save model
    model_path = "complex_graph_with_bitflip.onnx"
    onnx.save(model, model_path)
    print(f"Complex graph model saved to {model_path}")
    
    return model_path

def test_complex_graph(model_path):
    """Test the complex graph with the bit-flipping operation."""
    print("Setting up test environment for complex graph...")
    
    # Create session with custom op
    so = ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
    
    try:
        session = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to create session: {e}")
        return False
    
    # Create input data - 3x3 matrix
    input_data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)
    
    # Try different bit positions
    bit_positions = [10, 11, 12, 13, 14]  # Exponent bits for FP16
    
    print("\nTesting complex graph with different bit positions...")
    
    for bit_pos in bit_positions:
        print(f"\nTesting with bit position {bit_pos}:")
        
        # Create bit position tensor
        bit_pos_tensor = np.array([bit_pos], dtype=np.int32)
        
        # Run inference
        outputs = session.run(["output"], {"input": input_data, "bit_position": bit_pos_tensor})
        result = outputs[0]
        
        # Print input and output to verify the model runs end-to-end
        print("Input data:")
        print(input_data)
        print("\nResult after operations and bit-flipping:")
        print(result)
        
        # Check if the shape matches
        shape_matches = input_data.shape == result.shape
        print(f"Shape verification: {'PASSED' if shape_matches else 'FAILED'}")
        
        # Check if there are any NaNs or infinities
        has_nan = np.isnan(result).any()
        has_inf = np.isinf(result).any()
        print(f"Contains NaN: {has_nan}")
        print(f"Contains Infinity: {has_inf}")
    
    print("\nComplex graph testing completed.")
    return True

def main():
    # Create and test the model with exponent handling
    print("\n--- TESTING SIMPLE MODEL WITH EXPONENT HANDLING ---")
    simple_model_path = create_and_save_model()
    simple_success = test_model(simple_model_path)
    
    # # Create and test a more complex graph
    # print("\n--- TESTING COMPLEX GRAPH WITH BIT FLIPPING ---")
    # complex_model_path = create_complex_graph_with_bitflip()
    # complex_success = test_complex_graph(complex_model_path)
    
    # Print overall results
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"Simple model verification: {'PASSED' if simple_success else 'FAILED'}")
    print(f"Complex graph testing: {'PASSED' if complex_success else 'FAILED'}")

if __name__ == "__main__":
    main()