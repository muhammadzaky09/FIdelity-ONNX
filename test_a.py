from onnx import helper, TensorProto, numpy_helper
import onnx
import numpy as np
import struct
import onnxruntime as ort
from onnxruntime_extensions import onnx_op, PyCustomOpDef, get_library_path as _get_library_path

# Utility function for direct FP16 bit flipping using struct
def direct_bit_toggle_fp16(value, bit_position):
    """Toggle a specific bit directly using binary operations for FP16."""
    as_float16 = np.float16(value)
    bytes_data = as_float16.tobytes()
    bits = struct.unpack('H', bytes_data)[0]  # 'H' is for unsigned short (2 bytes)
    toggled_bits = bits ^ (1 << bit_position)
    bytes_data = struct.pack('H', toggled_bits)
    return np.frombuffer(bytes_data, dtype=np.float16)[0]

# A completely direct custom op that never goes through FP32
@onnx_op(op_type="DirectByteManipulationFp16BitFlip",
         inputs=[PyCustomOpDef.dt_float16, PyCustomOpDef.dt_int32],
         outputs=[PyCustomOpDef.dt_float16])
def direct_byte_fp16_bit_flip_op(x, bit_position):
    """
    Directly manipulate FP16 bytes to flip specific bits.
    Completely bypasses any conversions that might cause canonicalization.
    
    Args:
        x: Float16 tensor
        bit_position: INT32 tensor with bit positions to flip (0-15)
        
    Returns:
        Float16 tensor with bits flipped exactly
    """
    # Create output array
    result = np.empty_like(x)
    
    # Get the bit position
    if bit_position.size == 1:
        bit_pos = bit_position.item() if bit_position.ndim == 0 else bit_position[0]
        single_bit_pos = True
    else:
        single_bit_pos = False
    
    # Process each value
    for idx in np.ndindex(x.shape):
        # Get input value
        val = x[idx]
        
        # Get bit position for this element
        if single_bit_pos:
            pos = bit_pos
        else:
            pos = bit_position[idx] if idx < bit_position.shape else bit_position[0]
        
        # Check if bit position is valid
        if 0 <= pos < 16:
            # Get the raw bytes
            bytes_data = val.tobytes()
            
            # Convert to uint16
            bits = struct.unpack('H', bytes_data)[0]
            
            # Flip the bit
            toggled_bits = bits ^ (1 << pos)
            
            # Convert back to bytes
            toggled_bytes = struct.pack('H', toggled_bits)
            
            # Convert back to FP16 directly
            result[idx] = np.frombuffer(toggled_bytes, dtype=np.float16)[0]
        else:
            # Invalid bit position, copy input
            result[idx] = val
    
    return result

# Create a model that uses the direct byte manipulation
def create_direct_byte_model():
    """Create a model that directly manipulates FP16 bytes without FP32 conversion."""
    # Define names
    input_name = "input_fp16"
    bit_pos_name = "bit_position"
    output_name = "output_fp16"
    
    nodes = []
    
    # Apply the direct byte manipulation operator
    nodes.append(helper.make_node(
        "DirectByteManipulationFp16BitFlip", 
        [input_name, bit_pos_name], 
        [output_name], 
        domain="ai.onnx.contrib"
    ))
    
    # Define input and output tensors - both FP16
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, [None])
    bit_pos_tensor = helper.make_tensor_value_info(bit_pos_name, TensorProto.INT32, [None])
    output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, [None])
    
    # Create graph and model
    graph = helper.make_graph(
        nodes, 
        "direct_byte_fp16_bitflip", 
        [input_tensor, bit_pos_tensor], 
        [output_tensor]
    )
    
    model = helper.make_model(
        graph,
        producer_name="direct_byte_fp16_bitflip_model",
        opset_imports=[
            helper.make_operatorsetid("", 18),
            helper.make_operatorsetid("ai.onnx.contrib", 1)
        ]
    )
    
    # Save model
    model_path = "direct_byte_fp16_bitflip.onnx"
    onnx.save(model, model_path)
    print(f"Direct byte manipulation model saved to {model_path}")
    
    return model_path

# Test the direct byte manipulation model
def test_direct_byte_model(model_path):
    """Test the direct byte manipulation model."""
    # Create session
    so = ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
    
    try:
        session = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to create session: {e}")
        return False
    
    # Input and output names
    input_name = "input_fp16"
    bit_pos_name = "bit_position"
    output_name = "output_fp16"
    
    # Generate test values focusing on problematic cases
    test_values = [
        # Previously failed values
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
        
        # Regular values
        np.float16(1.0),
        np.float16(-1.0),
        np.float16(3.5),
        np.float16(-3.5)
    ]
    
    # Test the problematic exponent bits
    bit_positions = [10, 11, 12, 13, 14]
    
    print(f"\nTesting {len(test_values)} values across {len(bit_positions)} bit positions...")
    
    total_tests = 0
    successful_tests = 0
    failed_tests = 0
    failed_details = []
    
    for bit_pos in bit_positions:
        print(f"Testing bit position {bit_pos}...")
        
        # Prepare inputs
        input_values = np.array(test_values, dtype=np.float16)
        bit_position_array = np.array([bit_pos], dtype=np.int32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: input_values, bit_pos_name: bit_position_array})
        results = outputs[0]
        
        # Verify results
        for i, (input_val, result) in enumerate(zip(input_values, results)):
            total_tests += 1
            
            # Calculate expected result using the standalone function
            expected = direct_bit_toggle_fp16(input_val, bit_pos)
            
            # Get bit patterns for detailed comparison
            input_uint16 = input_val.view(np.uint16).item()
            result_uint16 = result.view(np.uint16).item()
            expected_uint16 = expected.view(np.uint16).item()
            
            # Check if result matches expected
            if result_uint16 == expected_uint16:
                successful_tests += 1
            else:
                failed_tests += 1
                
                # Format bit patterns for display
                input_bits = bin(input_uint16)[2:].zfill(16)
                result_bits = bin(result_uint16)[2:].zfill(16)
                expected_bits = bin(expected_uint16)[2:].zfill(16)
                
                failed_details.append({
                    "input_val": float(input_val),
                    "input_bits": input_bits,
                    "result_val": float(result),
                    "result_bits": result_bits,
                    "expected_val": float(expected),
                    "expected_bits": expected_bits,
                    "bit_pos": bit_pos
                })
    
    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total tests run: {total_tests}")
    print(f"Successful tests: {successful_tests} ({successful_tests/total_tests*100:.2f}%)")
    print(f"Failed tests: {failed_tests} ({failed_tests/total_tests*100:.2f}%)")
    
    # Print failures
    if failed_tests > 0:
        print("\nSample of failed tests:")
        for i, failure in enumerate(failed_details[:5]):
            print(f"\nFailure {i+1}:")
            print(f"  Input:    {failure['input_val']} ({failure['input_bits']})")
            print(f"  Result:   {failure['result_val']} ({failure['result_bits']})")
            print(f"  Expected: {failure['expected_val']} ({failure['expected_bits']})")
            print(f"  Bit pos:  {failure['bit_pos']}")
    
    return successful_tests == total_tests

def main():
    # Create and test the direct byte manipulation model
    model_path = create_direct_byte_model()
    success = test_direct_byte_model(model_path)
    
    if success:
        print("\nSUCCESS: Direct byte manipulation works with 100% bit-exactness!")
    else:
        print("\nWARNING: Some tests failed. Check the details above.")
        
    # Demonstrate bit-exact operations with a practical example
    demonstrate_practical_example()

def demonstrate_practical_example():
    """Demonstrate how to use this for practical FP16 testing."""
    # Load the model
    so = ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
    session = ort.InferenceSession("direct_byte_fp16_bitflip.onnx", so)
    
    # Create example data
    original = np.array([1.0, 65504.0, -65504.0], dtype=np.float16)
    
    # Create an array of bit positions to toggle
    bit_positions = np.array([10], dtype=np.int32)  # Exponent bit
    
    # Run inference
    results = session.run(["output_fp16"], {"input_fp16": original, "bit_position": bit_positions})[0]
    
    # Print results
    print("\nPRACTICAL EXAMPLE")
    print("================")
    for i, (orig, result) in enumerate(zip(original, results)):
        orig_bits = bin(orig.view(np.uint16).item())[2:].zfill(16)
        result_bits = bin(result.view(np.uint16).item())[2:].zfill(16)
        
        print(f"Original: {orig} ({orig_bits})")
        print(f"After bit flip at position 10: {result} ({result_bits})")
        print("-" * 40)

if __name__ == "__main__":
    main()