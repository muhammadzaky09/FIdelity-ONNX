from onnx import helper, TensorProto
import onnx
import numpy as np
import struct
import onnxruntime as ort
from onnxruntime_extensions import onnx_op, PyCustomOpDef, get_library_path as _get_library_path


def direct_bit_toggle_fp32(value, bit_position):
    as_float32 = np.float32(value)
    bytes_data = np.array(as_float32, dtype=np.float32).tobytes()
    bits = struct.unpack('I', bytes_data)[0]  
    toggled_bits = bits ^ (1 << bit_position)
    bytes_data = struct.pack('I', toggled_bits)
    return float(np.frombuffer(bytes_data, dtype=np.float32)[0])


@onnx_op(op_type="DirectBitToggleFp32",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_int32],
         outputs=[PyCustomOpDef.dt_float])
def direct_bit_toggle_fp32_op(x, bit_position):
    result = np.empty_like(x)
    if bit_position.size == 1:
        bit_pos = bit_position.item() if bit_position.ndim == 0 else bit_position[0]
        single_bit_pos = True
    else:
        single_bit_pos = False
    for idx in np.ndindex(x.shape):
        val = x[idx]
        if single_bit_pos:
            pos = bit_pos
        else:
            pos = bit_position[idx] if idx < bit_position.shape else bit_position[0]
        
        if 0 <= pos < 32:
            bytes_data = np.array(val, dtype=np.float32).tobytes()
            bits = struct.unpack('I', bytes_data)[0]
            toggled_bits = bits ^ (1 << pos)
            toggled_bytes = struct.pack('I', toggled_bits)
            result[idx] = np.frombuffer(toggled_bytes, dtype=np.float32)[0]
        else:
            result[idx] = val
    
    return result

def create_direct_toggle_model():
    input_name = "input_fp32"
    bit_pos_name = "bit_position"
    output_name = "output_fp32"
    
    nodes = []
    
    nodes.append(helper.make_node(
        "DirectBitToggleFp32", 
        [input_name, bit_pos_name], 
        [output_name], 
        domain="ai.onnx.contrib"
    ))
    
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None])
    bit_pos_tensor = helper.make_tensor_value_info(bit_pos_name, TensorProto.INT32, [None])
    output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None])
    
    graph = helper.make_graph(
        nodes, 
        "direct_toggle_fp32", 
        [input_tensor, bit_pos_tensor], 
        [output_tensor]
    )
    
    model = helper.make_model(
        graph,
        producer_name="direct_toggle_fp32_model",
        opset_imports=[
            helper.make_operatorsetid("", 18),
            helper.make_operatorsetid("ai.onnx.contrib", 1)
        ]
    )
    
    model_path = "direct_toggle_fp32.onnx"
    onnx.save(model, model_path)
    print(f"Direct toggle FP32 model saved to {model_path}")
    
    return model_path

# Test function with comprehensive verification
def test_direct_toggle_model(model_path):
    """Test the direct bit toggle model with detailed verification."""
    # Create session with custom op
    so = ort.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
    
    session = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
    
    # Generate test values with special focus on problematic cases
    test_values = [
        0.0, 
        -0.0,
        1.0, 
        -1.0,
        np.float32(np.inf),
        np.float32(-np.inf),
        np.float32(np.nan),
        np.float32(1.175494e-38),   # Min normal positive
        np.float32(-1.175494e-38),  # Min normal negative
        np.float32(1.401298e-45),   # Min subnormal positive
        np.float32(-1.401298e-45),  # Min subnormal negative
        np.float32(3.402823e+38),   # Max normal positive
        np.float32(-3.402823e+38)   # Max normal negative
    ]
    
    def generate_fp16_integer_test_values():
        """
        Generate unique FP16 representations of all integer values from -65000 to 65000.
        Many integers in that range will be rounded when cast to FP16, so we take the unique values.
        """
        values = [float(np.float32(i)) for i in range(-32000, 32000)]
        return sorted(set(values))
    
    # Add some regular values
    test_values.extend([float(i) for i in range(-5, 6)])
    test_values.extend(generate_fp16_integer_test_values())
    
    # Add powers of 2
    for i in range(-5, 6):
        test_values.append(float(2**i))
    
    # Test all bit positions
    bit_positions = list(range(32))
    
    total_tests = 0
    successful_tests = 0
    failed_tests = 0
    failed_details = []
    
    print(f"\nTesting {len(test_values)} values across {len(bit_positions)} bit positions...")
    
    for bit_pos in bit_positions:
        print(f"Testing bit position {bit_pos}...", end="")
        
        # Prepare inputs
        input_values = np.array(test_values, dtype=np.float32)
        bit_position_array = np.array([bit_pos], dtype=np.int32)
        
        # Run inference with ONNX model
        outputs = session.run(["output_fp32"], {"input_fp32": input_values, "bit_position": bit_position_array})
        model_results = outputs[0]
        
        # Compare with ground truth direct implementation
        pos_success = 0
        pos_fail = 0
        
        for i, (input_val, model_result) in enumerate(zip(input_values, model_results)):
            total_tests += 1
            
            # Calculate ground truth using standalone function
            ground_truth = direct_bit_toggle_fp32(input_val, bit_pos)
            
            # Convert both to their bit representation for comparison
            model_bits = struct.unpack('I', np.array(model_result, dtype=np.float32).tobytes())[0]
            truth_bits = struct.unpack('I', np.array(ground_truth, dtype=np.float32).tobytes())[0]
            
            # Check for exact match
            if model_bits == truth_bits:
                successful_tests += 1
                pos_success += 1
            else:
                failed_tests += 1
                pos_fail += 1
                
                # Only record detailed failure info for first few failures
                if len(failed_details) < 10:
                    input_bits = bin(struct.unpack('I', np.array(input_val, dtype=np.float32).tobytes())[0])[2:].zfill(32)
                    result_bits = bin(model_bits)[2:].zfill(32)
                    expected_bits = bin(truth_bits)[2:].zfill(32)
                    
                    failed_details.append({
                        "input_val": float(input_val),
                        "input_bits": input_bits,
                        "model_result": float(model_result),
                        "result_bits": result_bits,
                        "ground_truth": float(ground_truth),
                        "expected_bits": expected_bits,
                        "bit_pos": bit_pos
                    })
        
        # Print bit position success rate
        print(f" {pos_success}/{pos_success + pos_fail} passed ({pos_success/(pos_success + pos_fail)*100:.1f}%)")
    
    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total tests run: {total_tests}")
    print(f"Successful tests: {successful_tests} ({successful_tests/total_tests*100:.2f}%)")
    print(f"Failed tests: {failed_tests} ({failed_tests/total_tests*100:.2f}%)")
    
    # Print detailed failure analysis
    if failed_tests > 0:
        print("\nSample of failed tests:")
        for i, failure in enumerate(failed_details):
            print(f"\nFailure {i+1}:")
            print(f"  Input:    {failure['input_val']} ({failure['input_bits']})")
            print(f"  Result:   {failure['model_result']} ({failure['result_bits']})")
            print(f"  Expected: {failure['ground_truth']} ({failure['expected_bits']})")
            print(f"  Bit pos:  {failure['bit_pos']}")
            
            # Compare bit by bit to identify exact differences
            result_bits = failure['result_bits']
            expected_bits = failure['expected_bits']
            diff_indices = [j for j in range(32) if result_bits[j] != expected_bits[j]]
            
            if diff_indices:
                print(f"  Differing bits at positions: {diff_indices}")
    
    return successful_tests == total_tests

def main():
    # Create model
    model_path = create_direct_toggle_model()
    
    # Test model
    success = test_direct_toggle_model(model_path)
    
    if success:
        print("\nSUCCESS: Direct bit toggle implementation works with 100% bit-exactness!")
    else:
        print("\nWARNING: Some tests failed. Check the details above.")

if __name__ == "__main__":
    main()