import numpy as np
import onnx
import onnxruntime as ort
import struct
import os
from onnx import helper, TensorProto

def bit_flip_fp16(tensor, bit_position):
    tensor = np.array(tensor, dtype=np.float16, copy=True)
    binary_data = tensor.tobytes()
    as_uint16 = np.frombuffer(binary_data, dtype=np.uint16)
    flipped_uint16 = as_uint16 ^ (1 << bit_position)
    result = np.frombuffer(flipped_uint16.tobytes(), dtype=np.float16)
    return result.reshape(tensor.shape)

def create_test_model_parts(bit_position):
    input_name = "input"
    intermediate_name = "intermediate"
    
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, ["batch", "dim"])
    intermediate_tensor = helper.make_tensor_value_info(intermediate_name, TensorProto.FLOAT16, ["batch", "dim"])
    
    identity_node = helper.make_node(
        "Identity",
        inputs=[input_name],
        outputs=[intermediate_name]
    )
    
    graph1 = helper.make_graph(
        [identity_node],
        f"BitFlipTestModel_Before_{bit_position}",
        [input_tensor],
        [intermediate_tensor]
    )
    
    model1 = helper.make_model(graph1, producer_name="BitFlipTest")
    model1.opset_import[0].version = 13
    
    # Save the first model
    model1_path = f"model_before_{bit_position}.onnx"
    onnx.save(model1, model1_path)
    
    # Create the second model
    output_name = "output"
    
    intermediate_tensor = helper.make_tensor_value_info(intermediate_name, TensorProto.FLOAT16, ["batch", "dim"])
    output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, ["batch", "dim"])
    
    # Simple identity node
    identity_node = helper.make_node(
        "Identity",
        inputs=[intermediate_name],
        outputs=[output_name]
    )
    
    # Create graph and model
    graph2 = helper.make_graph(
        [identity_node],
        f"BitFlipTestModel_After_{bit_position}",
        [intermediate_tensor],
        [output_tensor]
    )
    
    model2 = helper.make_model(graph2, producer_name="BitFlipTest")
    model2.opset_import[0].version = 13
    
    # Save the second model
    model2_path = f"model_after_{bit_position}.onnx"
    onnx.save(model2, model2_path)
    
    return model1_path, model2_path

class BitFlipPipeline:
    def __init__(self, model_before_path, model_after_path, bit_position):
        self.bit_position = bit_position
        self.session_before = ort.InferenceSession(model_before_path)
        self.session_after = ort.InferenceSession(model_after_path)
        
        
        self.input_name = self.session_before.get_inputs()[0].name
        self.intermediate_name = self.session_before.get_outputs()[0].name
        self.final_output_name = self.session_after.get_outputs()[0].name
    
    def run(self, input_data):
        intermediate_output = self.session_before.run(
            [self.intermediate_name], 
            {self.input_name: input_data}
        )[0]
        
        flipped_output = bit_flip_fp16(intermediate_output, self.bit_position)
        
        
        final_output = self.session_after.run(
            [self.final_output_name], 
            {self.intermediate_name: flipped_output}
        )[0]
        
        return final_output

def direct_bit_toggle(value, bit_position):
    as_float16 = np.float16(value)
    bits = struct.unpack('H', np.array(as_float16, dtype=np.float16).tobytes())[0]
    toggled_bits = bits ^ (1 << bit_position)
    bytes_data = struct.pack('H', toggled_bits)
    return float(np.frombuffer(bytes_data, dtype=np.float16)[0])

def float16_to_binary(value):
    as_float16 = np.float16(value)
    bits = struct.unpack('H', np.array(as_float16, dtype=np.float16).tobytes())[0]
    binary = bin(bits)[2:].zfill(16)
    return f"{binary[0]} {binary[1:6]} {binary[6:]}"
def generate_fp16_integer_test_values():
    values = [float(np.float16(i)) for i in range(-65000, 65000)]
    return sorted(set(values))

def test_hybrid_bit_flip(bit_position, test_values=None):
    model_before_path, model_after_path = create_test_model_parts(bit_position)
    
    pipeline = BitFlipPipeline(model_before_path, model_after_path, bit_position)
    
    # Use default test values if none provided
    if test_values is None:
        # Create a comprehensive set of test values
        test_values = []
        
        # Powers of 2
        for exp in range(-14, 16):
            test_values.append(float(np.float16(2.0 ** exp)))
            test_values.append(float(np.float16(-2.0 ** exp)))
        
        # Some normal values with interesting bit patterns
        test_values.extend([
            0.0, -0.0,                               # Zero values
            1.0, -1.0, 3.0, -3.0,                    # Common values
            0.1, -0.1, 0.33, -0.33,                  # Decimal values
            65504.0, -65504.0,                       # Max normal values
            float(np.float16(5.96e-8)),              # Min denormal
            float(np.float16(6.10e-5)),              # All mantissa bits set
            float(np.float16(float('inf'))),         # Infinity
            float(np.float16(float('-inf')))         # Negative infinity
        ])
        test_values.extend(generate_fp16_integer_test_values())
        
        # Remove any duplicates and NaNs
        test_values = [v for v in test_values if not np.isnan(np.float16(v))]
        test_values = sorted(set(test_values))
    
    # Prepare for batch testing
    batch_size = 10  # Process in small batches for efficiency
    results = []
    
    # Process in batches
    for i in range(0, len(test_values), batch_size):
        batch = test_values[i:i+batch_size]
        
        # Create batch input
        input_data = np.array(batch, dtype=np.float16).reshape(-1, 1)
        
        # Run through pipeline
        pipeline_output = pipeline.run(input_data)
        
        # Calculate ground truth for comparison
        direct_outputs = [direct_bit_toggle(val, bit_position) for val in batch]
        direct_output_array = np.array(direct_outputs, dtype=np.float16).reshape(-1, 1)
        
        # Compare results
        for j in range(len(batch)):
            input_val = batch[j]
            hybrid_result = float(pipeline_output[j][0])
            direct_result = float(direct_output_array[j][0])
            
            # Check if binary representations match exactly
            hybrid_bits = struct.unpack('H', np.array(np.float16(hybrid_result), dtype=np.float16).tobytes())[0]
            direct_bits = struct.unpack('H', np.array(np.float16(direct_result), dtype=np.float16).tobytes())[0]
            match = (hybrid_bits == direct_bits)
            
            results.append({
                'value': input_val,
                'hybrid_result': hybrid_result,
                'direct_result': direct_result,
                'input_binary': float16_to_binary(input_val),
                'hybrid_binary': float16_to_binary(np.float16(hybrid_result)),
                'direct_binary': float16_to_binary(np.float16(direct_result)),
                'match': match
            })
    
    # Clean up test models
    try:
        os.remove(model_before_path)
        os.remove(model_after_path)
    except:
        pass
    
    # Print results
    success_count = sum(1 for r in results if r['match'])
    print(f"Testing bit position {bit_position}")
    print("-" * 60)
    print(f"Results: {success_count}/{len(results)} tests passed ({success_count/len(results)*100:.1f}%)")
    
    # Print detailed results (sample)
    print("\nSample of Detailed Results:")
    print("-" * 60)
    for r in results[:5]:  # Show first 5 results
        match_str = "MATCH" if r['match'] else "MISMATCH"
        print(f"Value: {r['value']} ({r['input_binary']})")
        print(f"  Hybrid:  {r['hybrid_result']} ({r['hybrid_binary']})")
        print(f"  Direct:  {r['direct_result']} ({r['direct_binary']})")
        print(f"  Result:  {match_str}")
        print("-" * 40)
    
    # Show any mismatches
    mismatches = [r for r in results if not r['match']]
    if mismatches:
        print(f"\nFound {len(mismatches)} mismatches:")
        for r in mismatches[:3]:  # Show first 3 mismatches
            print(f"Value: {r['value']} ({r['input_binary']})")
            print(f"  Hybrid:  {r['hybrid_result']} ({r['hybrid_binary']})")
            print(f"  Direct:  {r['direct_result']} ({r['direct_binary']})")
            print("-" * 40)
    else:
        print("\nNo mismatches found! 100% accuracy.")
    
    return results

# Run test for all exponent bits
def test_all_exponent_bits():
    """Test all exponent bits (10-14)."""
    for bit_position in range(10, 15):
        results = test_hybrid_bit_flip(bit_position)
        print("\n\n")

# Example usage with more comprehensive test suite
def comprehensive_test(bit_position):
    """Run a more comprehensive test for a specific bit position."""
    # Generate a large set of test values
    test_values = []
    
    # Add all FP16 representable integers from -1000 to 1000
    for i in range(-1000, 1001):
        test_values.append(float(np.float16(i)))
    
    # Add all powers of 2 in FP16 range and nearby values
    for exp in range(-15, 16):
        power = 2.0 ** exp
        test_values.append(float(np.float16(power)))
        test_values.append(float(np.float16(-power)))
        test_values.append(float(np.float16(power * 0.99)))
        test_values.append(float(np.float16(-power * 0.99)))
        test_values.append(float(np.float16(power * 1.01)))
        test_values.append(float(np.float16(-power * 1.01)))
    
    # Add tiny subnormal values
    for i in range(1, 20):
        test_values.append(float(np.float16(i * 5.96e-8)))  # Multiples of min subnormal
    
    # Add values near exponent transition boundaries
    for exp in range(-14, 15):
        boundary = 2.0 ** exp
        test_values.append(float(np.float16(boundary - np.float16(1e-6))))
        test_values.append(float(np.float16(boundary + np.float16(1e-6))))
    
    # Remove duplicates and NaNs
    test_values = [v for v in test_values if not np.isnan(np.float16(v))]
    test_values = sorted(set(test_values))
    
    print(f"Running comprehensive test for bit position {bit_position} with {len(test_values)} values")
    results = test_hybrid_bit_flip(bit_position, test_values)
    return results

if __name__ == "__main__":
    # Run standard tests for all exponent bits
    test_all_exponent_bits()
    
    # Optionally run comprehensive test for a specific bit
    # comprehensive_results = comprehensive_test(10)  # Test bit 10 comprehensively