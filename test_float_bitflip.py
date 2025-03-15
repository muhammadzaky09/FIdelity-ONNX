import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import struct
import os
import math
from tqdm import tqdm
from inject_ops import create_fp16_bit_flip  # assuming this is your implementation

def float16_to_binary(value):
    """Convert a float16 value to its binary representation for display."""
    as_float16 = np.float16(value)
    bits = struct.unpack('H', np.array(as_float16, dtype=np.float16).tobytes())[0]
    binary = bin(bits)[2:].zfill(16)
    # Format as sign | exponent | mantissa
    return f"{binary[0]} {binary[1:6]} {binary[6:]}"

def direct_bit_toggle(value, bit_position):
    """Toggle a specific bit directly (ground truth)."""
    as_float16 = np.float16(value)
    bits = struct.unpack('H', np.array(as_float16, dtype=np.float16).tobytes())[0]
    toggled_bits = bits ^ (1 << bit_position)
    bytes_data = struct.pack('H', toggled_bits)
    return float(np.frombuffer(bytes_data, dtype=np.float16)[0])

def create_simple_bitflip_model(bit_position):
    """Create a simple model that just performs bit flipping at the specified position."""
    input_name = 'input'
    output_name = 'output'
    
    # Create model inputs and outputs
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1])
    output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1])
    
    # Add the bit flip operations directly
    bit_flip_nodes = create_fp16_bit_flip(input_name, output_name, bit_position)
    
    # Create graph and model
    graph = helper.make_graph(
        bit_flip_nodes,
        'test_bit_flip',
        [input_tensor],
        [output_tensor]
    )
    
    model = helper.make_model(graph, producer_name='bit_flip_test')
    model.opset_import[0].version = 17
    
    return model

def test_fp16_bit_flip(value, bit_position):
    """Test the FP16 bit flip function on a specific value and bit position."""
    # Create the model
    model = create_simple_bitflip_model(bit_position)
    
    # Save model temporarily
    model_path = f"bit_flip_test_{bit_position}.onnx"
    onnx.save(model, model_path)
    
    try:
        # Run inference
        session = ort.InferenceSession(model_path)
        
        # Convert to FP32 but with FP16 precision
        input_fp16 = np.float16(value)
        input_data = np.array([float(input_fp16)], dtype=np.float32)
        
        outputs = session.run(['output'], {'input': input_data})
        result = float(outputs[0][0])
        
        # Calculate expected value (ground truth)
        expected = direct_bit_toggle(value, bit_position)
        
        # Use binary representation for exact bit comparison
        result_fp16 = np.float16(result)
        expected_fp16 = np.float16(expected)
        
        result_bits = struct.unpack('H', np.array(result_fp16, dtype=np.float16).tobytes())[0]
        expected_bits = struct.unpack('H', np.array(expected_fp16, dtype=np.float16).tobytes())[0]
        
        # Exact binary comparison
        is_match = (result_bits == expected_bits)
        
        return {
            'value': value,
            'bit_pos': bit_position,
            'input_binary': float16_to_binary(value),
            'result': float(result_fp16),
            'result_binary': float16_to_binary(result_fp16),
            'expected': expected,
            'expected_binary': float16_to_binary(expected_fp16),
            'match': is_match
        }
        
    except Exception as e:
        print(f"Error testing bit flip at position {bit_position} for value {value}:")
        print(f"  {str(e)}")
        return {
            'value': value,
            'bit_pos': bit_position,
            'error': str(e),
            'match': False
        }
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)

def is_round_fp16(value):
    """Check if a value has a 'round' representation in FP16 (mantissa is all zeros)."""
    # Convert to FP16 and get bits
    as_float16 = np.float16(value)
    bits = struct.unpack('H', np.array(as_float16, dtype=np.float16).tobytes())[0]
    
    # Extract mantissa (lower 10 bits)
    mantissa = bits & 0x3FF
    
    # Check if mantissa is zero
    return mantissa == 0

def generate_all_round_fp16_values():
    """Generate ALL possible 'round' FP16 values across the entire range."""
    values = []
    
    # Special values
    values.append(0.0)        # Zero
    values.append(-0.0)       # Negative zero
    
    # 1. Powers of 2 - these are guaranteed to have mantissa=0
    # Unbiased exponent range for normalized FP16 numbers is -14 to 15
    for exp in range(-14, 16):
        value = 2.0 ** exp
        values.append(value)
        values.append(-value)
    
    # 2. Powers of 10 that can be represented
    for exp in range(-4, 5):  # -10000 to 10000
        value = 10.0 ** exp
        fp16_value = float(np.float16(value))
        if is_round_fp16(fp16_value):
            values.append(fp16_value)
            values.append(-fp16_value)
    
    # 3. Other common "round" numbers in decimal
    for base in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        for exp in range(0, 5):  # 1-90000
            value = base * (10.0 ** exp)
            fp16_value = float(np.float16(value))
            if is_round_fp16(fp16_value) and fp16_value <= 65504.0:
                values.append(fp16_value)
                values.append(-fp16_value)
    
    # 4. Values near FP16 limits
    for value in [65000, 65500, 65504]:  # Max normal is 65504
        fp16_value = float(np.float16(value))
        if is_round_fp16(fp16_value):
            values.append(fp16_value)
            values.append(-fp16_value)
    
    # 5. Denormal numbers (exponent=0, mantissa≠0)
    min_normal = 6.103515625e-5  # 2^-14
    min_denormal = 5.960464477539063e-8  # 2^-24
    
    # Add a few denormal values that are powers of 2
    exp = -15
    while True:
        value = 2.0 ** exp
        fp16_value = float(np.float16(value))
        if fp16_value == 0:
            break  # Too small to represent
        if is_round_fp16(fp16_value):
            values.append(fp16_value)
            values.append(-fp16_value)
        exp -= 1
    
    # Sort and remove duplicates
    unique_values = sorted(set(values))
    
    # Verify these are truly "round" in FP16
    verified_values = [v for v in unique_values if is_round_fp16(v)]
    
    return verified_values

def generate_all_exact_fp16_values():
    """Generate a comprehensive set of FP16 values from smallest to largest."""
    values = []
    
    # Special values
    values.append(0.0)        # Zero
    
    # 1. All exact powers of 2 (about 30 values)
    for exp in range(-24, 16):  # From smallest denormal to largest normal
        value = 2.0 ** exp
        fp16_val = float(np.float16(value))
        if fp16_val != 0.0:  # Skip if it underflows to zero
            values.append(fp16_val)
            values.append(-fp16_val)
    
    # 2. Around key boundaries
    boundaries = [
        5.96e-8,   # Min denormal
        6.10e-5,   # Min normal
        1.0,       # Unity
        65504.0    # Max normal
    ]
    
    for boundary in boundaries:
        # Test values around each boundary
        for factor in [0.9, 0.99, 0.999, 1.0, 1.001, 1.01, 1.1]:
            value = boundary * factor
            fp16_val = float(np.float16(value))
            if is_round_fp16(fp16_val) and fp16_val != 0.0:
                values.append(fp16_val)
                values.append(-fp16_val)
    
    # 3. Common integer values
    for i in range(1, 1001):  # 1-1000
        if is_round_fp16(float(i)):
            values.append(float(i))
            values.append(-float(i))
    
    # 4. Powers of 10
    for exp in range(-4, 5):
        value = 10.0 ** exp
        fp16_val = float(np.float16(value))
        if is_round_fp16(fp16_val):
            values.append(fp16_val)
            values.append(-fp16_val)
    
    # Sort, deduplicate, and verify
    unique_values = sorted(set(values))
    verified_values = [v for v in unique_values if is_round_fp16(v)]
    
    return verified_values

def test_fp16_comprehensive():
    """Test bit flipping across all round FP16 values and all bit positions."""
    values = generate_all_exact_fp16_values()
    
    # Test ALL bit positions 0-15
    test_bits = list(range(16))
    
    results = []
    failures = []
    
    total_tests = len(values) * len(test_bits)
    print(f"Testing {len(values)} round FP16 values across all 16 bit positions ({total_tests} tests)")
    print(f"Round values range: {min(values)} to {max(values)}")
    
    # Progress bar for the tests
    for value in tqdm(values, desc="Testing values"):
        # For each value, test flipping all 16 bits
        for bit_pos in test_bits:
            result = test_fp16_bit_flip(value, bit_pos)
            results.append(result)
            if not result.get('match', False):
                failures.append(result)
    
    # Count and report successes
    success_count = sum(1 for r in results if r.get('match', False))
    print(f"\nFinal results: {success_count}/{len(results)} tests passed ({success_count/len(results)*100:.1f}%)")
    
    # Report failures - group by bit position for clarity
    if failures:
        failures_by_bit = {}
        for r in failures:
            bit = r['bit_pos']
            if bit not in failures_by_bit:
                failures_by_bit[bit] = []
            failures_by_bit[bit].append(r)
        
        print("\nFailures by bit position:")
        for bit, bit_failures in failures_by_bit.items():
            print(f"\nBit position {bit}: {len(bit_failures)} failures")
            for i, failure in enumerate(bit_failures[:3]):  # Show first 3 failures per bit
                if 'error' in failure:
                    print(f"  Error for value {failure['value']}: {failure['error']}")
                else:
                    print(f"  Value: {failure['value']}")
                    print(f"    Input:    {failure['input_binary']}")
                    print(f"    Result:   {failure['result_binary']}")
                    print(f"    Expected: {failure['expected_binary']}")
            if len(bit_failures) > 3:
                print(f"  ...and {len(bit_failures) - 3} more failures for bit {bit}")
    
    # Save detailed results to file
    with open("fp16_comprehensive_results.txt", "w") as f:
        f.write("value,bit_pos,input_binary,result,result_binary,expected,expected_binary,match\n")
        for r in results:
            if 'error' in r:
                f.write(f"{r['value']},{r['bit_pos']},ERROR,ERROR,ERROR,ERROR,ERROR,0\n")
            else:
                f.write(f"{r['value']},{r['bit_pos']},{r['input_binary']},{r['result']},{r['result_binary']},{r['expected']},{r['expected_binary']},{1 if r['match'] else 0}\n")
    
    print("\nDetailed results saved to fp16_comprehensive_results.txt")

if __name__ == "__main__":
    test_fp16_comprehensive()