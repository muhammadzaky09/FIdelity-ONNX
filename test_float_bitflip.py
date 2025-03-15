import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import struct
import os
import math
from tqdm import tqdm

def create_fp16_bit_flip(input_name, output_name, bit_position):

    nodes = []
    suffix = "_bf"
    
    # ------------------------------------------------
    # 1. BASIC CONSTANTS AND INPUT PREPROCESSING
    # ------------------------------------------------
    
    # Cast input to FP16 for precise handling
    nodes.append(helper.make_node(
        "Cast",
        inputs=[input_name],
        outputs=["input_fp16" + suffix],
        to=TensorProto.FLOAT16
    ))
    
    # Cast back to FP32 for calculations
    nodes.append(helper.make_node(
        "Cast",
        inputs=["input_fp16" + suffix],
        outputs=["input_fp32" + suffix],
        to=TensorProto.FLOAT
    ))
    
    # Basic constants needed for all operations
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero" + suffix],
        value=helper.make_tensor(
            name="zero_tensor" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[0.0]
        )
    ))
    
    # Check if input is zero (special handling required)
    nodes.append(helper.make_node(
        "Equal",
        inputs=["input_fp32" + suffix, "zero" + suffix],
        outputs=["is_zero" + suffix]
    ))
    
    # ------------------------------------------------
    # 2. HANDLE ZERO INPUT (SPECIAL CASE)
    # ------------------------------------------------
    
    # For zero input, we need to return exact values for each bit position
    # Precomputed FP16 values for each bit position flipped in zero
    if bit_position == 15:  # Sign bit
        zero_result = -0.0
    elif bit_position >= 10:  # Exponent bits
        # 2^(bit_position-10) is the value when flipping an exponent bit
        # For bit 10: 2^0 = 1, bit 11: 2^1 = 2, etc.
        power = bit_position - 10
        zero_result = float(2.0 ** power)
    else:  # Mantissa bits
        # 2^(bit_position-24) is the value for denormal numbers
        power = bit_position - 24
        zero_result = float(2.0 ** power)
    
    # Create the constant with the correct value for zero input
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero_result" + suffix],
        value=helper.make_tensor(
            name="zero_result_tensor" + suffix,
            data_type=TensorProto.FLOAT16,
            dims=[],
            vals=[zero_result]
        )
    ))
    
    # Cast to FP32 for consistency
    nodes.append(helper.make_node(
        "Cast",
        inputs=["zero_result" + suffix],
        outputs=["zero_result_fp32" + suffix],
        to=TensorProto.FLOAT
    ))
    
    # ------------------------------------------------
    # 3. HANDLE NON-ZERO INPUT BASED ON BIT TYPE
    # ------------------------------------------------
    
    if bit_position == 15:  # SIGN BIT
        # Simply negate the input
        nodes.append(helper.make_node(
            "Neg",
            inputs=["input_fp32" + suffix],
            outputs=["bit_result_raw" + suffix]
        ))
    
    elif bit_position >= 10:  # EXPONENT BITS
        # Get absolute value for exponent manipulation
        nodes.append(helper.make_node(
            "Abs",
            inputs=["input_fp32" + suffix],
            outputs=["abs_value" + suffix]
        ))
        
        # Add small epsilon to avoid log(0)
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["epsilon" + suffix],
            value=helper.make_tensor(
                name="epsilon_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[1e-30]
            )
        ))
        
        nodes.append(helper.make_node(
            "Add",
            inputs=["abs_value" + suffix, "epsilon" + suffix],
            outputs=["safe_abs" + suffix]
        ))
        
        # Calculate log2 for exponent extraction
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["ln2" + suffix],
            value=helper.make_tensor(
                name="ln2_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[0.693147180559945]  # ln(2)
            )
        ))
        
        nodes.append(helper.make_node(
            "Log",
            inputs=["safe_abs" + suffix],
            outputs=["log_value" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Div",
            inputs=["log_value" + suffix, "ln2" + suffix],
            outputs=["log2_value" + suffix]
        ))
        
        # Get exponent and calculate scaling factor
        nodes.append(helper.make_node(
            "Floor",
            inputs=["log2_value" + suffix],
            outputs=["exponent" + suffix]
        ))
        
        # Determine if target bit is set in exponent
        # Exponent in FP16 is biased by 15
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["exponent_bias" + suffix],
            value=helper.make_tensor(
                name="exponent_bias_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[15.0]
            )
        ))
        
        nodes.append(helper.make_node(
            "Add",
            inputs=["exponent" + suffix, "exponent_bias" + suffix],
            outputs=["biased_exponent" + suffix]
        ))
        
        # Calculate bit position in biased exponent
        bit_idx = bit_position - 10  # 0-based index in exponent field
        bit_value = 1 << bit_idx     # Bit mask for this position
        
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["bit_mask" + suffix],
            value=helper.make_tensor(
                name="bit_mask_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[float(bit_value)]
            )
        ))
        
        # Check if bit is set (using integer division and modulo)
        nodes.append(helper.make_node(
            "Div",
            inputs=["biased_exponent" + suffix, "bit_mask" + suffix],
            outputs=["bit_div" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Floor",
            inputs=["bit_div" + suffix],
            outputs=["bit_div_floor" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["two" + suffix],
            value=helper.make_tensor(
                name="two_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[2.0]
            )
        ))
        
        # Use division and subtraction instead of Mod for bit check
        nodes.append(helper.make_node(
            "Div",
            inputs=["bit_div_floor" + suffix, "two" + suffix],
            outputs=["half_div" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Floor",
            inputs=["half_div" + suffix],
            outputs=["half_div_floor" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Mul",
            inputs=["half_div_floor" + suffix, "two" + suffix],
            outputs=["twice_half_floor" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Sub",
            inputs=["bit_div_floor" + suffix, "twice_half_floor" + suffix],
            outputs=["bit_value" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one" + suffix],
            value=helper.make_tensor(
                name="one_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[1.0]
            )
        ))
        
        nodes.append(helper.make_node(
            "Equal",
            inputs=["bit_value" + suffix, "one" + suffix],
            outputs=["is_bit_set" + suffix]
        ))
        
        # Calculate scaling factor for this bit
        scaling_factor = 2.0 ** (2.0 ** bit_idx)
        
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["scale_up" + suffix],
            value=helper.make_tensor(
                name="scale_up_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[scaling_factor]
            )
        ))
        
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["scale_down" + suffix],
            value=helper.make_tensor(
                name="scale_down_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[1.0/scaling_factor]
            )
        ))
        
        # If bit is set, divide by scaling factor; otherwise multiply
        nodes.append(helper.make_node(
            "Where",
            inputs=["is_bit_set" + suffix, "scale_down" + suffix, "scale_up" + suffix],
            outputs=["scaling" + suffix]
        ))
        
        # Apply scaling
        nodes.append(helper.make_node(
            "Mul",
            inputs=["input_fp32" + suffix, "scaling" + suffix],
            outputs=["bit_result_raw" + suffix]
        ))
    
    else:  # MANTISSA BITS (0-9)
        # Get absolute value
        nodes.append(helper.make_node(
            "Abs",
            inputs=["input_fp32" + suffix],
            outputs=["abs_value" + suffix]
        ))
        
        # Check if input is negative
        nodes.append(helper.make_node(
            "Less",
            inputs=["input_fp32" + suffix, "zero" + suffix],
            outputs=["is_negative" + suffix]
        ))
        
        # Add small epsilon to avoid log(0)
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["epsilon" + suffix],
            value=helper.make_tensor(
                name="epsilon_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[1e-30]
            )
        ))
        
        nodes.append(helper.make_node(
            "Add",
            inputs=["abs_value" + suffix, "epsilon" + suffix],
            outputs=["safe_abs" + suffix]
        ))
        
        # Calculate log2 and exponent
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["ln2" + suffix],
            value=helper.make_tensor(
                name="ln2_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[0.693147180559945]  # ln(2)
            )
        ))
        
        nodes.append(helper.make_node(
            "Log",
            inputs=["safe_abs" + suffix],
            outputs=["log_value" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Div",
            inputs=["log_value" + suffix, "ln2" + suffix],
            outputs=["log2_value" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Floor",
            inputs=["log2_value" + suffix],
            outputs=["exponent" + suffix]
        ))
        
        # Calculate 2^exponent
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["two" + suffix],
            value=helper.make_tensor(
                name="two_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[2.0]
            )
        ))
        
        nodes.append(helper.make_node(
            "Pow",
            inputs=["two" + suffix, "exponent" + suffix],
            outputs=["pow2_exp" + suffix]
        ))
        
        # Calculate mantissa weight for this bit
        # Weight = 2^(bit_position-10) * 2^exponent
        mantissa_bit_weight = 2.0 ** (bit_position - 10)
        
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["mantissa_weight" + suffix],
            value=helper.make_tensor(
                name="mantissa_weight_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[mantissa_bit_weight]
            )
        ))
        
        nodes.append(helper.make_node(
            "Mul",
            inputs=["mantissa_weight" + suffix, "pow2_exp" + suffix],
            outputs=["scaled_weight" + suffix]
        ))
        
        # Extract normalized mantissa (0.0-1.0)
        nodes.append(helper.make_node(
            "Div",
            inputs=["abs_value" + suffix, "pow2_exp" + suffix],
            outputs=["normalized_mantissa" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one" + suffix],
            value=helper.make_tensor(
                name="one_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[1.0]
            )
        ))
        
        # Remove implicit 1 to get fractional part
        nodes.append(helper.make_node(
            "Sub",
            inputs=["normalized_mantissa" + suffix, "one" + suffix],
            outputs=["mantissa_frac" + suffix]
        ))
        
        # Scale to [0-1024) range
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["scale_1024" + suffix],
            value=helper.make_tensor(
                name="scale_1024_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[1024.0]
            )
        ))
        
        nodes.append(helper.make_node(
            "Mul",
            inputs=["mantissa_frac" + suffix, "scale_1024" + suffix],
            outputs=["scaled_mantissa" + suffix]
        ))
        
        # Use bit_position directly since mantissa bits are 0-9
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["bit_pos_value" + suffix],
            value=helper.make_tensor(
                name="bit_pos_value_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[float(2 ** bit_position)]
            )
        ))
        
        # Check if the bit is set using division
        nodes.append(helper.make_node(
            "Div",
            inputs=["scaled_mantissa" + suffix, "bit_pos_value" + suffix],
            outputs=["bit_div" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Floor",
            inputs=["bit_div" + suffix],
            outputs=["bit_div_floor" + suffix]
        ))
        
        # Use division and subtraction instead of Mod
        nodes.append(helper.make_node(
            "Div",
            inputs=["bit_div_floor" + suffix, "two" + suffix],
            outputs=["half_div" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Floor",
            inputs=["half_div" + suffix],
            outputs=["half_div_floor" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Mul",
            inputs=["half_div_floor" + suffix, "two" + suffix],
            outputs=["twice_half_floor" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Sub",
            inputs=["bit_div_floor" + suffix, "twice_half_floor" + suffix],
            outputs=["bit_value" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Equal",
            inputs=["bit_value" + suffix, "one" + suffix],
            outputs=["is_bit_set" + suffix]
        ))
        
        # Apply sign to the weight
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["neg_one" + suffix],
            value=helper.make_tensor(
                name="neg_one_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[-1.0]
            )
        ))
        
        nodes.append(helper.make_node(
            "Where",
            inputs=["is_negative" + suffix, "neg_one" + suffix, "one" + suffix],
            outputs=["sign_factor" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Mul",
            inputs=["scaled_weight" + suffix, "sign_factor" + suffix],
            outputs=["signed_weight" + suffix]
        ))
        
        # If bit is set, subtract; if not set, add
        nodes.append(helper.make_node(
            "Neg",
            inputs=["signed_weight" + suffix],
            outputs=["neg_signed_weight" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Where",
            inputs=["is_bit_set" + suffix, "neg_signed_weight" + suffix, "signed_weight" + suffix],
            outputs=["delta" + suffix]
        ))
        
        # Apply delta to original value
        nodes.append(helper.make_node(
            "Add",
            inputs=["input_fp32" + suffix, "delta" + suffix],
            outputs=["bit_result_raw" + suffix]
        ))
    
    # ------------------------------------------------
    # 4. REQUANTIZE RESULT AND COMBINE
    # ------------------------------------------------
    
    # Re-quantize to FP16 precision
    nodes.append(helper.make_node(
        "Cast",
        inputs=["bit_result_raw" + suffix],
        outputs=["bit_result_fp16" + suffix],
        to=TensorProto.FLOAT16
    ))
    
    # Cast back to FP32 for final output
    nodes.append(helper.make_node(
        "Cast",
        inputs=["bit_result_fp16" + suffix],
        outputs=["bit_result" + suffix],
        to=TensorProto.FLOAT
    ))
    
    # Choose between zero case and non-zero case
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_zero" + suffix, "zero_result_fp32" + suffix, "bit_result" + suffix],
        outputs=[output_name]
    ))
    
    return nodes

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
    """Generate a comprehensive set of FP16 values from smallest to largest, excluding infinity."""
    values = []
    
    # Special values
    values.append(0.0)        # Zero
    
    # 1. All exact powers of 2 (about 30 values)
    for exp in range(-24, 16):  # From smallest denormal to largest normal
        value = 2.0 ** exp
        fp16_val = float(np.float16(value))
        if fp16_val != 0.0 and not np.isinf(fp16_val):  # Skip if it underflows to zero or is infinity
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
            if is_round_fp16(fp16_val) and fp16_val != 0.0 and not np.isinf(fp16_val):
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
        if is_round_fp16(fp16_val) and not np.isinf(fp16_val):
            values.append(fp16_val)
            values.append(-fp16_val)

    # Sort, deduplicate, and verify
    unique_values = sorted(set(values))
    verified_values = [v for v in unique_values if is_round_fp16(v) and not np.isinf(v)]
    
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
            for i, failure in enumerate(bit_failures[:]):  # Show first 3 failures per bit
                if 'error' in failure:
                    print(f"  Error for value {failure['value']}: {failure['error']}")
                else:
                    print(f"  Value: {failure['value']}")
                    print(f"    Input:    {failure['input_binary']}")
                    print(f"    Result:   {failure['result_binary']}")
                    print(f"    Expected: {failure['expected_binary']}")

    
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