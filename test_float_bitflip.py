import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto

def create_float_bit_flip_simulation(input_name, output_name, bit_position):
    """
    Simulates flipping a specific bit in a float16 value using only floating-point operations.
    Consistently uses float16 throughout for compatibility.
    
    Args:
        input_name: Name of input tensor (must be float16)
        output_name: Name of output tensor
        bit_position: Position of bit to flip (0-15 for float16)
    
    Returns:
        List of ONNX nodes implementing the operation
    """
    nodes = []
    suffix = "_bitflip"
    
    # IEEE-754 float16 has:
    # - 1 sign bit (bit 15)
    # - 5 exponent bits (bits 10-14)
    # - 10 mantissa bits (bits 0-9)
    
    # 0. Add constant for value 2.0 as float16
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["two_constant" + suffix],
        value=helper.make_tensor(
            name="two_constant_tensor" + suffix,
            data_type=TensorProto.FLOAT16,  # Use float16 instead of float
            dims=[],
            vals=[np.float16(2.0).view(np.uint16).item()]  # Properly encode float16
        )
    ))
    
    # 1. Create bit position constants (no change needed - these are integers)
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["bit_pos" + suffix],
        value=helper.make_tensor(
            name="bit_pos_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],
            vals=[bit_position]
        )
    ))
    
    # 2. Thresholds for determining bit type (no change needed - these are integers)
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["sign_threshold" + suffix],
        value=helper.make_tensor(
            name="sign_threshold_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],
            vals=[15] 
        )
    ))
    
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["exponent_threshold" + suffix],
        value=helper.make_tensor(
            name="exponent_threshold_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],
            vals=[10]
        )
    ))
    
    # 3-5. Logic for determining bit type (no changes needed - these use boolean operators)
    # (Check if it's sign bit, exponent bit, or mantissa bit)
    nodes.append(helper.make_node(
        "Equal",
        inputs=["bit_pos" + suffix, "sign_threshold" + suffix],
        outputs=["is_sign_bit" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Greater",
        inputs=["bit_pos" + suffix, "exponent_threshold" + suffix],
        outputs=["gt_exp_threshold" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Less",
        inputs=["bit_pos" + suffix, "sign_threshold" + suffix],
        outputs=["lt_sign_threshold" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "And",
        inputs=["gt_exp_threshold" + suffix, "lt_sign_threshold" + suffix],
        outputs=["is_exponent_bit" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Less",
        inputs=["bit_pos" + suffix, "exponent_threshold" + suffix],
        outputs=["is_mantissa_bit" + suffix]
    ))
    
    # 6. Handle sign bit flip (no change needed - Neg preserves type)
    nodes.append(helper.make_node(
        "Neg",
        inputs=[input_name],
        outputs=["sign_flipped" + suffix]
    ))
    
    # 7. Handle exponent bit flip - ensure consistent float16 usage
    # Calculate which exponent bit is affected
    nodes.append(helper.make_node(
        "Sub",
        inputs=["bit_pos" + suffix, "exponent_threshold" + suffix],
        outputs=["exponent_bit_position" + suffix]
    ))
    
    # Cast to float16 instead of float32
    nodes.append(helper.make_node(
        "Cast",
        inputs=["exponent_bit_position" + suffix],
        outputs=["exponent_bit_float" + suffix],
        to=TensorProto.FLOAT16  # Change to float16
    ))
    
    # The rest of the operations preserve type
    nodes.append(helper.make_node(
        "Pow",
        inputs=["two_constant" + suffix, "exponent_bit_float" + suffix],
        outputs=["exponent_weight" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Pow",
        inputs=["two_constant" + suffix, "exponent_weight" + suffix],
        outputs=["exponent_scale" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Mul",
        inputs=[input_name, "exponent_scale" + suffix],
        outputs=["exponent_flipped" + suffix]
    ))
    
    # 8. Handle mantissa bit flip - ensure consistent float16 usage
    nodes.append(helper.make_node(
        "Cast",
        inputs=["bit_pos" + suffix],
        outputs=["mantissa_bit_float" + suffix],
        to=TensorProto.FLOAT16  # Change to float16
    ))
    
    # Use float16 for mantissa base constant
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["mantissa_base" + suffix],
        value=helper.make_tensor(
            name="mantissa_base_tensor" + suffix,
            data_type=TensorProto.FLOAT16,  # Change to float16
            dims=[],
            vals=[np.float16(2**(-10)).view(np.uint16).item()]  # Properly encode float16
        )
    ))
    
    nodes.append(helper.make_node(
        "Pow",
        inputs=["two_constant" + suffix, "mantissa_bit_float" + suffix],
        outputs=["bit_power" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Mul",
        inputs=["mantissa_base" + suffix, "bit_power" + suffix],
        outputs=["mantissa_delta" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Add",
        inputs=[input_name, "mantissa_delta" + suffix],
        outputs=["mantissa_flipped" + suffix]
    ))
    
    # 9. Selection logic (no changes needed - these operations preserve type)
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_sign_bit" + suffix, "sign_flipped" + suffix, "exponent_flipped" + suffix],
        outputs=["sign_or_exponent" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Or",
        inputs=["is_sign_bit" + suffix, "is_exponent_bit" + suffix],
        outputs=["is_sign_or_exponent" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_sign_or_exponent" + suffix, "sign_or_exponent" + suffix, "mantissa_flipped" + suffix],
        outputs=[output_name]
    ))
    
    return nodes

# Import your bit flip function or paste it here
# from your_module import create_float_bit_flip_simulation

def test_float_bit_flip_simulation(bit_position, input_values):
    """
    Test the mathematical effects of bit flip simulation.
    
    Args:
        bit_position: Which bit to flip (0-15 for float16)
        input_values: Array of input float values to test
    """
    # Convert input to float16
    input_data = np.array(input_values, dtype=np.float16)
    
    # Create model inputs and outputs
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT16, [len(input_values)])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT16, [len(input_values)])
    
    # Get nodes for bit flip simulation
    bit_flip_nodes = create_float_bit_flip_simulation("input", "output", bit_position)
    
    # Create graph and model
    graph = helper.make_graph(
        nodes=bit_flip_nodes,
        name=f"BitFlip_{bit_position}_Test",
        inputs=[X],
        outputs=[Y],
        initializer=[]
    )
    
    model = helper.make_model(graph)
    model.opset_import[0].version = 17
    
    # Run inference
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    result = session.run(None, {"input": input_data})[0]
    
    # Calculate expected values using our mathematical understanding
    expected = calculate_mathematical_expectation(input_data, bit_position)
    
    # Print results
    print(f"=== Testing Bit Position {bit_position} ===")
    print(f"{'Input':<10} {'Result':<12} {'Expected':<12} {'Match':<6}")
    
    for i in range(len(input_values)):
        inp = input_data[i]
        res = result[i]
        exp = expected[i]
        match = np.isclose(res, exp, rtol=1e-3, atol=1e-3)
        
        print(f"{inp:<10.4f} {res:<12.4f} {exp:<12.4f} {match}")
    
    # Summary
    matches = np.isclose(result, expected, rtol=1e-3, atol=1e-3)
    match_percentage = np.sum(matches) / len(matches) * 100
    print(f"Match rate: {match_percentage:.1f}% ({np.sum(matches)}/{len(matches)})")
    print("=" * 50)
    print()
    
    return result, expected

def calculate_mathematical_expectation(values, bit_position):
    """
    Calculate expected values based on mathematical transformations.
    No bit manipulation - just the math we're simulating.
    """
    # Copy the input
    result = values.copy()
    
    # Handle each case separately based on IEEE-754 float16 structure
    if bit_position == 15:  # Sign bit
        # Just negate all values
        result = -result
        
    elif 10 <= bit_position <= 14:  # Exponent bits
        # Calculate which exponent bit (0-4) is affected
        exponent_pos = bit_position - 10
        
        # Scaling factor: 2^(2^exponent_pos)
        scaling = 2.0 ** (2.0 ** exponent_pos)
        
        # Apply scaling - either multiply or divide
        # This is a simplification - proper handling would check current bit
        result = result * scaling
        
    else:  # Mantissa bits (0-9)
        # Weight of this bit: 2^(bit_position-10)
        delta = 2.0 ** (bit_position - 10)
        
        # Add the delta
        # This is a simplification - proper handling would check current bit
        result = result + delta
    
    # Convert back to float16 to match precision
    return result.astype(np.float16)

# Run tests for each bit category
def run_comprehensive_tests():
    # Test sign bit (bit 15)
    print("SIGN BIT TESTS (NEGATION):")
    test_float_bit_flip_simulation(15, [1.0, -1.0, 2.0, -2.0, 0.0, 0.5, -0.5])
    
    # Test exponent bits (bits 10-14)
    print("EXPONENT BIT TESTS (POWERS OF 2):")
    test_float_bit_flip_simulation(14, [1.0, 2.0, 4.0, 0.5, 0.25])  # Highest exponent bit
    test_float_bit_flip_simulation(13, [1.0, 2.0, 4.0, 0.5, 0.25])  # Second exponent bit
    test_float_bit_flip_simulation(10, [1.0, 1.25, 1.5, 1.75, 2.0])  # Lowest exponent bit
    
    # Test mantissa bits (bits 0-9)
    print("MANTISSA BIT TESTS (FRACTIONAL CHANGES):")
    test_float_bit_flip_simulation(9, [1.0, 1.5, 2.0, 2.5, 3.0])    # Highest mantissa bit (0.5)
    test_float_bit_flip_simulation(8, [1.0, 1.25, 1.5, 1.75, 2.0])  # Second mantissa bit (0.25)
    test_float_bit_flip_simulation(0, [1.0, 1.0625, 1.125, 1.25])   # Lowest mantissa bit

if __name__ == "__main__":
    run_comprehensive_tests()