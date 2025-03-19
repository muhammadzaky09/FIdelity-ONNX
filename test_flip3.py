import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import struct

from onnx import helper, TensorProto

from onnx import helper, TensorProto

def create_fp16_bit_flip_improved(input_name, output_name, bit_position):
    """
    Create an ONNX graph to flip a single FP16 bit using a hybrid approach.
    
    This version implements:
      • A special branch for exponent bits (10-14) that:
         - Computes the unbiased exponent and then the new biased exponent
         - Determines if the target bit is set
         - Reconstructs the new absolute value from the new exponent and original mantissa,
           with an extra Round op to reduce precision error.
         - Reapplies the original sign.
         - If the input was zero, substitutes a precomputed zero_result.
      • A simple branch for the sign bit (15) that just negates the input.
      • A placeholder for the mantissa bits (0-9).
    
    Args:
      input_name: Name of input tensor (FLOAT16)
      output_name: Name of output tensor (FLOAT16)
      bit_position: Bit to flip (0-15)
    
    Returns:
      List of ONNX nodes.
    """
    nodes = []
    suffix = "_imp"

    # 1. Cast input from FP16 to FP32 for arithmetic precision.
    nodes.append(helper.make_node(
        "Cast",
        inputs=[input_name],
        outputs=["input_fp32" + suffix],
        to=TensorProto.FLOAT
    ))

    # 2. Define basic constants.
    constants = {
        "zero": 0.0,
        "one": 1.0,
        "two": 2.0,
        "epsilon": 1e-30,
        "ln2": 0.693147180559945,
        "bias": 15.0  # FP16 bias.
    }
    for name, val in constants.items():
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[name + suffix],
            value=helper.make_tensor(f"{name}_tensor" + suffix, TensorProto.FLOAT, [], [val])
        ))
    # Constant for negative one (for sign reapplication).
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["neg_one" + suffix],
        value=helper.make_tensor("neg_one_tensor" + suffix, TensorProto.FLOAT, [], [-1.0])
    ))

    # 3. For exponent bit flipping, precompute the value that zero should map to.
    # For exponent bits: expected zero_result = 2^( (2^(bit_position-10)) - 15 ).
    if 10 <= bit_position <= 14:
        exponent_power = bit_position - 10
        bit_value = 2 ** exponent_power  # integer value of the exponent bit.
        zero_result_value = 2.0 ** (bit_value - 15)
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["zero_result" + suffix],
            value=helper.make_tensor("zero_result_tensor" + suffix, TensorProto.FLOAT, [], [zero_result_value])
        ))

    # 4. Detect if the input is zero.
    nodes.append(helper.make_node(
        "Equal",
        inputs=["input_fp32" + suffix, "zero" + suffix],
        outputs=["is_zero" + suffix]
    ))
    # Also detect negativity for sign reapplication.
    nodes.append(helper.make_node(
        "Less",
        inputs=["input_fp32" + suffix, "zero" + suffix],
        outputs=["is_negative" + suffix]
    ))

    # 5. Branch on bit_position.
    if bit_position == 15:
        # SIGN BIT: simply negate the input.
        nodes.append(helper.make_node(
            "Neg",
            inputs=["input_fp32" + suffix],
            outputs=["result_fp32" + suffix]
        ))
    elif 10 <= bit_position <= 14:
        # EXPONENT BITS branch.
        # a) Compute absolute value and its exponent.
        nodes.append(helper.make_node(
            "Abs",
            inputs=["input_fp32" + suffix],
            outputs=["abs_val" + suffix]
        ))
        nodes.append(helper.make_node(
            "Add",
            inputs=["abs_val" + suffix, "epsilon" + suffix],
            outputs=["abs_safe" + suffix]
        ))
        nodes.append(helper.make_node(
            "Log",
            inputs=["abs_safe" + suffix],
            outputs=["log_val" + suffix]
        ))
        nodes.append(helper.make_node(
            "Div",
            inputs=["log_val" + suffix, "ln2" + suffix],
            outputs=["log2_val" + suffix]
        ))
        nodes.append(helper.make_node(
            "Floor",
            inputs=["log2_val" + suffix],
            outputs=["exp_val" + suffix]
        ))
        # Biased exponent = exp_val + bias.
        nodes.append(helper.make_node(
            "Add",
            inputs=["exp_val" + suffix, "bias" + suffix],
            outputs=["biased_exp" + suffix]
        ))
        # b) Determine if the target exponent bit is set.
        bit_value = 2 ** (bit_position - 10)
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["bit_value" + suffix],
            value=helper.make_tensor("bit_value_tensor" + suffix, TensorProto.FLOAT, [], [float(bit_value)])
        ))
        nodes.append(helper.make_node(
            "Div",
            inputs=["biased_exp" + suffix, "bit_value" + suffix],
            outputs=["div_res" + suffix]
        ))
        nodes.append(helper.make_node(
            "Floor",
            inputs=["div_res" + suffix],
            outputs=["div_floor" + suffix]
        ))
        nodes.append(helper.make_node(
            "Div",
            inputs=["div_floor" + suffix, "two" + suffix],
            outputs=["div_by_two" + suffix]
        ))
        nodes.append(helper.make_node(
            "Floor",
            inputs=["div_by_two" + suffix],
            outputs=["div_by_two_floor" + suffix]
        ))
        nodes.append(helper.make_node(
            "Mul",
            inputs=["div_by_two_floor" + suffix, "two" + suffix],
            outputs=["two_mul" + suffix]
        ))
        nodes.append(helper.make_node(
            "Sub",
            inputs=["div_floor" + suffix, "two_mul" + suffix],
            outputs=["mod_bit" + suffix]
        ))
        nodes.append(helper.make_node(
            "Equal",
            inputs=["mod_bit" + suffix, "one" + suffix],
            outputs=["is_bit_set" + suffix]
        ))
        # c) Compute new biased exponent: subtract bit_value if set, else add.
        nodes.append(helper.make_node(
            "Sub",
            inputs=["biased_exp" + suffix, "bit_value" + suffix],
            outputs=["biased_exp_minus" + suffix]
        ))
        nodes.append(helper.make_node(
            "Add",
            inputs=["biased_exp" + suffix, "bit_value" + suffix],
            outputs=["biased_exp_plus" + suffix]
        ))
        nodes.append(helper.make_node(
            "Where",
            inputs=["is_bit_set" + suffix, "biased_exp_minus" + suffix, "biased_exp_plus" + suffix],
            outputs=["new_biased_exp" + suffix]
        ))
        # d) New exponent = new_biased_exp - bias.
        nodes.append(helper.make_node(
            "Sub",
            inputs=["new_biased_exp" + suffix, "bias" + suffix],
            outputs=["new_exp" + suffix]
        ))
        # e) Reconstruct the new absolute value.
        nodes.append(helper.make_node(
            "Pow",
            inputs=["two" + suffix, "new_exp" + suffix],
            outputs=["new_pow2" + suffix]
        ))
        nodes.append(helper.make_node(
            "Pow",
            inputs=["two" + suffix, "exp_val" + suffix],
            outputs=["pow2_exp" + suffix]
        ))
        nodes.append(helper.make_node(
            "Div",
            inputs=["abs_val" + suffix, "pow2_exp" + suffix],
            outputs=["normalized" + suffix]
        ))
        # Mantissa fraction = normalized - 1.
        nodes.append(helper.make_node(
            "Sub",
            inputs=["normalized" + suffix, "one" + suffix],
            outputs=["mantissa_frac" + suffix]
        ))
        # Insert rounding to reduce precision error.
        nodes.append(helper.make_node(
            "Round",
            inputs=["mantissa_frac" + suffix],
            outputs=["mantissa_frac_round" + suffix]
        ))
        # Reconstruct the mantissa: 1 + rounded fraction.
        nodes.append(helper.make_node(
            "Add",
            inputs=["one" + suffix, "mantissa_frac_round" + suffix],
            outputs=["mantissa_val" + suffix]
        ))
        # Multiply by 2^(new_exp) to get new absolute result.
        nodes.append(helper.make_node(
            "Mul",
            inputs=["new_pow2" + suffix, "mantissa_val" + suffix],
            outputs=["abs_result" + suffix]
        ))
        # f) Reapply the original sign.
        nodes.append(helper.make_node(
            "Where",
            inputs=["is_negative" + suffix, "neg_one" + suffix, "one" + suffix],
            outputs=["sign_factor" + suffix]
        ))
        nodes.append(helper.make_node(
            "Mul",
            inputs=["abs_result" + suffix, "sign_factor" + suffix],
            outputs=["regular_result" + suffix]
        ))
        # g) Handle overflow: if new_biased_exp >= 31, substitute a precomputed overflow value.
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["max_biased" + suffix],
            value=helper.make_tensor("max_biased_tensor" + suffix, TensorProto.FLOAT, [], [31.0])
        ))
        nodes.append(helper.make_node(
            "GreaterOrEqual",
            inputs=["new_biased_exp" + suffix, "max_biased" + suffix],
            outputs=["is_overflow" + suffix]
        ))
        # For demonstration, use a constant overflow value.
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["overflow_result" + suffix],
            value=helper.make_tensor("overflow_result_tensor" + suffix, TensorProto.FLOAT, [], [65535.0])
        ))
        nodes.append(helper.make_node(
            "Where",
            inputs=["is_overflow" + suffix, "overflow_result" + suffix, "regular_result" + suffix],
            outputs=["result_fp32" + suffix]
        ))
    else:
        # MANTISSA BITS (0-9) branch (placeholder).
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=["mantissa_flip_placeholder" + suffix],
            value=helper.make_tensor("mantissa_flip_tensor" + suffix, TensorProto.FLOAT, [], [0.0])
        ))
        nodes.append(helper.make_node(
            "Identity",
            inputs=["mantissa_flip_placeholder" + suffix],
            outputs=["result_fp32" + suffix]
        ))
    
    # 6. Final branch: if input was zero, then for exponent bits use the precomputed zero_result.
    if 10 <= bit_position <= 14:
        nodes.append(helper.make_node(
            "Where",
            inputs=["is_zero" + suffix, "zero_result" + suffix, "result_fp32" + suffix],
            outputs=["final_result" + suffix]
        ))
    else:
        nodes.append(helper.make_node(
            "Where",
            inputs=["is_zero" + suffix, "input_fp32" + suffix, "result_fp32" + suffix],
            outputs=["final_result" + suffix]
        ))
    
    # 7. Cast the final result back to FP16.
    nodes.append(helper.make_node(
        "Cast",
        inputs=["final_result" + suffix],
        outputs=[output_name],
        to=TensorProto.FLOAT16
    ))
    
    return nodes



def float16_to_binary(value):
    as_float16 = np.float16(value)
    bits = struct.unpack('H', np.array(as_float16, dtype=np.float16).tobytes())[0]
    return bin(bits)[2:].zfill(16)

def direct_bit_toggle(value, bit_position):
    as_float16 = np.float16(value)
    bits = struct.unpack('H', np.array(as_float16, dtype=np.float16).tobytes())[0]
    toggled_bits = bits ^ (1 << bit_position)
    return float(np.frombuffer(struct.pack('H', toggled_bits), dtype=np.float16)[0])

def test_improved_fp16_bit_flip(bit_position):
    # Create the model using the improved function
    input_name = "input"
    output_name = "output"
    nodes = create_fp16_bit_flip_improved(input_name, output_name, bit_position)
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, [1])
    output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, [1])
    graph = helper.make_graph(nodes, "improved_bit_flip", [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="test_improved_bit_flip")
    model.opset_import[0].version = 21
    onnx.save(model, "improved_bit_flip.onnx")
    
    # Create an inference session
    session = ort.InferenceSession("improved_bit_flip.onnx")
    
    # Define a set of test values (edge cases and normal values)
    test_values = [
        0.0,
        -0.0,
        np.float16(5.96e-8),   # min subnormal
        np.float16(6.10e-5),   # min normal
        np.float16(65504.0),   # max normal
        np.float16(256.0),
        np.float16(-256.0)
    ]
    
    for value in test_values:
        input_data = np.array([np.float16(value)], dtype=np.float16)
        output = session.run([output_name], {input_name: input_data})
        model_result = np.float16(output[0][0])
        expected = np.float16(direct_bit_toggle(value, bit_position))
        
        print(f"Testing value: {np.float16(value)} (binary: {float16_to_binary(value)})")
        print(f"Model result: {model_result} (binary: {float16_to_binary(model_result)})")
        print(f"Expected:     {expected} (binary: {float16_to_binary(expected)})")
        print("Match:" , float16_to_binary(model_result) == float16_to_binary(expected))
        print("-" * 50)

# Example: Test for exponent bit position 12
test_improved_fp16_bit_flip(12)
