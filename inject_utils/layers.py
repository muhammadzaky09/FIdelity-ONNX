from onnx import helper, ModelProto, TensorProto, OperatorSetIdProto, shape_inference
from qonnx.core.onnx_exec import execute_onnx
#from inject_utils.utils import *
import numpy as np
import struct

def fp32tobin(value):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', value))

def bin2fp32(bin_str):
    assert len(bin_str) == 32
    data = struct.unpack('!f',struct.pack('!I', int(bin_str, 2)))[0]
    if np.isnan(data):
        return 0
    else:
        return data

def delta_init():
    one_bin = ''
    for _ in range(32):
        one_bin += str(np.random.randint(0,2))
    return bin2fp32(one_bin)

def float32_bit_flip(faulty_tensor, target_indices):
    golden_value = faulty_tensor[tuple(target_indices)]
    golden_string = fp32tobin(golden_value)
    print("Original:",golden_string)
    flip_bit = np.random.randint(32)
    if golden_string[31-flip_bit] == '1':
        inject_string = golden_string[:31-flip_bit] + '0' + golden_string[31-flip_bit+1:]
    else:
        inject_string = golden_string[:31-flip_bit] + '1' + golden_string[31-flip_bit+1:]
    print("Bitflipped:", inject_string)
    faulty_value = bin2fp32(inject_string)
    return faulty_value, flip_bit

def float16_bit_flip(faulty_tensor, target_indices, bit_position=None):
    golden_value = faulty_tensor[tuple(target_indices)]
    golden_string = fp16tobin(golden_value)
    flip_bit = np.random.randint(16)
    if bit_position is not None:
        flip_bit = bit_position 
    if golden_string[15-flip_bit] == '1':
        inject_string = golden_string[:15-flip_bit] + '0' + golden_string[15-flip_bit+1:]
    else:
        inject_string = golden_string[:15-flip_bit] + '1' + golden_string[15-flip_bit+1:]
    faulty_value = bin2fp16(inject_string)
    return faulty_value, flip_bit

def flip_int4_bit(value, bit_position):
    mask = 1 << bit_position
    flipped_value = value ^ mask
    if flipped_value > 7:
        flipped_value -= 16
    if flipped_value < -8:
        flipped_value += 16
    print("VALUE")
    print(value)
    print("FLIPPED")
    print(flipped_value)
    return flipped_value

def flip_int8_bit(value, bit_position):
    mask = 1 << bit_position
    flipped_value = value ^ mask
    if flipped_value > 127:
        flipped_value -= 256
    if flipped_value < -128:
        flipped_value += 256
    return flipped_value

def int_bit_flip(input_dict, target_tensor, target_bit_position, bit_precision=4):
    faulty_tensor = input_dict[target_tensor]
    faulty_tensor = np.int8(faulty_tensor)
    random_indices = [np.random.randint(0, dim) for dim in faulty_tensor.shape]
    """
    print("ORIGINAL VALUE:")
    print(faulty_tensor[tuple(random_indices)])
    """
    faulty_value = flip_int8_bit(faulty_tensor[tuple(random_indices)], target_bit_position)
    assert faulty_value >= -128 and faulty_value <= 127
    """
    faulty_value = flip_int4_bit(faulty_tensor[tuple(random_indices)], target_bit_position)
    assert faulty_value >= -8 and faulty_value <= 7
    """
    return faulty_value, random_indices


def perturb_quantizer(graph, node, module, model, input_dict, weight_dict, faulty_tensor_name, faulty_bit_position):
    faulty_value, target_indices = int_bit_flip(weight_dict, faulty_tensor_name, faulty_bit_position, 4)
    """
    print("FAULTY VALUE:")
    print(faulty_value)
    
    print("INPUT")
    print(input_dict[list(input_dict.keys())[0]].shape)
    print(input_dict[list(input_dict.keys())[1]].shape)
    """
    golden_value = weight_dict[faulty_tensor_name][tuple(target_indices)]
    is_signed = ""
    if isinstance(golden_value, np.uint8):
        is_signed = "Unsigned"
    else:
        is_signed = "Signed"
    original_tensor_value = input_dict[faulty_tensor_name]

    input_perturb = np.zeros(weight_dict[faulty_tensor_name].shape, dtype=weight_dict[faulty_tensor_name].dtype)
    input_perturb[tuple(target_indices)] = faulty_value
    input_dict[faulty_tensor_name] = input_perturb
    
    """
    print("INPUT INNER:")
    print(input_dict[list(input_dict.keys())[0]].shape)
    print(input_dict[list(input_dict.keys())[1]].shape)
    print("FOCUS GRAPH:")

    print(graph)
    """
    if module == "Decoder":
        replacement_dictionary = {
            "onnx::ReduceMean_0_dynamic_axes_1": weight_dict["global_in"].shape[1],
            "onnx::Unsqueeze_3_dynamic_axes_1": weight_dict["global_in_3"].shape[1],
            "onnx::Unsqueeze_3_dynamic_axes_2": weight_dict["global_in_3"].shape[2],
        }
        for output in model.graph.output:
            for dimension in range(len(output.type.tensor_type.shape.dim)):
                for key in replacement_dictionary.keys():
                    if key in str(output.type.tensor_type.shape.dim[dimension]):
                        output.type.tensor_type.shape.dim[dimension].Clear()
                        output.type.tensor_type.shape.dim[dimension].dim_value = replacement_dictionary[key]
                    if "unk__" in str(output.type.tensor_type.shape.dim[dimension]):
                        output.type.tensor_type.shape.dim[dimension].Clear()
                        output.type.tensor_type.shape.dim[dimension].dim_value = weight_dict[output.name].shape[dimension]


    output_tensors = execute_onnx(model, input_dict)
    weight_dict["delta_4d"] = output_tensors[(list(output_tensors.keys())[0])]
    faulty_index_value = weight_dict["delta_4d"][tuple(target_indices)]

    dequantized_result_tensor_name = list(output_tensors.keys())[0]
    perturb = weight_dict["delta_4d"][tuple(target_indices)] - weight_dict[dequantized_result_tensor_name][tuple(target_indices)]
    weight_dict["delta_4d"][tuple(target_indices)] = perturb

    return weight_dict#, dequantized_result_tensor_name, target_indices, golden_value, faulty_value, is_signed


def perturb_fp16(model, input_dict, weight_dict, faulty_tensor_name, faulty_bit_position):
    target_indices = [np.random.randint(0, dim) for dim in weight_dict[faulty_tensor_name].shape]
    faulty_value, _ = float16_bit_flip(input_dict[faulty_tensor_name], target_indices, faulty_bit_position)
    golden_value = weight_dict[faulty_tensor_name][tuple(target_indices)]
    is_signed = "float16" 
    original_tensor_value = input_dict[faulty_tensor_name]

    input_perturb = np.zeros(weight_dict[faulty_tensor_name].shape, dtype=weight_dict[faulty_tensor_name].dtype)
    input_perturb[tuple(target_indices)] = faulty_value
    input_dict[faulty_tensor_name] = input_perturb

    output_tensors = execute_onnx(model, input_dict)
    weight_dict["delta_4d"] = output_tensors[(list(output_tensors.keys())[0])]
    faulty_index_value = weight_dict["delta_4d"][tuple(target_indices)]

    dequantized_result_tensor_name = list(output_tensors.keys())[0]
    perturb = weight_dict["delta_4d"][tuple(target_indices)] - weight_dict[dequantized_result_tensor_name][tuple(target_indices)]
    weight_dict["delta_4d"][tuple(target_indices)] = perturb

    return weight_dict, dequantized_result_tensor_name, target_indices, golden_value, faulty_value, is_signed

def perturb_conv(model, input_dict, weight_dict, input_tensor_name, bias_output_name):
    input_dict[input_tensor_name] = weight_dict["delta_4d"]
    no_bias = np.zeros(weight_dict[bias_output_name].shape, dtype=weight_dict[bias_output_name].dtype)
    input_dict[bias_output_name] = no_bias
    delta_perturb = execute_onnx(model, input_dict)
    delta_perturb = delta_perturb[list(delta_perturb.keys())[0]]
    return delta_perturb

def perturb_matmul(model, input_dict, weight_dict, input_tensor_name, transposed_axes=None):
    targetted_axes = None
    if transposed_axes:
        if (transposed_axes.input[0] in input_tensor_name):
            targetted_axes = list(transposed_axes.attribute[0].ints)
            input_tensor_name = transposed_axes.output[0]
            weight_dict["delta_4d"] = np.transpose(weight_dict["delta_4d"], tuple(targetted_axes))
    input_dict[input_tensor_name] = weight_dict["delta_4d"]
    delta_perturb = execute_onnx(model, input_dict)
    delta_perturb = delta_perturb[list(delta_perturb.keys())[0]]

    return delta_perturb

"""
def get_perturbation(golden_value, bit_position=None, fault_model=None):
    golden_value = np.int8(golden_value)
    if fault_model == "RANDOM":
        for _ in range(32):
            inj_bin += str(np.random.randint(0,2))
        perturb = bin2fp16(inj_bin) - golden_value
    else:
        flipped_value = flip_int8_bit(golden_value, bit_position)
"""
