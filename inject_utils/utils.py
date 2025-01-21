from PIL import Image
import numpy as np
import struct
import torchvision.transforms as transforms
import torch
import math
import sys
import onnx
from typing import List

def debug(faulty_value, golden_value, weight_dict, target_indices, input_dict, faulty_tensor_name, output_tensors, original_tensor_value, dequantized_result_tensor_name, perturb):
    random_indices = [np.random.randint(0, dim) for dim in weight_dict[faulty_tensor_name].shape]
    print("target shape:")
    print(weight_dict[faulty_tensor_name].shape)
    print("target index:")
    print(target_indices)
    print("Faulty Value:")
    print(input_dict[faulty_tensor_name][tuple(target_indices)])
    print("Original Value:")
    print(original_tensor_value[tuple(target_indices)])
    print("Injected Results:")
    print(output_tensors[(list(output_tensors.keys())[0])][tuple(target_indices)])
    print("Original Results:")
    print(weight_dict[dequantized_result_tensor_name][tuple(target_indices)])
    print("Perturb:")
    print(np.nonzero(weight_dict["delta_4d"]))
    print(weight_dict["delta_4d"][tuple(target_indices)])
    print("Just Perturb:")
    print(perturb)

def fp32tobin(value):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', value))

def bin2fp32(bin_str):
    assert len(bin_str) == 32
    data = struct.unpack('!f',struct.pack('!I', int(bin_str, 2)))[0]
    if np.isnan(data):
        return 0
    else:
        return data

# Converts a fp16 value to binary
def fp16tobin(fp):
    sign = math.copysign(1,fp)
    abs_fp = abs(fp)
    # Handling subnormal numbers
    if abs_fp < pow(2,-14):
        target_fp = abs_fp * pow(2,14)
        exponent_bin = '00000'
        frac_bin = ''
        frac_mid = target_fp
        for i in range(25):
            frac_mid *= 2
            if frac_mid >= 1.0:
                frac_bin += '1'
                frac_mid -= 1.0
            else:
                frac_bin += '0'
        mantissa_bin = frac_bin
    # Handling normal numbers
    else:
        int_part = int(np.fix(abs_fp))
        frac_part = abs_fp - int_part
        int_bin = bin(int_part)[2:]
        frac_bin = ''
        frac_mid = frac_part
        for i in range(25):
            frac_mid *= 2
            if frac_mid >= 1.0:
                frac_bin += '1'
                frac_mid -= 1.0
            else:
                frac_bin += '0'
        int_frac_bin = int_bin + frac_bin
        # Decimal point is at the back of variable decimal_point
        decimal_point = len(int_bin)-1
        # Looking for the first 1
        first_one = int_frac_bin.find('1')
        # Special case: 0
        if first_one < 0:
            return ('0x00', '0x00')
        exponent_val = decimal_point - first_one + 15
        assert exponent_val <= 31
        assert exponent_val >= 0
        exponent_bin = bin(exponent_val)[2:].zfill(5)
        mantissa_bin = int_frac_bin[first_one+1:]
        if len(mantissa_bin) < 10:
            mantissa_bin = mantissa_bin.zfill(10)
    if sign == 1.0:
        sign_bin = '0'
    else:
        sign_bin = '1'
    total_bin = (sign_bin + exponent_bin + mantissa_bin)[:16]
    return total_bin

# Converts a binary string to FP16 values
def bin2fp16(bin_str):
    assert len(bin_str) == 16
    sign_bin = bin_str[0]
    if sign_bin == '0':
        sign_val = 1.0
    else:
        sign_val = -1.0
    exponent_bin = bin_str[1:6]
    mantissa_bin = bin_str[6:]
    assert len(mantissa_bin) == 10
    exponent_val = int(exponent_bin,2)
    mantissa_val = 0.0
    for i in range(10):
        if mantissa_bin[i] == '1':
            mantissa_val += pow(2,-i-1)
    # Handling subnormal numbers
    if exponent_val == 0:
        return sign_val * pow(2,-14) * mantissa_val
    # Handling normal numbers
    else:
        value = sign_val * pow(2,exponent_val-15) * (1 + mantissa_val)
        # Handling NaNs and INFs
        if value == 65536:
            return 65535
        elif value == -65536:
            return -65535
        elif value > 65536 or value < -65536:
            return 0
        else:
            return value

def delta_init(is_float32=True):
    one_bin = ''
    if is_float32:
        for _ in range(32):
            one_bin += str(np.random.randint(0,2))
        return bin2fp32(one_bin)
    for _ in range(16):
        one_bin += str(np.random.randint(0,2))
    return bin2fp16(one_bin)

def delta_init_int8():
    one_bin = ''
    for _ in range(8):
        one_bin += str(np.random.randint(0,2))
    return np.int8(int(one_bin, 2))


def preprocess_cifar10(input_value):
    mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
    image = Image.fromarray(input_value)
    input_value = transforms.functional.pil_to_tensor(image)
    input_value = transforms.functional.resize(input_value, (224, 224), antialias=True)
    input_value = input_value/255
    input_value = transforms.functional.normalize(input_value, mean, std)
    return input_value

def preprocess_cifar10_inception(input_value):
    image = Image.fromarray(input_value)
    input_value = transforms.functional.pil_to_tensor(image)
    input_value = transforms.functional.resize(input_value, (256, 256), antialias=True)
    input_value = transforms.functional.center_crop(input_value, (224, 224))
    input_value = input_value/255
    x = input_value

    """
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)
    x_ch0 = torch.unsqueeze(x[0]) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    x_ch1 = torch.unsqueeze(x[1]) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    x_ch2 = torch.unsqueeze(x[2]) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    x = torch.cat((x_ch0, x_ch1, x_ch2), 0)
    """

    return x 

def get_target_inputs(graph, layer_name, input_name, weight_name, bias_name, output_tensor):
    input_quantizer_name = None
    weight_quantizer_name = None
    bias_quantizer_name = None

    int_input_tensor_name = None
    int_weight_tensor_name = None
    int_bias_tensor_name = None

    quantizer_input_node = None
    quantizer_weight_node = None

    """
    print("WEIGHT NAME:")
    print(weight_name)
    """
    # Retrieve the quantizer node
    # Additional retrieval of transpose node to get axes
    transposed_output_name = None
    #transposed_node_name = None
    transposed_node = None
    #transposed_axes = None
    for node in graph.node:
        if node.name == layer_name:
            layer_node = node
            for input_node in node.input:
                if "Transpose" in input_node:
                    transposed_output_name = input_node

        for input_node in node.input:
            if input_node == input_name:
                input_quantizer_name = node.name
                quantizer_input_node = node
                for input_tensor in node.input:
                    if "out0" in input_tensor:
                        int_input_tensor_name = input_tensor
                        break
                intermediate_operation_names = []
                intermediate_operation_names.append(quantizer_input_node.name)
                temporary_output_name = quantizer_input_node.output[0]
                for outer_node in graph.node:
                    if intermediate_operation_names[-1] == layer_name:
                        break
                    for inner_input_node in outer_node.input:
                        if temporary_output_name == inner_input_node:
                            intermediate_operation_names.append(outer_node.name)
                            temporary_output_name = outer_node.output[0]
                input_intermediate_operations = intermediate_operation_names

            if input_node == weight_name:
                weight_quantizer_name = node.name
                quantizer_weight_node = node
                for input_tensor in node.input:
                    if "out0" in input_tensor:
                        int_weight_tensor_name = input_tensor
                        break
                intermediate_operation_names = []
                intermediate_operation_names.append(quantizer_weight_node.name)
                temporary_output_name = quantizer_weight_node.output[0]
                for outer_node in graph.node:
                    if intermediate_operation_names[-1] == layer_name:
                        break
                    for inner_input_node in outer_node.input:
                        if temporary_output_name == inner_input_node:
                            intermediate_operation_names.append(outer_node.name)
                            temporary_output_name = outer_node.output[0]
                weight_intermediate_operations = intermediate_operation_names

    """
    print(input_intermediate_operations)
    print(weight_intermediate_operations)
    """

    check_1 = (int_input_tensor_name in quantizer_input_node.input) and (int_weight_tensor_name in quantizer_weight_node.input)
    check_2 = output_tensor in layer_node.output

    if not (check_1 and check_2):
        print(check_1)
        print(check_2)
        print(input_name, weight_name, bias_name)
        exit()
    return (input_quantizer_name, int_input_tensor_name), (weight_quantizer_name, int_weight_tensor_name), (bias_quantizer_name, int_bias_tensor_name), (input_intermediate_operations, weight_intermediate_operations)

def expand_node_inputs_outputs(graph, node):
    added_inputs = []
    added_outputs = []

    added_inputs += list(filter(lambda x: x.name in node.input, graph.input))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.output))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.output))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))

    return added_inputs, added_outputs

def get_tensor_shape(model: onnx.ModelProto, tensor_name: str) -> List[int]:
    # Check all possible tensor locations
    for tensor in (list(model.graph.input) + 
                  list(model.graph.output) + 
                  list(model.graph.value_info)):
        if tensor.name == tensor_name:
            try:
                shape = [dim.dim_value for dim in 
                        tensor.type.tensor_type.shape.dim]
                if all(isinstance(d, int) for d in shape):
                    return shape
            except AttributeError:
                continue
                
    raise ValueError(f"Could not find valid shape for tensor: {tensor_name}")

def total_bits_diff(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"

    flat_tensor1 = tensor1.flatten()
    flat_tensor2 = tensor2.flatten()

    total_diff = 0
    second_diff = 0
    for val1, val2 in zip(flat_tensor1, flat_tensor2):
        if val1 != val2:
            second_diff += 1
        """
        signed_val1 = np.int64(val1)
        signed_val2 = np.int64(val2)

        bin_val1 = np.binary_repr(signed_val1, width=64)
        bin_val2 = np.binary_repr(signed_val2, width=64)
        bin_val1 = np.binary_repr(val1)
        bin_val2 = np.binary_repr(val2)

        # Count the number of different bits
        diff_bits = sum(b1 != b2 for b1, b2 in zip(bin_val1, bin_val2))
        total_diff += diff_bits
        """

    #total_ff = (list(flat_tensor1.shape)[0])
    total_ff = 0
    print("SECOND DIFF:" + str(second_diff))
    return total_diff, second_diff, total_ff

def debug_inject_parameters(inject_input):
    for key in inject_input.keys():
        if key != "original_weight_dict" and key != "main_graph":
            print(inject_input[key])
