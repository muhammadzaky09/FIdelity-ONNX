from onnx import helper, ModelProto, TensorProto, OperatorSetIdProto, shape_inference
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
import onnx.numpy_helper as numpy_helper
import numpy as np
import torch
import onnx

from onnx.shape_inference import infer_shapes
#from inject_utils.layers import int_bit_flip
from inject_utils.layers import perturb_quantizer
from inject_utils.layers import float32_bit_flip 
from inject_utils.layers import delta_init 

import time
import copy

def execute_node(node, main_graph, final_output_node, weight_dict, module, inject_parameters=None):
    node_inputs = []
    node_outputs = []

    added_quant_inputs, added_quant_outputs, list_operation_time = expand_node_inputs_outputs(main_graph, node, weight_dict, module)
    node_inputs += added_quant_inputs
    node_outputs += added_quant_outputs

    desired_node_outputs = [x for x in node_outputs if x.name == final_output_node]
    intermediate_node_outputs = [x for x in node_outputs if x.name != final_output_node]
    
    for index, node_input in enumerate(node.input):
        if (len(node_input)) == 0:
            node.input[index] = node_inputs[-1].name

    graph = helper.make_graph(
            nodes = [node],
            name = "single_node_exec",
            inputs = node_inputs,
            outputs = desired_node_outputs
    )

    model = ModelProto()
    model = infer_shapes(model)
    model.graph.CopyFrom(graph)
    model.opset_import.append(OperatorSetIdProto(version=13))
    model = ModelWrapper(model)

    input_dict = {}
    for node_iter in node_inputs:
        if node_iter.name == [node_intermediate.name for node_intermediate in intermediate_node_outputs]:
            continue
        if node_iter.name in [node_intermediate.name for node_intermediate in node_outputs]:
            continue
        input_dict[node_iter.name] = weight_dict[node_iter.name]

    output_tensors = execute_onnx(model, input_dict)
    tensor_output_name = list(output_tensors.keys())[0]
    original_tensor_output = output_tensors[tensor_output_name]
    weight_dict[tensor_output_name] = output_tensors[tensor_output_name]

    if inject_parameters and ("RANDOM" in inject_parameters["inject_type"]) and (node.name == inject_parameters["faulty_operation_name"]):
        print("FOUND HERE RANDOM:")
        print(node.name)
        faulty_value = None
        target_indices = [np.random.randint(0, dim) for dim in weight_dict[tensor_output_name].shape]
        golden_value = weight_dict[tensor_output_name][tuple(target_indices)]
        print(weight_dict[tensor_output_name][tuple(target_indices)])
        if "BITFLIP" in inject_parameters["inject_type"]:
            faulty_value, flip_bit = float32_bit_flip(weight_dict[tensor_output_name], target_indices)
        else:
            faulty_value = delta_init()
        weight_dict[tensor_output_name][tuple(target_indices)] = faulty_value
        print("FAULTY:")
        print(faulty_value)

    if inject_parameters and (module in inject_parameters["targetted_module"]) and (inject_parameters["faulty_trace"]) and (node.name == inject_parameters["faulty_trace"][0]) and (inject_parameters["inject_type"] in ["INPUT", "WEIGHT", "INPUT16", "WEIGHT16"]):
        faulty_operation = inject_parameters["faulty_trace"][0]

        # First layer in faulty_trace, obtains perturbations
        if inject_parameters["faulty_tensor_name"] in node.input:
            assert(inject_parameters["faulty_quantizer_name"] == inject_parameters["faulty_trace"][0])
            weight_dict = perturb_quantizer(graph, node, module, model, input_dict, weight_dict, inject_parameters["faulty_tensor_name"], inject_parameters["faulty_bit_position"])
            inject_parameters["intermediate_output_name"] = tensor_output_name

        # Rest of the layers in faulty_trace
        else:
            intermediate_input_name = None
            for input_node in node.input:
                if input_node == inject_parameters["intermediate_output_name"]:
                    intermediate_input_name = input_node
            assert intermediate_input_name
            input_dict[intermediate_input_name] = weight_dict["delta_4d"]
            intermediate_output_tensors = execute_onnx(model, input_dict)
            """
            print("--")
            print(input_dict)
            print(np.nonzero(intermediate_output_tensors[(list(intermediate_output_tensors.keys())[0])]))
            print(intermediate_output_tensors.keys())
            """
            weight_dict["delta_4d"] = intermediate_output_tensors[(list(intermediate_output_tensors.keys())[0])]
            """
            print(np.nonzero(intermediate_output_tensors[(list(intermediate_output_tensors.keys())[0])]))
            print(intermediate_output_tensors.keys())
            print("--")
            """
            inject_parameters["intermediate_output_name"] = tensor_output_name

        # Final layer in faulty_trace, should be the target layer and applies the fault models
        if faulty_operation == inject_parameters["faulty_operation_name"]:
            print("FINAL LAYER")
            print(faulty_operation)
            assert(len(inject_parameters["faulty_trace"]) == 1)
            if "INPUT16" == inject_parameters["inject_type"]:
                delta_16 = np.zeros(weight_dict["delta_4d"].shape, dtype=np.float32)
                random_shape = list(weight_dict["delta_4d"].shape)
                row_index = random_shape[-1]//16
                if row_index == 0:
                    row_index = 0
                else:
                    row_index = np.random.randint(0, row_index)
                row_index = row_index*16
                indices = []
                if len(np.nonzero(weight_dict["delta_4d"])[0]) > 0:
                    for shape_index_array in np.nonzero(weight_dict["delta_4d"]):
                        indices.append(list(shape_index_array)[0])
                    indices[-1] = row_index

                    """
                    print(indices)
                    print(weight_dict["delta_4d"][tuple(indices)])
                    """

                    for i in range(16):
                        if i >= random_shape[-1]:
                            break
                        """
                        print(indices) 
                        """
                        delta_16[tuple(indices)] = weight_dict["delta_4d"][(tuple(indices))]
                        indices[-1] = indices[-1] + 1
                    weight_dict["delta_4d"] = delta_16
                    """
                    print("THIS:")
                    print(np.nonzero(weight_dict["delta_4d"]))
                    random_shape[-1] = random_shape[-1]//16
                    start_index = [np.random.randint(i) for i in random_shape]
                    start_index[-1] = start_index[-1]*16
                    print(start_index)
                    for i in range(16):
                        start_index = tuple(start_index)
                        delta_16[start_index] = weight_dict["delta_4d"][start_index]
        
                        start_index = list(start_index)
                        start_index[-1] = start_index[-1]+1
                    weight_dict["delta_4d"] = delta_16
                    print("INPUT16")
                    """
            elif "WEIGHT16" == inject_parameters["inject_type"]:
                delta_16 = np.zeros(weight_dict["delta_4d"].shape, dtype=np.float32)
                random_shape = list(weight_dict["delta_4d"].shape)
                column_index = random_shape[-2]//16
                if column_index == 0:
                    column_index = 0
                else:
                    column_index = np.random.randint(0, column_index)
                column_index = column_index*16
                indices = []
                if len(np.nonzero(weight_dict["delta_4d"])[0]) > 0:
                    for shape_index_array in np.nonzero(weight_dict["delta_4d"]):
                        indices.append(list(shape_index_array)[0])
                    indices[-2] = column_index

                    for i in range(np.random.randint(1,16)):
                        if i >= random_shape[-2]:
                            break
                        """
                        print(indices) 
                        """
                        delta_16[tuple(indices)] = weight_dict["delta_4d"][(tuple(indices))]
                        indices[-2] = indices[-2] + 1
                    weight_dict["delta_4d"] = delta_16
                    """
                    start_index = tuple([np.random.randint(i) for i in random_shape])
                    print(start_index)
                    delta_16[start_index] = weight_dict["delta_4d"][start_index]
                    weight_dict["delta_4d"] = delta_16
                    print("WEIGHT16")
                    """
            else:
                print("INPUTS/WEIGHTS")
            print("FAULT INJECTED!")
            print(np.nonzero(weight_dict["delta_4d"]))
            temp_variable = (np.add(weight_dict[tensor_output_name], weight_dict["delta_4d"]))
            weight_dict[tensor_output_name] = temp_variable
            """
            print(output_tensors)
            print(tensor_output_name)
            print("DIFF:")
            print(np.nonzero(output_tensors[tensor_output_name] - weight_dict[tensor_output_name]))
            """
            output_tensors[tensor_output_name] = temp_variable
            """
            print(output_tensors)
            """

        inject_parameters["faulty_trace"] = inject_parameters["faulty_trace"][1:]
    
    if output_tensors is None:
        print("HERE")
        print(input_dict)
        print(input_dict.keys())
        print(node)
        print(node.name)
    return output_tensors, weight_dict, list_operation_time

def inference(main_graph, weight_dict, module, inject_parameters=None):
    def execute_single_node(node, weight_dict, main_graph, module):
        final_output_node = node.output[0]
        output_tensors, weight_dict, list_operation_time = execute_node(node, main_graph, final_output_node, weight_dict, module, inject_parameters)
        return output_tensors, weight_dict, list_operation_time
    output_tensors = None
    for node in main_graph.node:
        start_time = time.time()
        output_tensors, weight_dict, list_operation_time = execute_single_node(node, weight_dict, main_graph, module)
        """
        if output_tensors is None:
            print("HERE")
            print(input_dict)
            print(input_dict.keys())
            print(node)
            print(node.name)
        """
    if output_tensors is None:
        print("SINI SINI SINI")
        print(output_tensors)
    return output_tensors, weight_dict

def expand_node_inputs_outputs(graph, node, weight_dict, module):
    added_inputs = []
    added_outputs = []

    added_inputs += list(filter(lambda x: x.name in node.input, graph.input))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.output))
    added_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.output))
    added_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))

    start_time = time.time()

    if len(node.input) != len(added_inputs):
        if "Clip" in node.name:
            added_inputs.append(copy.deepcopy(added_inputs[0]))
            added_inputs[-1].name = added_inputs[-1].name[:-1] + "2"
            weight_dict[added_inputs[-1].name] = np.array(3.4e38, dtype=np.float32)

    if module == "Decoder":
        replacement_dictionary = {
            "onnx::ReduceMean_0_dynamic_axes_1": weight_dict["global_in"].shape[1],
            "onnx::Unsqueeze_3_dynamic_axes_1": weight_dict["global_in_3"].shape[1],
            "onnx::Unsqueeze_3_dynamic_axes_2": weight_dict["global_in_3"].shape[-2],
        }

        for input_tensor in added_inputs:
            for dimension in range(len(input_tensor.type.tensor_type.shape.dim)):
                for key in replacement_dictionary.keys():
                    if key in str(input_tensor.type.tensor_type.shape.dim[dimension]):
                        input_tensor.type.tensor_type.shape.dim[dimension].Clear()
                        input_tensor.type.tensor_type.shape.dim[dimension].dim_value = replacement_dictionary[key]
                    if "unk__" in str(input_tensor.type.tensor_type.shape.dim[dimension]):
                        input_tensor.type.tensor_type.shape.dim[dimension].Clear()
                        input_tensor.type.tensor_type.shape.dim[dimension].dim_value = weight_dict[input_tensor.name].shape[dimension]

    return added_inputs, added_outputs, time.time() - start_time

def get_weight_dict(module_path):
    module = ModelWrapper(module_path)
    module_graph = module.graph
    module_weights = module.graph.initializer
    module_weight_dict = {}
    for weight in module_weights:
        module_weight_dict[weight.name] = numpy_helper.to_array(weight)
    return module_graph, module_weight_dict

def prepare_inference(module_path, module_input_values):
    module = ModelWrapper(module_path)
    output = [node.name for node in module.graph.output]

    input_all = [node.name for node in module.graph.input]
    input_initializers = [node.name for node in module.graph.initializer]
    module_input_names = list(set(input_all) - set(input_initializers))

    module_graph, module_weight_dict = get_weight_dict(module_path)

    for input_name in module_input_names:
        module_weight_dict[input_name] = module_input_values[input_name]

    return module_weight_dict, module_graph

def run_module(module, input_values, module_filepath, module_weight_dict, module_graph, inject_parameters=None):
    #module_weight_dict, module_graph = prepare_inference(module_filepath, input_values)
    #start_time = time.time()
    for input_name in list(input_values.keys()):
        module_weight_dict[input_name] = input_values[input_name]
    #print("LOAD TIME: " + str(time.time() - start_time))

    return inference(module_graph, module_weight_dict, module, inject_parameters)

if __name__ == "__main__":
    module = "encoder"
    encoder_input_values = {
        "global_in": np.random.rand(1, 72, 512).astype(np.float32), 
        "global_in_1": np.random.choice([True, False], size=(1, 1, 72))}
    module_filepath = "./onnx/new_fixed/encoder_fixed.onnx"
    output_tensors, module_weight_dict = run_module(module, encoder_input_values, module_filepath)
    torch.save(module_weight_dict, "encoder.pt")

    print("ENCODER OUT:")
    print(output_tensors)

    module = "decoder"
    decoder_input_values = {
        "global_in": np.random.rand(1, 1, 512).astype(np.float32), 
        "global_in_1": np.random.rand(1, 72, 512).astype(np.float32), 
        "global_in_2": np.random.choice([True, False], size=(1, 1, 72)),
        "global_in_3": np.random.rand(1, 1, 1).astype(np.int64)}
    module_filepath = "./onnx/new_fixed/decoder_fixed.onnx"

    output_tensors, module_weight_dict = run_module(module, decoder_input_values, module_filepath)
    torch.save(module_weight_dict, "decoder.pt")

    print("DECODER OUT:")
    print(output_tensors)
