import onnx
from collections import defaultdict
from onnx import helper, shape_inference, numpy_helper, TensorProto
from inject_ops import (create_quantized_fault_injection, create_random_bitflip_injection, 
                        create_random_fault_injection, create_quantized_fault_injection_weight,
                        create_input16_mask, create_weight16_mask, create_fp16_fault_injection,
                        create_fp16_fault_injection_weight)
import numpy as np
from typing import Dict, Any, List, Set
from axes_parser import patch_reduce_ops, move_initializers_to_constant_for_matmul

def find_node_by_output(graph, output_name: str):
    for node in graph.graph.node:
        if output_name in node.output:
            return node
    return None

def find_nodes_consuming(graph, tensor_name: str) -> List:
    consumers = []
    for node in graph.graph.node:
        if tensor_name in node.input:
            consumers.append(node)
    return consumers

def modify_onnx_graph_input_fp16(config: Dict[str, Any], fault_model: str, bit_position: int = 3, precision: str = 'float16'):
    model_path = config["decoder_path"]
    output_path = config.get("output_path", model_path.replace(".onnx", "_injected.onnx"))

    model = onnx.load(model_path)
    model = patch_reduce_ops(model, reduce_ops=("ReduceMean"))
    model = shape_inference.infer_shapes(model)
    
    
    input_tensor = config["input_tensor"]
    weight_tensor = config["weight_tensor"]
    output_tensor = config["output_tensor"]
    
    
    input_producer = find_node_by_output(model, input_tensor)
    if not input_producer:
        raise ValueError(f"Could not find a node that produces '{input_tensor}'")
    
    
    matmul_node = find_node_by_output(model, output_tensor)
    if not matmul_node or matmul_node.op_type != "MatMul":
        raise ValueError(f"Could not find MatMul node that produces '{output_tensor}'")
    
   
    faulty_input = f"{input_tensor}_faulty"
    
    if precision == 'int8':
        injection_nodes = create_quantized_fault_injection(
            input_name=input_tensor,
            output_name=faulty_input,
            bit_position=bit_position
        )
    elif precision == 'float16':
        injection_nodes = create_fp16_fault_injection(
            input_name=input_tensor,
            output_name=faulty_input,
            bit_position=bit_position
        )
    
    
    faulty_matmul_output = f"{output_tensor}_fault"
    faulty_matmul_node = helper.make_node(
        "MatMul",
        inputs=[faulty_input, weight_tensor],
        outputs=[faulty_matmul_output],
        name=f"{matmul_node.name}_fault"
    )
    
   
    if "16" in fault_model:
        mask_nodes = create_input16_mask(
            matmul_output=faulty_matmul_output,
            masked_output=f"{faulty_matmul_output}_masked",
            block_length=16
        )
        faulty_matmul_output = f"{faulty_matmul_output}_masked"
    else:
        mask_nodes = []
    
    
    final_output = f"{output_tensor}_final"
    add_node = helper.make_node(
        "Add",
        inputs=[output_tensor, faulty_matmul_output],
        outputs=[final_output],
        name=f"{matmul_node.name}_Add"
    )
    
   
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == output_tensor and node != add_node:
                node.input[i] = final_output
    
   
    new_nodes = list(model.graph.node)
    
   
    insert_pos = next((i for i, n in enumerate(new_nodes) if n == input_producer), 0) + 1
    
    
    new_nodes = (
        new_nodes[:insert_pos] +
        injection_nodes +
        [faulty_matmul_node] +
        mask_nodes +
        [add_node] +
        new_nodes[insert_pos:]
    )
    
   
    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)
    
    
    model.opset_import[0].version = 18
    if precision == 'float16':
        existing_opsets = {op.domain: op.version for op in model.opset_import}
        if 'custom.bitflip' not in existing_opsets:
            model.opset_import.append(helper.make_opsetid('custom.bitflip', 1))
    
    onnx.save(model, output_path)
    print(f"Modified input injection model saved to {output_path}")
    return output_path

def modify_onnx_graph_weight_fp16(config: Dict[str, Any], fault_model: str, bit_position: int = 3, precision: str = 'float16'):
    model_path = config["decoder_path"]
    output_path = config.get("output_path", model_path.replace(".onnx", "_injected.onnx"))
    
    
    model = onnx.load(model_path)
    model = patch_reduce_ops(model, reduce_ops=("ReduceMean"))
    model = move_initializers_to_constant_for_matmul(model)
    
    input_tensor = config["input_tensor"]
    weight_tensor = config["weight_tensor"]
    output_tensor = config["output_tensor"]
    
   
    matmul_node = find_node_by_output(model, output_tensor)
    if not matmul_node or matmul_node.op_type != "MatMul":
        raise ValueError(f"Could not find MatMul node that produces '{output_tensor}'")
    
    
    weight_producer = find_node_by_output(model, weight_tensor)
    if not weight_producer:
        weight_initializer = None
        for init in model.graph.initializer:
            if init.name == weight_tensor:
                weight_initializer = init
                break
        
        if not weight_initializer:
            raise ValueError(f"Could not find weight tensor '{weight_tensor}' in graph")
    
    faulty_weight = f"{weight_tensor}_faulty"
    
    if precision == 'int8':
        injection_nodes = create_quantized_fault_injection_weight(
            input_name=weight_tensor,
            output_name=faulty_weight,
            bit_position=bit_position
        )
    elif precision == 'float16':
        injection_nodes = create_fp16_fault_injection_weight(
            input_name=weight_tensor,
            output_name=faulty_weight,
            bit_position=bit_position
        )
    
    faulty_matmul_output = f"{output_tensor}_fault"
    faulty_matmul_node = helper.make_node(
        "MatMul",
        inputs=[input_tensor, faulty_weight],
        outputs=[faulty_matmul_output],
        name=f"{matmul_node.name}_fault"
    )
    
    if "16" in fault_model:
        size = np.random.randint(1, 16)
        mask_nodes = create_weight16_mask(
            matmul_output=faulty_matmul_output,
            masked_output=f"{faulty_matmul_output}_masked",
            block_length=size
        )
        faulty_matmul_output = f"{faulty_matmul_output}_masked"
    else:
        mask_nodes = []
    
    final_output = f"{output_tensor}_final"
    add_node = helper.make_node(
        "Add",
        inputs=[output_tensor, faulty_matmul_output],
        outputs=[final_output],
        name=f"{matmul_node.name}_Add"
    )
    
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == output_tensor and node != add_node:
                node.input[i] = final_output
    
    new_nodes = []
    injection_added = False
    
    for node in model.graph.node:
        new_nodes.append(node)
        if node == weight_producer and not injection_added:
            new_nodes.extend(injection_nodes)
            injection_added = True
    
    if not injection_added:
        new_nodes = injection_nodes + new_nodes
    
    new_nodes.extend([faulty_matmul_node] + mask_nodes + [add_node])

    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)
    
    model.opset_import[0].version = 18
    if precision == 'float16':
        existing_opsets = {op.domain: op.version for op in model.opset_import}
        if 'custom.bitflip' not in existing_opsets:
            model.opset_import.append(helper.make_opsetid('custom.bitflip', 1))
    # model = shape_inference.infer_shapes(model)
    onnx.save(model, output_path)
    print(f"Modified weight injection model saved to {output_path}")
    return output_path

def modify_onnx_graph_random_fp16(config: Dict[str, Any], fault_model: str, bit_position: int = 3):
    model_path = config["decoder_path"]
    output_path = config.get("output_path", model_path.replace(".onnx", "_injected.onnx"))
    
    model = onnx.load(model_path)
    model = patch_reduce_ops(model, reduce_ops=("ReduceMean"))
   

    output_tensor = config["output_tensor"]
    
    matmul_node = find_node_by_output(model, output_tensor)
    if not matmul_node or matmul_node.op_type != "MatMul":
        raise ValueError(f"Could not find MatMul node that produces '{output_tensor}'")
    
    faulty_output = f"{output_tensor}_faulty"
    
    if "BITFLIP" in fault_model:
        injection_nodes = create_random_bitflip_injection(
            output_name=output_tensor,
            bit_position=bit_position
        )
    else:
        from inject_utils.utils import delta_init
        injection_nodes = create_random_fault_injection(
            output_name=output_tensor,
            random_value=delta_init(is_float32=False)
        )
    
    consumers = find_nodes_consuming(model, output_tensor)
    new_nodes = []
    for node in model.graph.node:
        if node == matmul_node:
            new_nodes.append(node)
            new_nodes.extend(injection_nodes)
        elif node in consumers:
            new_inputs = [
                faulty_output if inp == output_tensor else inp
                for inp in node.input
            ]
            new_node = helper.make_node(
                node.op_type,
                new_inputs,
                node.output,
                node.name
            )
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)
    
    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)
    
    model.opset_import[0].version = 18
    if 'BITFLIP' in fault_model:
        existing_opsets = {op.domain: op.version for op in model.opset_import}
        if 'custom.bitflip' not in existing_opsets:
            model.opset_import.append(helper.make_opsetid('custom.bitflip', 1))
    model = shape_inference.infer_shapes(model)
    onnx.save(model, output_path)
    print(f"Modified random fault injection model saved to {output_path}")
    return output_path

if __name__ == "__main__":
    config = {
    "node_id": "/self_attn/MatMul_1",
    "input_tensor": "/self_attn/Softmax_output_0",
    "weight_tensor": "past_value",
    "output_tensor": "/self_attn/MatMul_1_output_0",
    "decoder_path": "decoders/fp16/decoder-merge-20-patched.onnx"
}
    fault_model = "WEIGHT"  # or "WEIGHT16"
    bit_position = 9
    modify_onnx_graph_input_fp16(config, fault_model, bit_position)
#     # print('doing injection')
    # modify_onnx_graph_weight_fp16(config, fault_model, bit_position)
    # modify_onnx_graph_random_fp16(config, fault_model, bit_position)