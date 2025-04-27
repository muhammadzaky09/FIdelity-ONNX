import onnx
from collections import deque, defaultdict
from onnx import helper, shape_inference, numpy_helper, TensorProto
from inject_ops import create_quantized_fault_injection,  create_random_bitflip_injection, create_random_fault_injection, create_input16_mask, create_weight16_mask, create_fp16_fault_injection, create_random_bitflip_fp32
from typing import List
from itertools import chain
import numpy as np
from inject_utils.utils import delta_init
from axes_parser import patch_reduce_ops, move_initializers_to_constant_for_matmul
from loguru import logger
from llama.decoder import Decoder
from llama.memory_pool import MemoryPoolSimple
from llama.utils import npsoftmax, seeded_npmultinomial2D
from llama.logits_process import warp_temperature, warp_topp
from llama.tokenizer import Tokenizer
import argparse
from datasets import load_dataset
import re
import os
import random
from datetime import datetime
import gc
import csv
import json

def analyze_path(model, start_pattern, target_config):
    # Build consumer map lazily and only explore relevant paths
    consumers = defaultdict(list)
    allowed_op_type = target_config.split("/")[-1]
    
    # Only find source nodes matching the pattern
    source_nodes = [n for n in model.graph.node if any(start_pattern in output for output in n.output)]
    
    for src_node in source_nodes:
        # Perform iterative DFS rather than BFS for better memory locality
        visited = set()
        stack = [(src_node.output[0], [src_node])]
        
        while stack:
            current_tensor, path = stack.pop()
            if current_tensor in visited:
                continue
            visited.add(current_tensor)
            
            # Lazily build consumer map only for tensors we're visiting
            if current_tensor not in consumers:
                for node in model.graph.node:
                    if current_tensor in node.input:
                        consumers[current_tensor].append(node)
            
            # Check if any consumer is our target
            for consumer in consumers[current_tensor]:
                if consumer.op_type == allowed_op_type and target_config in consumer.name:
                    # Target found, collect external inputs and return
                    external_inputs = []
                    for node in path + [consumer]:
                        if node.op_type == 'Mul':
                            external_inputs.extend(
                                inp for inp in node.input if inp not in {n.output[0] for n in path + [consumer]}
                            )
                    return (src_node, consumer, path + [consumer], external_inputs)
                
                # If not target, add to stack for further exploration
                new_path = path + [consumer]
                for out in consumer.output:
                    stack.append((out, new_path))
    
    return None

def modify_onnx_graph_input(config, llama_config, fault_model, bit_position=3):
    model_path = config["model_name"]
    output_path = config.get("output_path", model_path.replace(".onnx", "_injected.onnx"))

    model = onnx.load(model_path)
    model = patch_reduce_ops(model, reduce_ops=("ReduceMean", "ReduceMax"))
    path_info = analyze_path(model, config["input_tensor"], config["target_layer"])
    print("Path info:", path_info)
    if path_info is None:
        raise ValueError("Could not find a path matching the given patterns.")
    src_node, target_node, full_path, external_inputs = path_info

    clone_suffix = "_fault_injected"
    original_target_output = target_node.output[0]

    tensor_map = {}
    cloned_nodes = []
  
    tensor_map[src_node.output[0]] = f"{src_node.output[0]}{clone_suffix}"
    for node in full_path[1:]:
        new_inputs = [tensor_map.get(inp, inp) for inp in node.input]
        new_outputs = [f"{out}{clone_suffix}" for out in node.output]
        cloned_node = helper.make_node( node.op_type,
                                       new_inputs,new_outputs,
                                       name=f"{node.name}{clone_suffix}",
                                       **{attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
                                       )
        cloned_nodes.append(cloned_node)
        for orig_out, new_out in zip(node.output, new_outputs):
            tensor_map[orig_out] = new_out
    if llama_config['fp16']:
        if llama_config['precision'] == 'int8':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp32=False,
                is_signed=True
            )
        elif llama_config['precision'] == 'int4':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp32=False,
                is_signed=False,
            )    
        elif llama_config['precision'] == 'float16':
            injection_nodes = create_fp16_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp32=False,
            )
            
    else:
        if llama_config['precision'] == 'int8':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp32=True,
                is_signed=True
            )
        elif llama_config['precision'] == 'int4':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp32=True,
                is_signed=False,
            )    
        elif llama_config['precision'] == 'float16':
            injection_nodes = create_fp16_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp32=True,
            )

    original_nodes = list(model.graph.node)
    insert_pos = next(i for i, n in enumerate(original_nodes) if n.name == src_node.name) + 1
    new_nodes = (
        original_nodes[:insert_pos] +
        injection_nodes +
        cloned_nodes +
        original_nodes[insert_pos:]
    )
    
    cloned_target_output = tensor_map[original_target_output]
    
    if "16" in fault_model:
        mask_nodes = create_input16_mask(
            matmul_output=cloned_target_output, 
            masked_output=f"{cloned_target_output}_masked",
            block_length=16
        )
        new_nodes.extend(mask_nodes)
        cloned_target_output = f"{cloned_target_output}_masked"
    target_output_node = helper.make_node(
        'Identity',
        inputs=[cloned_target_output],
        outputs=['target_layer_output'],
        name='target_layer_output_identity'
    )
    new_nodes.append(target_output_node)
    add_node = helper.make_node(
        'Add',
        [original_target_output, cloned_target_output],
        [f"{original_target_output}_final"],
        f"{original_target_output}_Add"
    )
    new_nodes.append(add_node)

    for node in new_nodes:
        if node != add_node and original_target_output in node.input:
            node.input[:] = [
                f"{original_target_output}_final" if inp == original_target_output else inp
                for inp in node.input
            ]

    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)
    
    model.graph.output.extend([
        helper.make_tensor_value_info(
            'target_layer_output',
            TensorProto.FLOAT16 if llama_config['fp16'] else TensorProto.FLOAT,
            None  # Shape will be inferred
        )
    ])

    # Clone any external initializers if needed
    for inp in external_inputs:
        if inp in [i.name for i in model.graph.initializer]:
            orig_init = next(i for i in model.graph.initializer if i.name == inp)
            cloned_init = numpy_helper.from_array(numpy_helper.to_array(orig_init), name=f"{inp}{clone_suffix}")
            model.graph.initializer.append(cloned_init)
    
    model = shape_inference.infer_shapes(model)
    model.opset_import[0].version = 18
    if llama_config['precision'] == 'float16':
        existing_opsets = {op.domain: op.version for op in model.opset_import}
        if 'custom.perturb' not in existing_opsets:
            model.opset_import.append(helper.make_opsetid('custom.perturb', 1))
    onnx.save(model, output_path)
    print(f"Modified model saved to {output_path}")
    return output_path

def modify_onnx_graph_weight(config, llama_config, fault_model, bit_position=3):
    model_path = config["model_name"]
    output_path = config.get("output_path", model_path.replace(".onnx", "_injected.onnx"))
    model = onnx.load(model_path)
    model = patch_reduce_ops(model, reduce_ops=("ReduceMean", "ReduceMax"))
    model = move_initializers_to_constant_for_matmul(model)
    print("Output path:", output_path)
    path_info = analyze_path(model, config["weight_tensor"], config["target_layer"])
    if path_info is None:
        raise ValueError(f"Could not find a weight path from '{config['weight_tensor']}' to target '{config['target_layer']}'.")
    src_node, target_node, full_path, external_inputs = path_info

    clone_suffix = "_fault_injected"
    original_target_output = target_node.output[0]

    # Clone the chain of nodes from the weight source to the target.
    tensor_map = {}
    cloned_nodes = []
    tensor_map[src_node.output[0]] = f"{src_node.output[0]}{clone_suffix}"
    for node in full_path[1:]:
        new_inputs = [tensor_map.get(inp, inp) for inp in node.input]
        new_outputs = [f"{out}{clone_suffix}" for out in node.output]
        cloned_node = helper.make_node(node.op_type,new_inputs, new_outputs, name=f"{node.name}{clone_suffix}",**{attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute})
        cloned_nodes.append(cloned_node)
        for orig_out, new_out in zip(node.output, new_outputs):
            tensor_map[orig_out] = new_out
    if llama_config['fp16']:
        if llama_config['precision'] == 'int8':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                is_signed=True
            )
        elif llama_config['precision'] == 'int4':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                is_signed=False
            )
        elif llama_config['precision'] == 'float16':
            injection_nodes = create_fp16_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position
            )
    else:
        if llama_config['precision'] == 'int8':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp32=True,
                is_signed=True
            )
        elif llama_config['precision'] == 'int4':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp32=True,
                is_signed=False
            )
        elif llama_config['precision'] == 'float16':
            injection_nodes = create_fp16_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp32=True
            )
    

    original_nodes = list(model.graph.node)
    insert_pos = next(i for i, n in enumerate(original_nodes) if n.name == src_node.name) + 1
    new_nodes = (
        original_nodes[:insert_pos] +
        injection_nodes +
        cloned_nodes +
        original_nodes[insert_pos:]
    )
    
    cloned_target_output = tensor_map[original_target_output]
    
    if "16" in fault_model:
        mask_nodes = create_weight16_mask(
            matmul_output=cloned_target_output,
            masked_output=f"{cloned_target_output}_masked",
            block_length=np.random.randint(1, 16)
        )
        new_nodes.extend(mask_nodes)
        cloned_target_output = f"{cloned_target_output}_masked"
    target_output_node = helper.make_node(
        'Identity',
        inputs=[cloned_target_output],
        outputs=['target_layer_output'],
        name='target_layer_output_identity'
    )
    new_nodes.append(target_output_node)
    add_node = helper.make_node(
        "Add",
        inputs=[original_target_output, cloned_target_output],
        outputs=[f"{original_target_output}_final"],
        name=f"{target_node.name}_Add"
    )
    new_nodes.append(add_node)
    
    for node in new_nodes:
        if node != add_node and original_target_output in node.input:
            node.input[:] = [
                f"{original_target_output}_final" if inp == original_target_output else inp
                for inp in node.input
            ]
    
    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    model.graph.output.extend([
        helper.make_tensor_value_info(
            'target_layer_output',
            TensorProto.FLOAT16 if llama_config['fp16'] else TensorProto.FLOAT,
            None  # Shape will be inferred
        )
    ])
    
    for inp in external_inputs:
        if inp in [i.name for i in model.graph.initializer]:
            orig_init = next(i for i in model.graph.initializer if i.name == inp)
            cloned_init = numpy_helper.from_array(numpy_helper.to_array(orig_init), name=f"{inp}{clone_suffix}")
            model.graph.initializer.append(cloned_init)
    
    model = shape_inference.infer_shapes(model)
    model.opset_import[0].version = 18
    if llama_config['precision'] == 'float16':
        existing_opsets = {op.domain: op.version for op in model.opset_import}
        if 'custom.perturb' not in existing_opsets:
            model.opset_import.append(helper.make_opsetid('custom.perturb', 1))
    onnx.save(model, output_path)
    print(f"Modified WEIGHT injection model saved to {output_path}")
    return output_path


def modify_onnx_graph_random(config, llama_config, fault_model, bit_position=None):
    model_path = config["model_name"]
    output_path = config.get("output_path", model_path.replace(".onnx", "_random.onnx"))
    target_pattern = config["target_layer"]

    model = onnx.load(model_path)
    model = patch_reduce_ops(model, reduce_ops=("ReduceMean", "ReduceMax"))

    target_node = None
    for node in model.graph.node:
        if node.op_type in {'MatMul', 'Linear', 'FullyConnected'} and target_pattern in node.name:
            target_node = node
            break
    if not target_node:
        raise ValueError(f"Target node with pattern {target_pattern} not found")

    target_output = target_node.output[0]
    consumers = defaultdict(list)
    for node in model.graph.node:
        for inp in node.input:
            consumers[inp].append(node)
    # (downstream_nodes not used further)

    if "BITFLIP" in fault_model:
        if llama_config['fp16']:
            injection_nodes = create_random_bitflip_injection(
                output_name=target_output,
                bit_position=bit_position
            )
        else:
            injection_nodes = create_random_bitflip_fp32(
                output_name=target_output,
                bit_position=bit_position,
            )
    else:
        if llama_config['fp16']:
            value = delta_init(is_float32=False)
        else:
            value = delta_init(is_float32=True)
        injection_nodes = create_random_fault_injection(
            output_name=target_output,
            random_value=value
        )
    
    new_nodes = []
    faulty_output = f"{target_output}_faulty"

    for node in model.graph.node:
        if node == target_node:
            new_nodes.append(node)  
            new_nodes.extend(injection_nodes)  
        else:
            if target_output in node.input:
                new_inputs = [
                    faulty_output if inp == target_output else inp 
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
        if 'ai.onnx.contrib' not in existing_opsets:
            model.opset_import.append(helper.make_opsetid('ai.onnx.contrib', 1))
    onnx.save(model, output_path)
    print(f"Modified random fault injection model saved to {output_path}")
    return output_path

if __name__ == "__main__":
    config = {
    "input_tensor": "/self_attn/o_proj/Round_output_0",  # Use the full output tensor name
    "target_layer": "/self_attn/o_proj/MatMul",
    "weight_tensor": "onnx::MatMul_431",
    "model_name": "decoders/decoder-merge-20.onnx"
}
    llama_config = {
        "fp16": True,
        "precision": "int8"
    }
    fault_model = "INPUT"  # or "WEIGHT16"
    bit_position = 3
    modify_onnx_graph_input(config, llama_config, fault_model, bit_position)