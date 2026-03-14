import onnx
import onnxruntime as ort
import numpy as np
import json
import os
import onnx
from collections import deque, defaultdict
from onnx import helper, shape_inference, numpy_helper, TensorProto
from inject_ops import create_quantized_fault_injection,  create_random_bitflip_injection, create_random_fault_injection, create_input16_mask, create_weight16_mask, create_fp16_fault_injection, create_random_bitflip_fp32
from typing import List
from itertools import chain
import numpy as np
from inject_utils.utils import delta_init
from axes_parser import patch_reduce_ops, move_initializers_to_constant_for_matmul

def print_node_info(node, description="Node"):
    """Print detailed information about a node's inputs and outputs"""
    print(f"\n{description}: {node.name}")
    print("OPERATION:", node.op_type)
    print("INPUTS (receives):")
    for i, input_name in enumerate(node.input):
        print(f"  Input {chr(65+i)}")
        print(f"  name: {input_name}")
    print("OUTPUTS:")
    for i, output_name in enumerate(node.output):
        print(f"  Output {chr(67+i)}")
        print(f"  name: {output_name}")
    print()

def analyze_path(model, start_pattern, target_config):
    """
    Find a path from nodes matching start_pattern to the target node with output matching target_config.
    
    Args:
        model: The ONNX model
        start_pattern: Pattern to match for source nodes' outputs
        target_config: Identifier for target node (can be node name or output tensor name)
    
    Returns:
        Tuple of (source_node, target_node, path, external_inputs) or None if no path found
    """
    # Match target by output tensor name OR node name
    target_node = None
    for node in model.graph.node:
        # Check if target_config matches any of the node's outputs
        if any(output == target_config for output in node.output):
            target_node = node
            print(f"Found target node by output tensor: {target_config}")
            break
        # Fall back to traditional name matching
        elif (node.name == target_config or 
             (node.name and target_config in node.name) or 
             (target_config.isdigit() and node.name and node.name.endswith(target_config))):
            target_node = node
            print(f"Found target node by name: {node.name}")
            break
    
    if target_node is None:
        print(f"Could not find target node with output or name '{target_config}'")
        for node in model.graph.node:
            if node.op_type == "MatMul":
                print(f"Available MatMul node - outputs: {node.output}, name: '{node.name}'")
        return None
    
    # Get the operation type from the found node
    allowed_op_type = target_node.op_type
    node_id = target_node.name if target_node.name else f"unnamed (output: {target_node.output[0]})"
    print(f"Found target node: {node_id} with op_type: {allowed_op_type}")
    
    # Build consumer map
    consumers = defaultdict(list)
    for node in model.graph.node:
        for inp in node.input:
            consumers[inp].append(node)
    
    # Find source nodes matching the start pattern in outputs
    source_nodes = []
    for node in model.graph.node:
        for output in node.output:
            if start_pattern == output or start_pattern in output:
                source_nodes.append(node)
                break
    
    if not source_nodes:
        print(f"Could not find any source nodes with output matching '{start_pattern}'")
        return None
    
    print(f"Found {len(source_nodes)} potential source nodes")
    
    # Path finding (DFS)
    for src_node in source_nodes:
        src_id = src_node.name if src_node.name else f"unnamed (output: {src_node.output[0]})"
        print(f"Searching for path from {src_id}")
        
        visited = set()
        stack = [(src_node.output[0], [src_node])]
        max_depth = 20  # Reasonable limit
        
        while stack:
            current_tensor, path = stack.pop()
            if current_tensor in visited:
                continue
            visited.add(current_tensor)
            
            if len(path) > max_depth:
                continue
            
            for consumer in consumers.get(current_tensor, []):
                if consumer == target_node:
                    external_inputs = []
                    for node in path + [consumer]:
                        if node.op_type == 'Mul':
                            external_inputs.extend(
                                inp for inp in node.input if inp not in {n.output[0] for n in path + [consumer]}
                            )
                    print(f"Found path from {src_id} to {node_id}, length: {len(path) + 1}")
                    return (src_node, consumer, path + [consumer], external_inputs)
                
                new_path = path + [consumer]
                for out in consumer.output:
                    stack.append((out, new_path))
    
    print(f"No path found from any source nodes to target")
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
                fp16=True,
                is_signed=True
            )
        elif llama_config['precision'] == 'int4':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp16=True,
                is_signed=False,
            )    
        elif llama_config['precision'] == 'float16':
            injection_nodes = create_fp16_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp32=False
            )
            
    else:
        if llama_config['precision'] == 'int8':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp16=False,
                is_signed=True
            )
        elif llama_config['precision'] == 'int4':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp16=False,
                is_signed=False,
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
    print("target node:", target_output_node)
    print(f"Original target: {original_target_output}")
    print(f"Cloned target: {cloned_target_output}")
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
    print_node_info(add_node, description="Add Node")
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
    model.opset_import[0].version = 18
    if llama_config['precision'] == 'float16':
        existing_opsets = {op.domain: op.version for op in model.opset_import}
        print('existing_opsets', existing_opsets)
        if 'custom.perturb' not in existing_opsets:
            model.opset_import.append(helper.make_opsetid('custom.perturb', 1))
        model = shape_inference.infer_shapes(model)
    else:
        model = shape_inference.infer_shapes(model)
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
    print(full_path)

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
                fp16=True,
                is_signed=True
            )
        elif llama_config['precision'] == 'int4':
            injection_nodes = create_quantized_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp16=True,
                is_signed=False
            )
        elif llama_config['precision'] == 'float16':
            injection_nodes = create_fp16_fault_injection(
                input_name=src_node.output[0],
                output_name=tensor_map[src_node.output[0]],
                bit_position=bit_position,
                fp32=False
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
    print_node_info(add_node, description="Add Node")
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
    
    
    model.opset_import[0].version = 18
    if llama_config['precision'] == 'float16':
        existing_opsets = {op.domain: op.version for op in model.opset_import}
        if 'custom.perturb' not in existing_opsets:
            model.opset_import.append(helper.make_opsetid('custom.perturb', 1))
        model = shape_inference.infer_shapes(model)
    else:
        model = shape_inference.infer_shapes(model)
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
            injection_nodes = create_random_fault_injection(
                output_name=target_output,
                random_value=value,
                fp16=True
            )
        else:
            value = delta_init(is_float32=True)
            injection_nodes = create_random_fault_injection(
                output_name=target_output,
                random_value=value,
                fp16=False
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

def run_single_decoder_with_fault(config, fault_model='INPUT', bit_position=3):
    """
    Run fault injection on a decoder model and perform inference with dummy inputs.
    
    Args:
        config: Dictionary containing model configuration
        fault_model: The fault model to use (INPUT, WEIGHT, INPUT16, WEIGHT16)
        bit_position: The bit position for fault injection
    """
    # Make sure model_name is set correctly
    if "decoder_path" in config and "model_name" not in config:
        config["model_name"] = config["decoder_path"]
    
    # Configuration for the model
    llama_config = {
        "fp16": True,  # Based on model path containing "fp16"
        "precision": "float16"
    }
    
    # Set up output path
    output_dir = "fault_injection_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{os.path.basename(config['model_name']).replace('.onnx', '')}_{fault_model}_injected.onnx")
    config["output_path"] = output_path
    
    print(f"Applying {fault_model} fault injection with bit position {bit_position}...")
    
    # Apply fault injection based on the model type
    try:
        if fault_model in ['INPUT', 'INPUT16']:
            injected_model_path = modify_onnx_graph_input(config, llama_config, fault_model, bit_position)
        elif fault_model in ['WEIGHT', 'WEIGHT16']:
            injected_model_path = modify_onnx_graph_weight(config, llama_config, fault_model, bit_position)
        else:
            injected_model_path = modify_onnx_graph_random(config, llama_config, fault_model, bit_position)
        
        print(f"Created fault-injected model: {injected_model_path}")
    except Exception as e:
        print(f"Error during fault injection: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create dummy input data
    N = 40           # Sequence length for current tokens
    lastN = 5        # Previous sequence length
    totalN = N + lastN  # Total sequence length
    
    # Set random seed for reproducibility
    np.random.seed(bit_position + 42)
    
    # Create attention mask (causal attention pattern)
    attn_mask = np.zeros((1, 1, N, totalN), dtype=np.float16)
    for i in range(N):
        attn_mask[0, 0, i, :lastN+i+1] = 1.0
    
    # Create position ids
    position_ids = np.arange(lastN, lastN + N, dtype=np.int64).reshape(1, N)
    
    # Create input tensors
    inputs = {
        'hidden_in': np.random.rand(1, N, 4096).astype(np.float16),
        'attn_mask': attn_mask,
        'position_ids': position_ids,
        'past_key_in': np.random.rand(1, 32, lastN, 128).astype(np.float16),
        'past_value_in': np.random.rand(1, 32, lastN, 128).astype(np.float16)
    }
    
    # Run inference on the fault-injected model
    try:
        print(f"Creating ONNX Runtime session for {injected_model_path}")
        providers = ['CUDAExecutionProvider']
        perturb_lib_path = "llama/onnx_perturb.so"
        custom_op_lib_path = "llama/onnx_bitflip.so"
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library(perturb_lib_path)
        sess_options.register_custom_ops_library(custom_op_lib_path)
        session = ort.InferenceSession(injected_model_path, sess_options, providers=providers)
        
        # Get all available outputs
        output_names = [output.name for output in session.get_outputs()]
        print(f"Available outputs: {output_names}")
        
        # Run inference
        print(f"Running inference with inputs: {list(inputs.keys())}")
        start_time = time.time()
        outputs = session.run(output_names, inputs)
        end_time = time.time()
        
        # Create dictionary with outputs
        output_dict = {name: tensor for name, tensor in zip(output_names, outputs)}
        
        # Print output information
        print(f"Inference completed in {end_time - start_time:.3f} seconds")
        print("\nOutput summary:")
        for name, tensor in output_dict.items():
            print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
            if "target_layer_output" in name or "fault_injected" in name:
                nonzeros = np.count_nonzero(tensor)
                print(f"    - Non-zero elements: {nonzeros}")
                print(f"    - Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean()}")
                
                # Save important outputs to file
                output_file = os.path.join(output_dir, f"{os.path.basename(config['model_name'])}_{fault_model}_{name}.npy")
                np.save(output_file, tensor)
                print(f"    - Saved to {output_file}")
        
        return output_dict
    
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import time
    
    # Configuration for fault injection
    config = {
    "target_layer": "131",
    "input_tensor": "/input_layernorm/Mul_1_output_0",
    "weight_tensor": "130",
    "model_name": "decoders/fp16/decoder-merge-20-fp16.onnx"
}
    # Available fault models: INPUT, WEIGHT, INPUT16, WEIGHT16
    fault_model = "INPUT"
    bit_position = 3
    
    # Run fault injection and inference
    results = run_single_decoder_with_fault(config, fault_model, bit_position)
    
    if results is not None:
        print("\nFault injection and inference completed successfully!")
        
        # Additional analysis if needed
        if 'target_layer_output' in results:
            fault_tensor = results['target_layer_output']
            print(f"\nFault analysis for target layer output:")
            print(f"  - Shape: {fault_tensor.shape}")
            print(f"  - Non-zero count: {np.count_nonzero(fault_tensor)}")
            print(f"  - Mean absolute value: {np.abs(fault_tensor).mean()}")
            print(f"  - Max absolute value: {np.abs(fault_tensor).max()}")
            print(np.linalg.norm(fault_tensor))
    else:
        print("\nFault injection or inference failed.")