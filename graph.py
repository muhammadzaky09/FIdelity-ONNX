import struct
import onnx
import onnx_graphsurgeon as gs
from collections import deque, defaultdict
from onnx import helper, shape_inference, numpy_helper, TensorProto
from inject_ops import create_quantized_fault_injection, create_random_bitflip_injection, create_random_fault_injection, create_input16_mask, create_weight16_mask, create_fp16_fault_injection, create_random_bitflip_fp32, create_conv_input16_mask, create_conv_weight16_mask
from typing import List, Dict, Set
from itertools import chain
import numpy as np
from axes_parser import patch_reduce_ops, move_initializers_to_constant_for_matmul

def bin2fp32(bin_str):
    assert len(bin_str) == 32
    data = struct.unpack('!f',struct.pack('!I', int(bin_str, 2)))[0]
    if np.isnan(data):
        return 0
    else:
        return data
    
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

def _is_fp16_tensor(graph: gs.Graph, tensor_name: str):
    """Return True if *tensor_name* has FLOAT16 element type in *graph*."""
    tensors = graph.tensors()
    if tensor_name in tensors:
        tensor = tensors[tensor_name]
        if hasattr(tensor, 'dtype') and tensor.dtype is not None:
            return tensor.dtype == np.float16
        # gs.Constant stores data in .values (a numpy array)
        if hasattr(tensor, 'values') and tensor.values is not None:
            return tensor.values.dtype == np.float16
    return False

def analyze_paths_gs(graph: gs.Graph, target_layer: str, input_tensor_name: str, weight_tensor_name: str = None):
    """Find paths from input/weight tensors to target layer using GraphSurgeon."""
    
    # Find nodes
    src_node = None
    weight_node = None
    target_node = None
    
    # Build producer and consumer maps for efficiency (O(n) instead of O(n²))
    producers = {}  # tensor_name -> node that produces it
    consumers = defaultdict(list)  # tensor_name -> list of nodes that consume it
    
    for node in graph.nodes:
        # Check outputs for source nodes
        for output in node.outputs:
            producers[output.name] = node  # Use tensor name as key
            if output.name == input_tensor_name:
                src_node = node
            if weight_tensor_name and output.name == weight_tensor_name:
                weight_node = node
        
        # Build consumer map
        for inp in node.inputs:
            if not isinstance(inp, gs.Constant):
                consumers[inp.name].append(node)  # Use tensor name as key
        
        # Check for target node — match by node name OR by output tensor name
        if node.name == target_layer or any(o.name == target_layer for o in node.outputs):
            target_node = node
    
    if not src_node or not target_node:
        return None, None

    # If weight_node was not found as a node output, it may be an initializer
    # (gs.Constant) consumed directly by the target node. Treat the target node
    # itself as a single-element path — modify_onnx_graph detects this below.
    if weight_tensor_name and weight_node is None:
        for inp in target_node.inputs:
            if inp.name == weight_tensor_name:
                weight_node = target_node
                break
    
    # Extract paths using GraphSurgeon's built-in graph traversal
    def extract_path(start_node: gs.Node, end_node: gs.Node) -> List[gs.Node]:
        """Extract all nodes on paths from start_node to end_node."""
        # Find all nodes that can reach end_node (backward search)
        can_reach_end = set()  # Set of node names
        visited = set()  # Set of node names
        queue = deque([end_node])
        
        while queue:
            node = queue.popleft()
            if node.name in visited:
                continue
            visited.add(node.name)
            can_reach_end.add(node.name)
            
            # Add all nodes that produce inputs to this node
            for inp in node.inputs:
                if not isinstance(inp, gs.Constant) and inp.name in producers:
                    producer = producers[inp.name]  # Use tensor name to lookup
                    if producer.name not in visited:
                        queue.append(producer)
        
        # Forward search from start_node, only following paths that reach end_node
        path_nodes = []
        visited = set()  # Set of node names
        queue = deque([start_node])
        
        while queue:
            node = queue.popleft()
            if node.name in visited or node.name not in can_reach_end:
                continue
            visited.add(node.name)
            path_nodes.append(node)
            
            # Follow outputs that lead to nodes in can_reach_end
            for output in node.outputs:
                # Use the pre-built consumer map
                for consumer in consumers.get(output.name, []):  # Use tensor name to lookup
                    if consumer.name in can_reach_end and consumer.name not in visited:
                        queue.append(consumer)
        
        # Sort by original order in graph
        node_indices = {node.name: i for i, node in enumerate(graph.nodes)}
        path_nodes.sort(key=lambda n: node_indices.get(n.name, float('inf')))
        
        return path_nodes
    
    input_path = extract_path(src_node, target_node) if src_node else None
    weight_path = extract_path(weight_node, target_node) if weight_node else None
    
    return input_path, weight_path


def modify_onnx_graph(config,
                      model_config,
                      fault_model: str,
                      bit_position: int = 3):
    model_path = config["model_name"]
    default_tag = "_injected.onnx"
    output_path = config.get("output_path", model_path.replace(".onnx", default_tag))
    
    # Load model and apply patches
    model = onnx.load(model_path)
    model = patch_reduce_ops(model, reduce_ops=("ReduceMean", "ReduceMax"))
    prec = model_config['precision']
    # Convert to GraphSurgeon graph
    graph = gs.import_onnx(model)
    tgt_fp16 = False  # set to True once we know the target tensor is FP16
    
    if 'RANDOM' in fault_model:
        # -------------------- RANDOM / BITFLIP ----------------------
        tgt_pat = config["target_layer"]
        tgt_node = None
        
        # Find target node: match by node name substring OR by output tensor name
        for node in graph.nodes:
            if node.op in {"MatMul", "Conv", "Linear", "FullyConnected"} and \
               (tgt_pat in node.name or any(o.name == tgt_pat for o in node.outputs)):
                tgt_node = node
                break
        
        if tgt_node is None:
            raise ValueError(f"Target node with pattern {tgt_pat} not found")
        
        tgt_out = tgt_node.outputs[0]
        tgt_out_name = tgt_out.name
        
        # Check if FP16
        fp16_flag = _is_fp16_tensor(graph, tgt_out_name)
        tgt_fp16 = fp16_flag
        
        # Create injection nodes
        if "BITFLIP" in fault_model:
            if fp16_flag:
                # BitFlip (custom.bitflip) copies the full tensor and flips one bit
                # at fault_index — bit-exact, no rounding. Index supplied externally
                # for reproducibility and systematic sweeping.
                rand_idx_var = gs.Variable(name="rand_idx_inject", dtype=np.int64, shape=[])
                graph.inputs.append(rand_idx_var)
                injection_nodes = create_random_bitflip_injection(tgt_out_name, bit_position,
                                                                  fp16=True,
                                                                  rand_idx_name="rand_idx_inject")
            else:
                # fp32: DirectBitToggleFp32 (Python custom op) with external rand index.
                rand_idx_var = gs.Variable(name="rand_idx_inject", dtype=np.int64, shape=[])
                graph.inputs.append(rand_idx_var)
                injection_nodes = create_random_bitflip_fp32(tgt_out_name, bit_position,
                                                             rand_idx_name="rand_idx_inject")
        else:
            rand_idx_var = gs.Variable(name="rand_idx_inject", dtype=np.int64, shape=[])
            graph.inputs.append(rand_idx_var)
            rnd_val = delta_init(is_float32=not fp16_flag)
            injection_nodes = create_random_fault_injection(tgt_out_name, rnd_val,
                                                            fp16=fp16_flag,
                                                            rand_idx_name="rand_idx_inject")

        faulty_out_name = f"{tgt_out_name}_faulty"

        gs_injection_nodes = []
        tensor_map = graph.tensors()  # includes rand_idx_inject added above
        new_vars = {}  # name → gs.Variable, for injection-internal tensors

        def _get_var(name):
            if name in tensor_map:
                return tensor_map[name]
            if name not in new_vars:
                new_vars[name] = gs.Variable(name=name)
            return new_vars[name]

        for i, inj_node in enumerate(injection_nodes):
            inputs  = [_get_var(n) for n in inj_node.input]
            outputs = [_get_var(n) for n in inj_node.output]

            for out_name, out_var in zip(inj_node.output, outputs):
                if out_name == faulty_out_name:
                    faulty_tensor = out_var
                    # Inherit dtype/shape from tgt_out so gs.export_onnx can
                    # write a valid ValueInfoProto when this becomes a graph output.
                    faulty_tensor.dtype = tgt_out.dtype
                    faulty_tensor.shape = tgt_out.shape

            attrs = {attr.name: onnx.helper.get_attribute_value(attr)
                     for attr in inj_node.attribute}

            gs_node = gs.Node(
                op=inj_node.op_type,
                name=f"inj_{inj_node.op_type}_{i}",
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                domain=inj_node.domain if inj_node.domain else None,
            )
            gs_injection_nodes.append(gs_node)
        
        # Insert injection nodes after target node
        target_idx = graph.nodes.index(tgt_node)
        for i, node in enumerate(gs_injection_nodes):
            graph.nodes.insert(target_idx + 1 + i, node)

        # Re-wire all original consumers of tgt_out to use faulty_tensor.
        # Injection nodes are excluded — they intentionally read from tgt_out,
        # and re-wiring them would create a cycle (faulty → inject → faulty).
        injection_node_ids = {id(n) for n in gs_injection_nodes}
        for node in graph.nodes:
            if id(node) in injection_node_ids:
                continue
            for i, inp in enumerate(node.inputs):
                if inp == tgt_out:
                    node.inputs[i] = faulty_tensor

        # Also re-wire graph.outputs: if tgt_out is a graph output (e.g. the
        # MatMul is the last node), cleanup() would discard the injection nodes
        # because faulty_tensor would be unreachable from any graph output.
        for i, out in enumerate(graph.outputs):
            if out == tgt_out:
                graph.outputs[i] = faulty_tensor

    else:
        # -------------------- INPUT / WEIGHT variants ---------------
        inp_path, wgt_path = analyze_paths_gs(
            graph, config["target_layer"], config["input_tensor"], config.get("weight_tensor"))
        
        mapping = {
            "INPUT":    (inp_path, None),
            "INPUT16":  (inp_path, create_input16_mask),
            "WEIGHT":   (wgt_path, None),
            "WEIGHT16": (wgt_path, create_weight16_mask),
        }
        
        path, mask_helper = mapping[fault_model]
        if not path:
            raise ValueError("Could not find a path matching the given patterns.")
        
        src_node, tgt_node = path[0], path[-1]
        clone_suffix = "_fault_injected"
        orig_tgt_out = tgt_node.outputs[0]
        orig_tgt_out_name = orig_tgt_out.name

        # Detect initializer-weight case: single-node path where src == tgt.
        # The "source" is a gs.Constant (initializer) with no producer node.
        initializer_src = (src_node is tgt_node)
        
        # Clone the path
        cloned_nodes = []
        tensor_mapping = {}
        created_tensors = {}  # Keep track of created tensors
        existing_tensors = graph.tensors()  # Get existing tensors
        
        # Map source output — for initializer weights, use the weight tensor name directly
        if initializer_src:
            src_out_name = config.get("weight_tensor") or config.get("input_tensor")
        else:
            src_out_name = src_node.outputs[0].name
        cloned_src_out_name = f"{src_out_name}{clone_suffix}"
        tensor_mapping[src_out_name] = cloned_src_out_name
        
        # Clone each node in the path.
        # For initializer sources, clone the whole path (no source node to skip).
        for node in (path if initializer_src else path[1:]):
            # Map inputs
            cloned_inputs = []
            for inp in node.inputs:
                if isinstance(inp, gs.Constant):
                    if inp.name in tensor_mapping:
                        # This constant IS the weight we're perturbing — replace with variable
                        mapped_name = tensor_mapping[inp.name]
                        if mapped_name not in created_tensors:
                            created_tensors[mapped_name] = gs.Variable(name=mapped_name)
                        cloned_inputs.append(created_tensors[mapped_name])
                    else:
                        cloned_inputs.append(inp)  # unrelated constant — keep as-is
                else:
                    inp_name = inp.name
                    if inp_name in tensor_mapping:
                        # Create or get the mapped tensor
                        mapped_name = tensor_mapping[inp_name]
                        if mapped_name not in created_tensors:
                            created_tensors[mapped_name] = gs.Variable(name=mapped_name)
                        cloned_inputs.append(created_tensors[mapped_name])
                    else:
                        cloned_inputs.append(inp)
            
            # Map outputs
            cloned_outputs = []
            for out in node.outputs:
                cloned_out_name = f"{out.name}{clone_suffix}"
                tensor_mapping[out.name] = cloned_out_name
                if cloned_out_name not in created_tensors:
                    created_tensors[cloned_out_name] = gs.Variable(name=cloned_out_name)
                cloned_outputs.append(created_tensors[cloned_out_name])
            
            # Clone node
            cloned_node = gs.Node(
                op=node.op,
                name=f"{node.name}{clone_suffix}",
                inputs=cloned_inputs,
                outputs=cloned_outputs,
                attrs=node.attrs.copy()
            )
            cloned_nodes.append(cloned_node)
        
        fp16_flag = _is_fp16_tensor(graph, orig_tgt_out_name)
        tgt_fp16 = fp16_flag

        # rand_idx_inject: INT64 scalar input — caller supplies np.random.randint(0, N)
        # at each inference call so fault location varies across runs.
        rand_idx_var = gs.Variable(name="rand_idx_inject", dtype=np.int64, shape=[])
        graph.inputs.append(rand_idx_var)
        created_tensors["rand_idx_inject"] = rand_idx_var

        if prec == 'int8':
            inj_nodes = create_quantized_fault_injection(
                src_out_name, cloned_src_out_name, bit_position,
                fp16=fp16_flag, is_signed=True,
                rand_idx_name="rand_idx_inject")
        elif prec == 'int4':
            inj_nodes = create_quantized_fault_injection(
                src_out_name, cloned_src_out_name, bit_position,
                fp16=fp16_flag, is_signed=False,
                rand_idx_name="rand_idx_inject")
        elif prec == 'float16':
            inj_nodes = create_fp16_fault_injection(
                src_out_name, cloned_src_out_name, bit_position,
                fp32=not fp16_flag, rand_idx_name="rand_idx_inject")
        else:
            raise ValueError("Unsupported precision")
        
        gs_injection_nodes = []
        for i, inj_node in enumerate(inj_nodes):
            inputs = []
            for inp_name in inj_node.input:
                if inp_name in existing_tensors:
                    inputs.append(existing_tensors[inp_name])
                elif inp_name in created_tensors:
                    inputs.append(created_tensors[inp_name])
                else:
                    var = gs.Variable(name=inp_name)
                    created_tensors[inp_name] = var
                    inputs.append(var)
            
            outputs = []
            for out_name in inj_node.output:
                if out_name in existing_tensors:
                    outputs.append(existing_tensors[out_name])
                elif out_name in created_tensors:
                    outputs.append(created_tensors[out_name])
                else:
                    var = gs.Variable(name=out_name)
                    created_tensors[out_name] = var
                    outputs.append(var)
            
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) 
                     for attr in inj_node.attribute}
            
            gs_node = gs.Node(
                op=inj_node.op_type,
                name=inj_node.name if inj_node.name else f"{inj_node.op_type}_inj_{i}",
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                domain=inj_node.domain if inj_node.domain else None
            )
            gs_injection_nodes.append(gs_node)
        
        # Insert nodes after source node
        src_idx = graph.nodes.index(src_node)
        insert_pos = src_idx + 1
        
        # Insert injection nodes
        for i, node in enumerate(gs_injection_nodes):
            graph.nodes.insert(insert_pos + i, node)
        insert_pos += len(gs_injection_nodes)
        
        # Insert cloned nodes
        for i, node in enumerate(cloned_nodes):
            graph.nodes.insert(insert_pos + i, node)
        
        # Handle masking for "16" variants
        cloned_tgt_out_name = tensor_mapping[orig_tgt_out_name]
        if "16" in fault_model:
            if tgt_node.op == "Conv":
                if "INPUT" in fault_model:
                    m_nodes = create_conv_input16_mask(
                        cloned_tgt_out_name, f"{cloned_tgt_out_name}_masked", 16, fp16=fp16_flag)
                else:
                    m_nodes = create_conv_weight16_mask(
                        cloned_tgt_out_name, f"{cloned_tgt_out_name}_masked", 16, fp16=fp16_flag)
            else:
                # MatMul / linear path
                if "INPUT" in fault_model:
                    m_nodes = create_input16_mask(
                        cloned_tgt_out_name, f"{cloned_tgt_out_name}_masked", 16, fp16=fp16_flag)
                else:
                    m_nodes = create_weight16_mask(
                        cloned_tgt_out_name, f"{cloned_tgt_out_name}_masked", 16, fp16=fp16_flag)
            
            # Convert mask nodes to GraphSurgeon
            for mi, m_node in enumerate(m_nodes):
                inputs = []
                for inp_name in m_node.input:
                    if inp_name in existing_tensors:
                        inputs.append(existing_tensors[inp_name])
                    elif inp_name in created_tensors:
                        inputs.append(created_tensors[inp_name])
                    else:
                        var = gs.Variable(name=inp_name)
                        created_tensors[inp_name] = var
                        inputs.append(var)
                
                outputs = []
                for out_name in m_node.output:
                    if out_name not in created_tensors:
                        created_tensors[out_name] = gs.Variable(name=out_name)
                    outputs.append(created_tensors[out_name])
                
                attrs = {attr.name: onnx.helper.get_attribute_value(attr) 
                         for attr in m_node.attribute}
                
                gs_mask_node = gs.Node(
                    op=m_node.op_type,
                    name=m_node.name if m_node.name else f"{m_node.op_type}_mask_{mi}",
                    inputs=inputs,
                    outputs=outputs,
                    attrs=attrs
                )
                graph.nodes.append(gs_mask_node)
            
            cloned_tgt_out_name = f"{cloned_tgt_out_name}_masked"
        
        # Add final Add node
        final_out_name = f"{orig_tgt_out_name}_final"
        if final_out_name not in created_tensors:
            created_tensors[final_out_name] = gs.Variable(name=final_out_name)
            
        add_node = gs.Node(
            op='Add',
            name=f"{orig_tgt_out_name}_Add",
            inputs=[orig_tgt_out, created_tensors[cloned_tgt_out_name]],
            outputs=[created_tensors[final_out_name]]
        )
        graph.nodes.append(add_node)
        
        # Update all consumers of orig_tgt_out to use final output
        for node in graph.nodes:
            if node != add_node:
                for i, inp in enumerate(node.inputs):
                    if isinstance(inp, gs.Variable) and inp.name == orig_tgt_out_name:
                        node.inputs[i] = created_tensors[final_out_name]
    
    # Clean up and finalize
    graph.cleanup()
    graph.toposort()
    
    # Export back to ONNX
    model = gs.export_onnx(graph)
    
    # Update opset versions
    model.opset_import[0].version = 18
    exist = {o.domain: o.version for o in model.opset_import}
    if "BITFLIP" in fault_model:
        if tgt_fp16:
            # fp16 RANDOM_BITFLIP: BitFlip op from llama/onnx_bitflip.so
            if 'custom.bitflip' not in exist:
                model.opset_import.append(helper.make_opsetid('custom.bitflip', 1))
        else:
            # fp32 RANDOM_BITFLIP: DirectBitToggleFp32 Python custom op
            if 'ai.onnx.contrib' not in exist:
                model.opset_import.append(helper.make_opsetid('ai.onnx.contrib', 1))
    elif prec == 'float16' and 'RANDOM' not in fault_model:
        # INPUT / WEIGHT / INPUT16 / WEIGHT16 with float16 precision:
        # create_fp16_fault_injection now uses BitFlip (custom.bitflip)
        if 'custom.bitflip' not in exist:
            model.opset_import.append(helper.make_opsetid('custom.bitflip', 1))
    
    # Save the model
    onnx.save(model, output_path)
    print(f"Modified model saved to {output_path}")
    return output_path


# if __name__ == "__main__":
#     # config = {
#     #     "input_tensor": "/self_attn/q_proj/Round_output_0",
#     #     "target_layer": "/self_attn/v_proj/MatMul",
#     #     "weight_tensor": "/self_attn/v_proj/Round_output_0",
#     #     "model_name": "decoders/7B16/decoder-merge-8.onnx"
#     # }
#     # config = {
#     # "input_tensor": "/self_attn/q_proj/Round_2_output_0",
#     # "target_layer": "/self_attn/MatMul",
#     # "weight_tensor": "/self_attn/k_proj/Round_1_output_0",
#     # "model_name": "decoders/7B16/decoder-merge-8.onnx"
#     # }
    
#     config = {
#         "target_layer": "131",
#         "input_tensor": "/input_layernorm/Mul_1_output_0",
#         "weight_tensor": "130",
#         "model_name": "decoders/fp16/decoder-merge-20-fp16.onnx"
#     }
#     model_config = {
#         "precision": "float16"
#     }
#     fault_model = "RANDOM_BITFLIP"
#     bit_position = 7   # high-impact bit (MSB of exponent); survives INT8 quantization
#     modify_onnx_graph(config, model_config, fault_model, bit_position)



