import onnx
from collections import deque, defaultdict
from onnx import helper, shape_inference, numpy_helper, TensorProto
from inject_ops import create_quantized_fault_injection,  create_random_bitflip_injection, create_random_fault_injection, create_input16_mask, create_weight16_mask, create_fp16_fault_injection, create_random_bitflip_fp32, create_conv_input16_mask, create_conv_weight16_mask
from typing import List
from itertools import chain
import numpy as np
from inject_utils.utils import delta_init
from axes_parser import patch_reduce_ops, move_initializers_to_constant_for_matmul
# CODE DEPRECATED
def _is_fp16_tensor(model, tensor_name: str):
    """Return True if *tensor_name* has FLOAT16 element type in *model*."""

    for collection in (model.graph.value_info, model.graph.output, model.graph.input):
        for value_info in collection:
            if value_info.name == tensor_name:
                elem_type = value_info.type.tensor_type.elem_type
                return elem_type == TensorProto.FLOAT16

    for initializer in model.graph.initializer:
        if initializer.name == tensor_name:
            return initializer.data_type == TensorProto.FLOAT16

    return False

def analyze_paths(model, target_layer, input_tensor_name, weight_tensor_name=None):
    # 1) find the three key nodes
    src_node = weight_node = target_node = None
    for node in model.graph.node:
        if input_tensor_name in node.output:
            src_node = node
        if weight_tensor_name and weight_tensor_name in node.output:
            weight_node = node
        if node.name == target_layer:
            target_node = node

    if not src_node or not target_node:
        return None, None

    # 2) build producer + consumer maps
    producers = {}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for out in node.output:
            producers[out] = node
        for inp in node.input:
            consumers[inp].append(node)

    # 3) extract *raw* subgraphs
    raw_input = _extract_subgraph(src_node, target_node, consumers, producers)
    raw_weight = None
    if weight_node:
        raw_weight = _extract_subgraph(weight_node, target_node, consumers, producers)

    # 4) now sort each subgraph by the original model order (which is topological)
    node_order = {id(n): i for i,n in enumerate(model.graph.node)}
    def topo_sort(raw):
        if not raw: 
            return None
        ids = {id(n) for n in raw}
        # keep only those in the raw set, in model‑decl order
        return [n for n in model.graph.node if id(n) in ids]

    return topo_sort(raw_input), topo_sort(raw_weight)


def _extract_subgraph(start_node, end_node, consumers, producers):
    # backward: find everything that can reach end_node
    back_ids = set()
    q = deque([end_node])
    while q:
        n = q.popleft()
        nid = id(n)
        if nid in back_ids:
            continue
        back_ids.add(nid)
        for inp in n.input:
            parent = producers.get(inp)
            if parent:
                q.append(parent)

    # forward: from start, only follow into back_ids
    sub_ids = set()
    sub_nodes = []
    q = deque([start_node])
    while q:
        n = q.popleft()
        nid = id(n)
        if nid in sub_ids or nid not in back_ids:
            continue
        sub_ids.add(nid)
        sub_nodes.append(n)
        for out in n.output:
            for c in consumers.get(out, []):
                q.append(c)

    return sub_nodes


def modify_onnx_graph(config,
                      llama_config,
                      fault_model: str,
                      bit_position: int = 3):
    model_path  = config["model_name"]
    default_tag = "_injected.onnx"
    output_path = config.get("output_path", model_path.replace(".onnx", default_tag))

    model  = onnx.load(model_path)
    model  = patch_reduce_ops(model, reduce_ops=("ReduceMean", "ReduceMax"))

    new_nodes = []  # will be populated in both branches

    if 'RANDOM' in fault_model:
        # -------------------- RANDOM / BITFLIP ----------------------
        tgt_pat = config["target_layer"]
        tgt_node = None
        for n in model.graph.node:
            if n.op_type in {"MatMul","Linear","FullyConnected"} and tgt_pat in n.name:
                tgt_node = n; break
        if tgt_node is None:
            raise ValueError(f"Target node with pattern {tgt_pat} not found")

        tgt_out = tgt_node.output[0]
        # build injection
        fp16_flag = _is_fp16_tensor(model, tgt_out)

        if "BITFLIP" in fault_model:
            injection_nodes = (create_random_bitflip_injection(tgt_out, bit_position)
                               if fp16_flag else
                               create_random_bitflip_fp32(tgt_out, bit_position))
        else:
            rnd_val = delta_init(is_float32=not fp16_flag)
            injection_nodes = create_random_fault_injection(tgt_out, rnd_val, fp16=fp16_flag)

        faulty_out = f"{tgt_out}_faulty"

        for n in model.graph.node:
            if n == tgt_node:
                new_nodes.append(n)
                new_nodes.extend(injection_nodes)
            else:
                if tgt_out in n.input:
                    n.input[:] = [faulty_out if x==tgt_out else x for x in n.input]
                new_nodes.append(n)
            
    else:
        # -------------------- INPUT / WEIGHT variants ---------------
        # Gather both paths once
        inp_path, wgt_path = analyze_paths(
            model, config["target_layer"], config["input_tensor"], config["weight_tensor"])

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
        orig_tgt_out = tgt_node.output[0]

        # clone chain
        tmap = {src_node.output[0]: f"{src_node.output[0]}{clone_suffix}"}
        for node in path[1:]:
            n_in  = [tmap.get(i,i) for i in node.input]
            n_out = [f"{o}{clone_suffix}" for o in node.output]
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            new_nodes.append(helper.make_node(node.op_type,n_in,n_out,name=f"{node.name}{clone_suffix}",**attrs))
            for o, n in zip(node.output, n_out):
                tmap[o]=n

        # injection at source
        fp16_flag = _is_fp16_tensor(model, orig_tgt_out)
        prec = llama_config['precision']
        def _qinj(signed):
            return create_quantized_fault_injection(src_node.output[0], tmap[src_node.output[0]], bit_position, fp16=fp16_flag, is_signed=signed)

        if prec=='int8': inj_nodes=_qinj(True)
        elif prec=='int4': inj_nodes=_qinj(False)
        elif prec=='float16': inj_nodes = create_fp16_fault_injection(src_node.output[0], tmap[src_node.output[0]], bit_position, fp32=not fp16_flag)
        else: raise ValueError("Unsupported precision")

        # splice
        orig_nodes=list(model.graph.node)
        insert_pos = next(i for i,n in enumerate(orig_nodes) if n.name==src_node.name)+1
        new_nodes = orig_nodes[:insert_pos]+inj_nodes+new_nodes+orig_nodes[insert_pos:]

        cloned_tgt_out = tmap[orig_tgt_out]
        if "16" in fault_model:
            if tgt_node.op_type == "Conv":
                if "INPUT" in fault_model:
                    m_nodes = create_conv_input16_mask(cloned_tgt_out, f"{cloned_tgt_out}_masked", 16, fp16=fp16_flag)
                else:
                    m_nodes = create_conv_weight16_mask(cloned_tgt_out, f"{cloned_tgt_out}_masked", 16, fp16=fp16_flag)
            else:
                # MatMul / linear path
                if "INPUT" in fault_model:
                    m_nodes = create_input16_mask(cloned_tgt_out, f"{cloned_tgt_out}_masked", 16, fp16=fp16_flag)
                else:
                    m_nodes = create_weight16_mask(cloned_tgt_out, f"{cloned_tgt_out}_masked", 16, fp16=fp16_flag)
            new_nodes.extend(m_nodes)
            cloned_tgt_out = f"{cloned_tgt_out}_masked"

        new_nodes.append(helper.make_node('Identity', [cloned_tgt_out], ['target_layer_output'], name='target_layer_output_identity'))
        add_node = helper.make_node('Add', [orig_tgt_out, cloned_tgt_out], [f"{orig_tgt_out}_final"], name=f"{orig_tgt_out}_Add")
        new_nodes.append(add_node)

        for n in new_nodes:
            if n != add_node and orig_tgt_out in n.input:
                n.input[:] = [f"{orig_tgt_out}_final" if x == orig_tgt_out else x for x in n.input]

    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)

    model.opset_import[0].version = 18
    if "BITFLIP" in fault_model:
        exist = {o.domain:o.version for o in model.opset_import}
        if 'ai.onnx.contrib' not in exist:
            model.opset_import.append(helper.make_opsetid('ai.onnx.contrib',1))
    if llama_config['precision']=='float16' and 'RANDOM' not in fault_model:
        exist={o.domain:o.version for o in model.opset_import}
        if 'custom.perturb' not in exist:
            model.opset_import.append(helper.make_opsetid('custom.perturb',1))

    #model = shape_inference.infer_shapes(model)
    onnx.save(model, output_path)
    print(f"Modified model saved to {output_path}")
    return output_path






