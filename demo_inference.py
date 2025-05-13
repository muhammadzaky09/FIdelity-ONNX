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

def modify_onnx_graph_input(config, llama_config, fault_model, bit_position=3):
    model_path = config["model_name"]
    output_path = config.get("output_path", model_path.replace(".onnx", "_injected.onnx"))

    model = onnx.load(model_path)
    model = patch_reduce_ops(model, reduce_ops=("ReduceMean", "ReduceMax"))
    input_path, _ = analyze_paths(model, config["target_layer"], 
                              config["input_tensor"], 
                              config["weight_tensor"])
    
    if not input_path:
        raise ValueError("Could not find a path matching the given patterns.")
    src_node = input_path[0]
    target_node = input_path[-1]
    

    clone_suffix = "_fault_injected"
    original_target_output = target_node.output[0]

    tensor_map = {}
    cloned_nodes = []
  
    tensor_map[src_node.output[0]] = f"{src_node.output[0]}{clone_suffix}"
    for node in input_path[1:]:
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
    print("Output path:", output_path)
    _, weight_path =  analyze_paths(model, config["target_layer"], 
                              config["input_tensor"], 
                              config["weight_tensor"])
    if not weight_path:
        raise ValueError(f"Could not find a weight path from '{config['weight_tensor']}' to target '{config['target_layer']}'.")
    
    src_node = weight_path[0]
    target_node = weight_path[-1]


    clone_suffix = "_fault_injected"
    original_target_output = target_node.output[0]

    # Clone the chain of nodes from the weight source to the target.
    tensor_map = {}
    cloned_nodes = []
    
    tensor_map[src_node.output[0]] = f"{src_node.output[0]}{clone_suffix}"
    for node in weight_path[1:]:
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
    indices_output_node = helper.make_node(
        'Identity',
        inputs=['indices_int64_inject'],  # Adjust suffix as needed
        outputs=['fault_injection_indices'],  # This becomes available to retrieve
            name='fault_indices_output'
    )
    new_nodes.append(target_output_node)
    new_nodes.append(indices_output_node)
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
            None  
            ),
        helper.make_tensor_value_info(
            'fault_injection_indices',
            TensorProto.INT64,
            None  
        )
    ])
    
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

def extract_answer(full_response, prompt):
    model_answer = full_response[len(prompt):].strip()
    
    patterns = [
        r"^([A-D])",  # Just starts with A, B, C, or D
        r"([A-D])\.",  # Letter followed by period
        r"Answer:\s*([A-D])",  # Explicit "Answer: X"
        r"[Tt]he answer is\s*([A-D])",  # "The answer is X"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_answer)
        if match:
            return match.group(1)
    
    # for char in model_answer:
    #     if char in "ABCD":
    #         return char
    
    return None

class Llama:
    def __init__(self, onnxdir='decoders/7B16', config: dict = {}):
        if not os.path.exists(onnxdir):
            logger.error('{} not exist'.format(onnxdir))

        assert os.path.isdir(onnxdir)

        self.DECODER_COUNT = 32
        # EOS token
        self.FINISH_TOKEN = 2
        self.tokenizer = Tokenizer(os.path.join(onnxdir, 'tokenizer.model'))

        pool = MemoryPoolSimple(config['poolsize'])
        self.decoder = Decoder(pool, onnxdir, 'decoder-merge-{}.onnx',
                               self.DECODER_COUNT)
        self.config = config
        self.device = 'cuda'
        self.seed = None

        # cache
        self.pastkeys = [None for i in range(self.DECODER_COUNT)]
        self.pastvalues = [None for i in range(self.DECODER_COUNT)]
        
        self.faulty_decoders = {}
        self.fault_indices = None

        pool.check()

    # Modified transformers.models.llama.modeling_llama._make_causal_mask with np.array
    def _make_causal_mask(self,
                          input_ids_shape,
                          dtype,
                          past_key_values_length: int = 0):
        """    
        Make causal mask used for bi-directional self-attention. 
        Output triangle-matrix if `past_key_values_length`=0
        Padding left if `past_key_values_length`>0
        """
        bsz, tgt_len = input_ids_shape
        mask = np.full((tgt_len, tgt_len), fill_value=np.finfo(dtype).min)

        mask_cond = np.arange(mask.shape[1])
        cond = mask_cond < (mask_cond + 1).reshape(-1, 1)
        mask = np.ma.array(mask, mask=cond, fill_value=0).filled()
        # masked_fill_result = np.ma.masked_fill_(mask, condition_row_array)

        if past_key_values_length > 0:
            mask = np.concatenate([
                np.zeros((tgt_len, past_key_values_length), dtype=dtype), mask
            ],
                                  axis=1)

        return mask.reshape(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(self, mask, dtype, tgt_len=None):
        """  
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.  
        """
        bsz, src_len = mask.shape
        if tgt_len is None:
            tgt_len = src_len
        # expand [bsz,38] to [bsz,1,1,38]
        expanded_mask = np.expand_dims(mask, axis=1)
        expanded_mask = np.expand_dims(mask, axis=1)
        expanded_mask = np.broadcast_to(expanded_mask,
                                        (bsz, 1, tgt_len, src_len))
        inverted_mask = 1.0 - expanded_mask

        cond = inverted_mask > 0
        return np.ma.array(inverted_mask,
                           mask=cond,
                           fill_value=np.finfo(dtype).min).filled()


    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask,
                                                   inputs_embeds.dtype,
                                                   tgt_len=input_shape[-1])
            combined_attention_mask = (expanded_attn_mask
                                       if combined_attention_mask is None else
                                       expanded_attn_mask +
                                       combined_attention_mask)

        return combined_attention_mask

    def convert_to_fp16(self, inputs):
        outputs = dict()
        for k, v in inputs.items():
            if v.dtype == np.float32:
                outputs[k] = v.astype(np.float16)
            else:
                outputs[k] = v
        return outputs

    def decode(self, token: np.array):
        # embed space
        hidden = self.decoder.embed(token)
        assert hidden.shape[-1] == 4096

        if self.pastkeys[0] is None:
            pastlen = 0
        else:
            pastlen = self.pastkeys[0].shape[-2]
        seqlen = hidden.shape[1]

        position_ids = np.arange(seqlen, dtype=np.int64).reshape((1, seqlen))
        position_ids[0][0] = pastlen

        attention_mask = np.ones((1, seqlen + pastlen), dtype=np.float32)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (1, seqlen), hidden, pastlen)

        for idx in range(self.DECODER_COUNT):
            past_key = self.pastkeys[idx]
            past_value = self.pastvalues[idx]

            if past_key is None:
                zero_tensor = np.zeros((1, 32, 0, 128), dtype=np.float32)
                inputs = {
                    'hidden_in': hidden,
                    'attn_mask': attention_mask,
                    'position_ids': position_ids,
                    'past_key_in': zero_tensor,
                    'past_value_in': zero_tensor
                }
            else:
                inputs = {
                    'hidden_in': hidden,
                    'attn_mask': attention_mask,
                    'position_ids': position_ids,
                    'past_key_in': past_key,
                    'past_value_in': past_value
                }

            if self.config['fp16']:
                inputs = self.convert_to_fp16(inputs)
            outputs = self.decoder.decode(inputs, idx)

            hidden = outputs['hidden_out']
            self.pastkeys[idx] = outputs['past_key']
            self.pastvalues[idx] = outputs['past_value']

        hidden = self.decoder.norm_head(hidden)
        return hidden
    
    def decode_faulty(self, token: np.ndarray):
        """
        Faulty decode: Runs the decoder layers as normal except for one layer.
        At the target decoder layer (specified in fault_config), the normal call is
        replaced by a call to _faulty_decode().
        """
        # Embed tokens.
        hidden = self.decoder.embed(token)
        assert hidden.shape[-1] == 4096

        pastlen = 0 if self.pastkeys[0] is None else self.pastkeys[0].shape[-2]
        seqlen = hidden.shape[1]
        position_ids = np.arange(seqlen, dtype=np.int64).reshape((1, seqlen))
        position_ids[0][0] = pastlen

        attention_mask = np.ones((1, seqlen + pastlen), dtype=np.float32)
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (1, seqlen), hidden, pastlen)

        target_layer_output = None

        for idx in range(self.DECODER_COUNT):
            past_key = self.pastkeys[idx]
            past_value = self.pastvalues[idx]

            if past_key is None:
                zero_tensor = np.zeros((1, 32, 0, 128), dtype=np.float32)
                inputs = {
                    'hidden_in': hidden,
                    'attn_mask': attention_mask,
                    'position_ids': position_ids,
                    'past_key_in': zero_tensor,
                    'past_value_in': zero_tensor
                }
            else:
                inputs = {
                    'hidden_in': hidden,
                    'attn_mask': attention_mask,
                    'position_ids': position_ids,
                    'past_key_in': past_key,
                    'past_value_in': past_value
                }

            if self.config['fp16']:
                inputs = self.convert_to_fp16(inputs)

            # If this is the target decoder layer, call the faulty module.
            if idx == self.fault_config['target_decoder_idx']:
                outputs = self._faulty_decode(inputs, idx)
                print('wawawawaw target token, inject fault')
                if 'target_layer_output' or 'fault_injection_indices' in outputs:
                    target_layer_output = outputs['target_layer_output']
                    print("target: ",np.count_nonzero(target_layer_output))
                    print("norm:", np.linalg.norm(target_layer_output))
                    fault_indices = outputs['fault_injection_indices']
            else:
                outputs = self.decoder.decode(inputs, idx)

            hidden = outputs['hidden_out']
            self.pastkeys[idx] = outputs['past_key']
            self.pastvalues[idx] = outputs['past_value']

        hidden = self.decoder.norm_head(hidden)
        return hidden, target_layer_output, fault_indices

    def _faulty_decode(self, inputs: dict, idx: int):
        from llama.memory_pool import OrtWrapper
        path = self.fault_config['faulty_decoder_path']
        print("path: ", path)
        if path not in self.faulty_decoders:
            self.faulty_decoders[path] = OrtWrapper(path)
        faulty_handler = self.faulty_decoders[path]
        print("faulty handler: ", faulty_handler)
        outputs = faulty_handler.forward(inputs)
        return outputs

    def apply_warp(self, tensor: np.array):
        tensor = warp_temperature(tensor, self.config['temperature'])
        tensor = warp_topp(tensor, self.config['topp'])
        return tensor

    def sample_golden(self, prompt: str = 'bonjour'):
        """
        Golden run: Runs full inference with no fault injection.
        """
        # Print the prompt with clear separation
        logger.debug("=" * 80)
        logger.debug("GOLDEN RUN START")
        logger.debug("=" * 80)
        
        # Log the prompt (truncated if very long)
        if len(prompt) > 500:
            logger.debug(f"PROMPT (beginning):\n{prompt[:200]}...")
            logger.debug(f"PROMPT (end):\n...{prompt[-300:]}")
        else:
            logger.debug(f"PROMPT:\n{prompt}")
        
        logger.debug("-" * 80)  # Separator between prompt and response
        
        prompt = prompt.strip()
        input_ids = self.tokenizer.encode(prompt, True, False)
        input_ids = np.array(input_ids, dtype=np.int64).reshape(
            (1, len(input_ids)))
        
        # Reset caches
        self.pastkeys = [None] * self.DECODER_COUNT
        self.pastvalues = [None] * self.DECODER_COUNT
        
        generated_tokens = []  
        token_count = 0
        next_token = input_ids
        golden_logits = None
        
        while True:
            logits = self.decode(next_token)
            if token_count == self.fault_config.get('target_token_idx', 0):
                golden_logits = logits[:, -1, :].copy()
            next_token_scores = logits[:, -1, :]
            next_token_scores = self.apply_warp(next_token_scores)
            probs = npsoftmax(next_token_scores.astype(np.float64), axis=1)
            next_token = seeded_npmultinomial2D(probs, self.seed).astype(input_ids.dtype)
            token_id = int(next_token[0, 0])
            generated_tokens.append(token_id)
            input_ids = np.concatenate([input_ids, next_token.reshape((1, 1))], axis=1)
            token_count += 1
        
            if input_ids.shape[-1] >= self.config['max'] or next_token[0,0] == self.FINISH_TOKEN:
                break
        
        # Get full response
        full_response = self.tokenizer.decode(input_ids[0].tolist())
        
        # Extract just the model's answer (after the prompt)
        model_answer = full_response[len(prompt):].strip()
        
        # Log the model's response
        logger.debug(f"MODEL ANSWER:\n{model_answer}")
        logger.debug("=" * 80)
        logger.debug("GOLDEN RUN END")
        logger.debug("=" * 80)
        
        # Add token length to return values
        token_length = len(generated_tokens)
        
        # Get token at the first position for experiment 0
        first_token = generated_tokens[0] if generated_tokens else None

            
        return full_response, first_token, golden_logits
    
    def sample_faulty(self, prompt: str = 'bonjour'):
        """
        Faulty run: Runs inference with fault injection at specified token.
        """
        # Print the prompt with clear separation
        logger.debug("=" * 80)
        logger.debug("FAULTY RUN START")
        logger.debug("=" * 80)
        
        # Log the prompt (truncated if very long)
        if len(prompt) > 500:
            logger.debug(f"PROMPT (beginning):\n{prompt[:200]}...")
            logger.debug(f"PROMPT (end):\n...{prompt[-300:]}")
        else:
            logger.debug(f"PROMPT:\n{prompt}")
        
        logger.debug(f"Fault Config: Target Decoder Idx = {self.fault_config['target_decoder_idx']}, "
                    f"Target Token Idx = {self.fault_config['target_token_idx']}")
        logger.debug("-" * 80)  # Separator between prompt and response
        
        prompt = prompt.strip()
        input_ids = self.tokenizer.encode(prompt, True, False)
        input_ids = np.array(input_ids, dtype=np.int64).reshape(
            (1, len(input_ids)))

        # Reset caches
        self.pastkeys = [None] * self.DECODER_COUNT
        self.pastvalues = [None] * self.DECODER_COUNT

        token_count = 0
        generated_tokens = []
        next_token = input_ids
        target_nonzeros = None
        faulty_logits = None
        
        while True:
            # At the target token generation, use decode_faulty
            if token_count == self.fault_config['target_token_idx']:
                logger.debug(f"Injecting fault at token position {token_count}")
                logits, layer_output, fault_indices = self.decode_faulty(next_token)
                faulty_logits = logits[:, -1, :].copy()
                if layer_output is not None:
                    # Get non-zero indices
                    target_nonzeros = np.count_nonzero(layer_output)
            else:
                logits = self.decode(next_token)
                    
            next_token_scores = logits[:, -1, :]
            next_token_scores = self.apply_warp(next_token_scores)
            probs = npsoftmax(next_token_scores.astype(np.float64), axis=1)
            next_token = seeded_npmultinomial2D(probs, self.seed).astype(input_ids.dtype)
            token_id = int(next_token[0, 0])
            generated_tokens.append(token_id)
            input_ids = np.concatenate([input_ids, next_token.reshape((1, 1))], axis=1)
            token_count += 1

            if input_ids.shape[-1] >= self.config['max'] or token_id == self.FINISH_TOKEN:
                break

        # Get full response
        full_response = self.tokenizer.decode(input_ids[0].tolist())
        
        # Extract just the model's answer (after the prompt)
        model_answer = full_response[len(prompt):].strip()
        
        # Log the model's response
        logger.debug(f"MODEL ANSWER:\n{model_answer}")
        logger.debug("=" * 80)
        logger.debug("FAULTY RUN END")
        logger.debug("=" * 80)
        
        # Get the token at the target position
        faulty_token = None
        if self.fault_config['target_token_idx'] < len(generated_tokens):
            faulty_token = generated_tokens[self.fault_config['target_token_idx']]
                
        return full_response, faulty_token, target_nonzeros, faulty_logits, fault_indices

def extract_decoder_idx(path):
    """Extract decoder index from filename"""
    import os
    filename = os.path.basename(path)
    if 'decoder-merge-' in filename:
        decoder_idx_str = filename.split('decoder-merge-')[1].split('_')[0]
        return int(decoder_idx_str)

def are_logits_equal(golden_logits, faulty_logits):
    if golden_logits is None or faulty_logits is None:
        return False
    if golden_logits.shape != faulty_logits.shape:
        return False
    return np.array_equal(golden_logits, faulty_logits)

def create_demo_prompt():
    """Create the demonstration prompt with one-shot example"""
    one_shot = "Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\n"
    problem = "(1+i)^10 = \nA. 1\nB. i\nC. 32\nD. 32i"
    return one_shot + problem

def run_fault_injection_demo():
    """Single run fault injection demonstration"""
    
    # Configure Llama model with low temperature for multiple choice
    llama_config = {
        'temperature': 0.001,
        'topp': 0.1,
        'max': 300,
        'poolsize': 44,
        'fp16': True,
        'precision': 'int8',
        'onnxdir': 'decoders/7B16',
        'layer_files': 'injection_llm',
    }
    print("Starting LLM fault injection demo with config:", llama_config)

    # Create Llama instance
    persistent_llama = Llama(onnxdir=llama_config['onnxdir'], config=llama_config)
    
    # Fixed parameters for our demo
    layer_file = "decoder-merge-0_down_proj.json"
    fault_model = "WEIGHT"
    bit_position = 4
    
    # Set a fixed random seed for reproducibility
    random_seed = 42
    persistent_llama.seed = random_seed
    
    # Load the layer configuration
    config_path = os.path.join(llama_config['layer_files'], layer_file)
    config = json.load(open(config_path))
    
    # Create the faulty model
    print(f"\nCreating fault model with: {layer_file}, {fault_model}, bit position {bit_position}")
    faulty_path = modify_onnx_graph_weight(config, llama_config, fault_model, bit_position)
    
    # Set up fault config
    fault_config = {
        'target_decoder_idx': 0,  # Fixed to decoder 0
        'target_token_idx': 0,    # Inject at first token
        'faulty_decoder_path': faulty_path
    }
    persistent_llama.fault_config = fault_config
    
    # Create our demo prompt
    prompt = create_demo_prompt()
    print(f"\nPrompt:\n{prompt}\n")
    
    # Golden run (normal execution)
    print("\n=== Running Golden (Normal) Inference ===")
    golden_output, first_token, golden_logits = persistent_llama.sample_golden(prompt)
    
    # Extract and print the model's answer
    golden_answer = extract_answer(golden_output, prompt)
    print(f"Golden Output: {golden_output[len(prompt):].strip()}")
    print(f"Extracted Answer: {golden_answer}")
    
    # Faulty run (with injection)
    print("\n=== Running Faulty Inference with Bit Flip ===")
    faulty_output, faulty_token, target_nonzeros, faulty_logits, fault_indices = persistent_llama.sample_faulty(prompt)
    
    # Extract and print the model's answer
    faulty_answer = extract_answer(faulty_output, prompt)
    print(f"Faulty Output: {faulty_output[len(prompt):].strip()}")
    print(f"Extracted Answer: {faulty_answer}")
    print(f"Fault injected at indices: {fault_indices}")
    
    # Compare results
    print("\n=== Results Comparison ===")
    print(f"Golden answer: {golden_answer}")
    print(f"Faulty answer: {faulty_answer}")
    print(f"Answer changed: {'YES' if golden_answer != faulty_answer else 'NO'}")
    
    # Clean up the faulty decoder
    if faulty_path in persistent_llama.faulty_decoders:
        del persistent_llama.faulty_decoders[faulty_path]
    if os.path.exists(faulty_path):
        os.remove(faulty_path)
    
    # Run garbage collection
    gc.collect()
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    run_fault_injection_demo()