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

EVALUATION_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology"
]

def analyze_path(model, start_pattern, target_config):
    # Build consumer map lazily and only explore relevant paths
    consumers = defaultdict(list)
    
    # Extract operation type more robustly - remove numeric suffixes
    import re
    target_name = target_config.split("/")[-1]
    allowed_op_type = re.sub(r'_\d+$', '', target_name)  # Remove trailing _number
    print(f"Looking for operation type: {allowed_op_type} in target: {target_config}")
    
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
                # More flexible matching for operation types
                if (consumer.op_type == allowed_op_type and target_config in consumer.name):
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
    
    # Add debugging info
    print(f"No path found. Checked {len(source_nodes)} source nodes.")
    print(f"Source nodes: {[n.name for n in source_nodes]}")
    print(f"Looking for target pattern: {target_config}")
    
    # List all MatMul nodes to help with debugging
    matmul_nodes = [node for node in model.graph.node if node.op_type == allowed_op_type]
    print(f"Found {len(matmul_nodes)} {allowed_op_type} nodes in model:")
    for node in matmul_nodes[:10]:  # Show only first 10 to avoid overwhelming output
        print(f"  - Name: {node.name}, Outputs: {node.output}")
    
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
                fp16=True
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

def load_mmlu_dataset():
    try:
        test_dataset = load_dataset("cais/mmlu", "all", split="test")
        test_dataset = test_dataset.filter(lambda x: x['subject'] in EVALUATION_SUBJECTS)
        dev_dataset = load_dataset("cais/mmlu", "all", split="dev")
        dev_dataset = test_dataset.filter(lambda x: x['subject'] in EVALUATION_SUBJECTS)
        dev_list = [ex for ex in dev_dataset]
        test_list = [ex for ex in test_dataset]
        subjects = sorted(list(set(ex['subject'] for ex in test_list)))
        logger.info(f"Found {len(subjects)} subjects in MMLU dataset")
        
        subject_to_examples = {}
        for subject in subjects:
            subject_exs = [ex for ex in test_list if ex['subject'] == subject]
            subject_to_examples[subject] = subject_exs
        
        total_examples = sum(len(exs) for exs in subject_to_examples.values())
        logger.info(f"Loaded {total_examples} examples total across {len(subjects)} subjects")
        return dev_list, subject_to_examples, subjects
        
    except Exception as e:
        logger.error(f"Error loading MMLU dataset: {e}")
        return None, None, None

def create_few_shot_prompt(dev_examples, test_example, num_shots=5):
    subject = test_example['subject']
    subject_examples = [ex for ex in dev_examples if ex['subject'] == subject]
    random.seed(42)
    shot_examples = random.sample(subject_examples, num_shots)
    prompt = ""
    
    for example in shot_examples:
        question = example['question']
        choices = example['choices']
        answer_idx = example['answer']
        correct_letter = chr(65 + answer_idx) 
        
        prompt += f"Question: {question}\n"
        prompt += f"A. {choices[0]}\n"
        prompt += f"B. {choices[1]}\n"
        prompt += f"C. {choices[2]}\n"
        prompt += f"D. {choices[3]}\n"
        prompt += f"Answer: {correct_letter}\n\n"
    
    prompt += f"Question: {test_example['question']}\n"
    prompt += f"A. {test_example['choices'][0]}\n"
    prompt += f"B. {test_example['choices'][1]}\n"
    prompt += f"C. {test_example['choices'][2]}\n"
    prompt += f"D. {test_example['choices'][3]}\n"
    prompt += "Answer:"
    
    return prompt

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
                if 'target_layer_output' in outputs:
                    target_layer_output = outputs['target_layer_output']
                    print("target: ",np.count_nonzero(target_layer_output))
                    print("norm:", np.linalg.norm(target_layer_output))
                    print(target_layer_output)
            else:
                outputs = self.decoder.decode(inputs, idx)

            hidden = outputs['hidden_out']
            self.pastkeys[idx] = outputs['past_key']
            self.pastvalues[idx] = outputs['past_value']

        hidden = self.decoder.norm_head(hidden)
        return hidden, target_layer_output

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
                logits, layer_output = self.decode_faulty(next_token)
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
                
        return full_response, faulty_token, target_nonzeros, faulty_logits

    # Add MMLU-specific methods
    def process_mmlu_example(self, test_example, dev_examples, num_shots=1):
        """Run MMLU inference and extract results"""
        # Create few-shot prompt
        prompt = create_few_shot_prompt(dev_examples, test_example, num_shots)
        
        # Get golden output
        golden_output, first_token, golden_logits = self.sample_golden(prompt)
        
        # Extract answer
        predicted_letter = extract_answer(golden_output, prompt)
        correct_letter = chr(65 + test_example['answer'])  # Convert 0,1,2,3 to A,B,C,D
        
        # Print results in a user-friendly way
        logger.info("\nGOLDEN RUN RESULTS:")
        logger.info("-" * 40)
        logger.info(f"Question: {test_example['question']}")
        logger.info(f"A: {test_example['choices'][0]}")
        logger.info(f"B: {test_example['choices'][1]}")
        logger.info(f"C: {test_example['choices'][2]}")
        logger.info(f"D: {test_example['choices'][3]}")
        logger.info(f"Correct Answer: {correct_letter}")
        model_output = golden_output[len(prompt):].strip()
        logger.info(f"Model Output: {model_output[:10]}{'...' if len(model_output) > 100 else ''}")
        logger.info(f"Predicted: {predicted_letter or 'None'}")
        logger.info(f"Correct? {'✓ YES' if predicted_letter == correct_letter else '✗ NO'}")
        
        return {
            'prompt': prompt,
            'output': golden_output, 
            'model_output': model_output,  # Just the answer part without prompt
            'predicted': predicted_letter,
            'correct': correct_letter,
            'is_correct': (predicted_letter == correct_letter) if predicted_letter else False,
            'first_token': first_token,
            'golden_logits': golden_logits
        }
    
    def process_mmlu_example_faulty(self, test_example, dev_examples, prompt, num_shots=1):
        """Run faulty MMLU inference and extract results"""
        # Get faulty output
        faulty_output, faulty_token, target_nonzeros, faulty_logits = self.sample_faulty(prompt)
        
        # Extract answer
        predicted_letter = extract_answer(faulty_output, prompt)
        correct_letter = chr(65 + test_example['answer'])
        
        # Print results in a user-friendly way
        logger.info("\nFAULTY RUN RESULTS:")
        logger.info("-" * 40)
        model_output = faulty_output[len(prompt):].strip()
        logger.info(f"Model Output: {model_output[:100]}{'...' if len(model_output) > 100 else ''}")
        logger.info(f"Predicted: {predicted_letter or 'None'}")
        logger.info(f"Correct? {'✓ YES' if predicted_letter == correct_letter else '✗ NO'}")
        
        return {
            'output': faulty_output,
            'model_output': model_output,
            'predicted': predicted_letter,
            'correct': correct_letter,
            'is_correct': (predicted_letter == correct_letter) if predicted_letter else False,
            'faulty_token': faulty_token,
            'target_nonzeros': target_nonzeros,
            'faulty_logits': faulty_logits
        }

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

if __name__ == "__main__":
    logger.info("Starting MMLU fault injection experiments...")

    # Load dataset
    dev_examples, subject_to_examples, subjects = load_mmlu_dataset()
    if not dev_examples or not subject_to_examples or not subjects:
        logger.error("Failed to load MMLU dataset. Exiting.")
        exit(1)

    # Configure Llama model with low temperature for multiple choice
    llama_config = {
        'temperature': 0.001,
        'topp': 0.1,
        'max': 300,
        'poolsize': 44,
        'fp16': True,
        'precision': 'int8',
        'onnxdir': 'alpaca',
        'layer_files': 'injection_llm',
    }

    # Create Llama instance
    persistent_llama = Llama(onnxdir=llama_config['onnxdir'], config=llama_config)

    # Create CSV file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = 'mmlu_fault_injection_results1.csv'
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, 'a' if file_exists else 'w', newline='') as csvfile:
        fieldnames = [
            'Timestamp', 'Subject', 
            'Question', 'Option_A', 'Option_B', 'Option_C', 'Option_D',
            'Layer_Config', 'Fault_Model', 'Bit_Position', 
            'Target_Decoder_Idx', 'Target_Token_Idx', 'Experiment',
            'Golden_Answer', 'Faulty_Answer', 'Correct_Answer',
            'Golden_Correct', 'Faulty_Correct', 'Answer_Changed',
            'Correctness_Changed', 'Golden_Raw_Output', 'Faulty_Raw_Output', 'Golden_Token', 'Faulty_Token','Target_Nonzeros','Logits_Equal'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    # Track subject usage to ensure all subjects are covered
    subject_index = 0
    subjects_used = set()
    
    
    for layer_file in os.listdir(llama_config['layer_files']):
        config_path = os.path.join(llama_config['layer_files'], layer_file)
        # Skip directories
        if os.path.isdir(config_path):
            continue
            
        config = json.load(open(config_path))
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing layer configuration: {layer_file}")
        logger.info(f"{'='*40}")
        
        # For each fault model
        for fault_model in ['INPUT','WEIGHT','INPUT16','WEIGHT16']: 
            for bit_position in range(8):
                curr_subject = subjects[subject_index % len(subjects)]
                subjects_used.add(curr_subject)
                examples = subject_to_examples[curr_subject]
                
                
                if len(examples) >= 2:
                    test_examples = random.sample(examples, 2)
                else:
                    test_examples = [examples[0], examples[0]]
                
                print(f"\n{'-'*40}")
                print(f"Layer: {layer_file}, Fault Model: {fault_model}, Bit: {bit_position}")
                print(f"Using subject: {curr_subject} (Subject {subject_index % len(subjects) + 1}/57)")
                print(f"{'-'*40}")
                
                
                random_seed = (bit_position * 1000 + 1)
                persistent_llama.seed = random_seed
                
                
                logger.info(f"Creating faulty decoder for {fault_model} on bit position {bit_position}...")
                if fault_model in ['INPUT', 'INPUT16']:
                    faulty_path = modify_onnx_graph_input(config, llama_config, fault_model, bit_position)
                elif fault_model in ['WEIGHT', 'WEIGHT16']:
                    faulty_path = modify_onnx_graph_weight(config, llama_config,fault_model, bit_position)
                else:
                    faulty_path = modify_onnx_graph_random(config, llama_config,fault_model, bit_position)
                
            
                for experiment in range(1):
                    test_example = test_examples[experiment]
                    print(f"\nRunning experiment {experiment} with token position 0 (first token)")
                    print(f"Question: {test_example['question']}")
                    
                    # Set up fault config (always target token 0)
                    fault_config = {
                        'target_decoder_idx': extract_decoder_idx(faulty_path),
                        'target_token_idx': 0,  # Always target first token
                        'faulty_decoder_path': faulty_path
                    }
                    persistent_llama.fault_config = fault_config
                    
                    # Run golden inference
                    print("Running golden inference...")
                    golden_result = persistent_llama.process_mmlu_example(test_example, dev_examples)
                    
                    
                    
                    # Run faulty inference
                    print("Running faulty inference...")
                    faulty_result = persistent_llama.process_mmlu_example_faulty(
                        test_example, dev_examples, golden_result['prompt']
                    )
                    
                    # In the main experiment loop, after running golden and faulty inference:
                    if golden_result.get('golden_logits') is not None:
                        logits_dir = 'logits_data'
                        os.makedirs(logits_dir, exist_ok=True)
                        
                        # Save golden logits
                        golden_logits_path = os.path.join(
                            logits_dir, 
                            f"golden_logits_layer{layer_file}_model{fault_model}_bit{bit_position}_exp{experiment}.npy"
                        )
                        np.save(golden_logits_path, golden_result['golden_logits'])
                        
                        # Save faulty logits
                        if faulty_result.get('faulty_logits') is not None:
                            faulty_logits_path = os.path.join(
                                logits_dir, 
                                f"faulty_logits_layer{layer_file}_model{fault_model}_bit{bit_position}_exp{experiment}.npy"
                            )
                            np.save(faulty_logits_path, faulty_result['faulty_logits'])
                    # Analyze changes
                    answer_changed = golden_result['predicted'] != faulty_result['predicted']
                    correctness_changed = golden_result['is_correct'] != faulty_result['is_correct']
                    logits_equal = are_logits_equal(golden_result.get('golden_logits'), faulty_result.get('faulty_logits'))
                    # Display comparison
                    print("\nCOMPARISON RESULTS:")
                    print(f"{'='*40}")
                    print(f"Golden Answer: {golden_result['predicted'] or 'None'}")
                    print(f"Faulty Answer: {faulty_result['predicted'] or 'None'}")
                    print(f"Correct Answer: {golden_result['correct']}")
                    print(f"Answer Changed: {'YES' if answer_changed else 'NO'}")
                    print(f"Correctness Changed: {'YES' if correctness_changed else 'NO'}")
                    
                    # Save results to CSV
                    with open(csv_filename, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({
                            'Timestamp': datetime.now().isoformat(),
                            'Subject': curr_subject,
                            'Question': test_example['question'],
                            'Option_A': test_example['choices'][0],
                            'Option_B': test_example['choices'][1],
                            'Option_C': test_example['choices'][2],
                            'Option_D': test_example['choices'][3],
                            'Layer_Config': layer_file,
                            'Fault_Model': fault_model,
                            'Bit_Position': bit_position,
                            'Target_Decoder_Idx': fault_config['target_decoder_idx'],
                            'Target_Token_Idx': fault_config['target_token_idx'],
                            'Experiment': experiment,
                            'Golden_Answer': golden_result['predicted'] or 'None',
                            'Faulty_Answer': faulty_result['predicted'] or 'None',
                            'Correct_Answer': golden_result['correct'],
                            'Golden_Correct': golden_result['is_correct'],
                            'Faulty_Correct': faulty_result['is_correct'],
                            'Answer_Changed': answer_changed,
                            'Correctness_Changed': correctness_changed,
                            'Golden_Raw_Output': golden_result['model_output'],
                            'Faulty_Raw_Output': faulty_result['model_output'],
                            'Golden_Token': golden_result['first_token'],
                            'Faulty_Token': faulty_result['faulty_token'],
                            'Target_Nonzeros': str(faulty_result['target_nonzeros']) if faulty_result.get('target_nonzeros') is not None else "None",
                            'Logits_Equal': logits_equal
                        })
                
                # Clean up the faulty decoder
                if faulty_path in persistent_llama.faulty_decoders:
                    del persistent_llama.faulty_decoders[faulty_path]
                if os.path.exists(faulty_path):
                    os.remove(faulty_path)
                
                # Move to next subject and run garbage collection
                subject_index += 1
                gc.collect()

    logger.info("\nExperiments completed.")
    logger.info(f"Used {len(subjects_used)} out of {len(subjects)} subjects")
    
    if len(subjects_used) < len(subjects):
        unused_subjects = set(subjects) - subjects_used
        logger.info(f"Unused subjects: {', '.join(unused_subjects)}")
    else:
        logger.info("All MMLU subjects were used in the experiments.")
    
    logger.info(f"Results saved to {csv_filename}")
