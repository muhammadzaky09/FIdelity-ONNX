import os
import onnxruntime as ort
import json
import re
from loguru import logger
from llama.decoder import Decoder
from llama.memory_pool import MemoryPoolSimple
from llama.utils import npsoftmax, seeded_npmultinomial2D
from llama.logits_process import warp_temperature, warp_topp
from llama.tokenizer import Tokenizer
import argparse
from datasets import load_dataset
import csv
from graph import modify_onnx_graph
import numpy as np
import random
from datetime import datetime
import gc


def load_prompts(args) -> tuple:
    """
    Load prompts and optional ground-truth labels from either:
      --csv PATH          local CSV file; --prompt_field and --label_field
                          select columns (default: 'question' / None)
      --dataset NAME      HuggingFace dataset; same --prompt_field /
                          --label_field apply, plus --dataset_split

    Returns (prompts, labels): two lists of equal length.  labels[i] is None
    when --label_field is not set.
    """
    if args.csv:
        import csv as _csv
        with open(args.csv, newline='') as f:
            rows = list(_csv.DictReader(f))
        assert rows, f"--csv file '{args.csv}' is empty or has no header"
        prompts = [str(r[args.prompt_field]) for r in rows]
        labels  = ([str(r[args.label_field]) for r in rows]
                   if args.label_field else [None] * len(rows))
        logger.info(f"Loaded {len(prompts)} prompts from '{args.csv}' "
                    f"(prompt={args.prompt_field}"
                    + (f", label={args.label_field})" if args.label_field else ")"))
        return prompts, labels

    # HuggingFace dataset mode
    ds = load_dataset(args.dataset, split=args.dataset_split)
    prompts = [str(ex[args.prompt_field]) for ex in ds]
    labels  = ([str(ex[args.label_field]) for ex in ds]
               if args.label_field else [None] * len(prompts))
    logger.info(f"Loaded {len(prompts)} prompts from dataset '{args.dataset}' "
                f"(split={args.dataset_split}, prompt={args.prompt_field}"
                + (f", label={args.label_field})" if args.label_field else ")"))
    return prompts, labels

class Llama:
    def __init__(self, onnxdir='decoders/7B16', config: dict = {}, model_spec: dict = {}):
        if not os.path.exists(onnxdir):
            logger.error('{} not exist'.format(onnxdir))

        assert os.path.isdir(onnxdir)

        self.DECODER_COUNT = model_spec.get('decoder_count', 32)
        self.FINISH_TOKEN  = model_spec.get('eos_token_id', 2)
        self.hidden_dim    = model_spec.get('hidden_dim', 4096)
        self.n_heads       = model_spec.get('n_heads', 32)
        self.head_dim      = model_spec.get('head_dim', 128)
        self.decoder_template = model_spec.get('decoder_template', 'decoder-merge-{}.onnx')

        # Tensor names for decoder inputs/outputs
        in_names  = model_spec.get('input_names', {})
        out_names = model_spec.get('output_names', {})
        self.in_hidden     = in_names.get('hidden',       'hidden_in')
        self.in_attn_mask  = in_names.get('attn_mask',    'attn_mask')
        self.in_pos_ids    = in_names.get('position_ids', 'position_ids')
        self.in_past_key   = in_names.get('past_key',     'past_key_in')
        self.in_past_value = in_names.get('past_value',   'past_value_in')
        self.out_hidden    = out_names.get('hidden',    'hidden_out')
        self.out_past_key  = out_names.get('past_key',  'past_key')
        self.out_past_value= out_names.get('past_value','past_value')

        tokenizer_file = model_spec.get('tokenizer_file', 'tokenizer.model')
        self.tokenizer = Tokenizer(os.path.join(onnxdir, tokenizer_file))

        pool = MemoryPoolSimple(config['poolsize'])
        self.decoder = Decoder(
            pool, onnxdir, self.decoder_template, self.DECODER_COUNT,
            embed_file   = model_spec.get('embed_file',   'embed.onnx'),
            norm_file    = model_spec.get('norm_file',    'norm.onnx'),
            head_file    = model_spec.get('head_file',    'head.onnx'),
            embed_input  = model_spec.get('embed_input',  'input'),
            embed_output = model_spec.get('embed_output', 'embed'),
            norm_input   = model_spec.get('norm_input',   'input'),
            norm_output  = model_spec.get('norm_output',  'output'),
            head_input   = model_spec.get('head_input',   'input'),
            head_output  = model_spec.get('head_output',  'output'),
        )
        self.config = config
        self.device = 'cuda'
        self.seed = None

        # KV cache
        self.pastkeys   = [None] * self.DECODER_COUNT
        self.pastvalues = [None] * self.DECODER_COUNT

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
        hidden = self.decoder.embed(token)
        assert hidden.shape[-1] == self.hidden_dim

        pastlen = 0 if self.pastkeys[0] is None else self.pastkeys[0].shape[-2]
        seqlen = hidden.shape[1]

        position_ids = np.arange(seqlen, dtype=np.int64).reshape((1, seqlen))
        position_ids[0][0] = pastlen

        attention_mask = np.ones((1, seqlen + pastlen), dtype=np.float32)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (1, seqlen), hidden, pastlen)

        zero_tensor = np.zeros((1, self.n_heads, 0, self.head_dim), dtype=np.float32)

        for idx in range(self.DECODER_COUNT):
            past_key   = self.pastkeys[idx]
            past_value = self.pastvalues[idx]

            inputs = {
                self.in_hidden:     hidden,
                self.in_attn_mask:  attention_mask,
                self.in_pos_ids:    position_ids,
                self.in_past_key:   zero_tensor if past_key   is None else past_key,
                self.in_past_value: zero_tensor if past_value is None else past_value,
            }

            if self.config['fp16']:
                inputs = self.convert_to_fp16(inputs)
            outputs = self.decoder.decode(inputs, idx)

            hidden = outputs[self.out_hidden]
            self.pastkeys[idx]   = outputs[self.out_past_key]
            self.pastvalues[idx] = outputs[self.out_past_value]

        hidden = self.decoder.norm_head(hidden)
        return hidden
    
    def decode_faulty(self, token: np.ndarray):
        """
        Faulty decode: runs all decoder layers normally except the target layer,
        which is replaced by the injected ONNX module (_faulty_decode).
        """
        hidden = self.decoder.embed(token)
        assert hidden.shape[-1] == self.hidden_dim

        pastlen = 0 if self.pastkeys[0] is None else self.pastkeys[0].shape[-2]
        seqlen = hidden.shape[1]
        position_ids = np.arange(seqlen, dtype=np.int64).reshape((1, seqlen))
        position_ids[0][0] = pastlen

        attention_mask = np.ones((1, seqlen + pastlen), dtype=np.float32)
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (1, seqlen), hidden, pastlen)

        zero_tensor = np.zeros((1, self.n_heads, 0, self.head_dim), dtype=np.float32)

        for idx in range(self.DECODER_COUNT):
            past_key   = self.pastkeys[idx]
            past_value = self.pastvalues[idx]

            inputs = {
                self.in_hidden:     hidden,
                self.in_attn_mask:  attention_mask,
                self.in_pos_ids:    position_ids,
                self.in_past_key:   zero_tensor if past_key   is None else past_key,
                self.in_past_value: zero_tensor if past_value is None else past_value,
            }

            if self.config['fp16']:
                inputs = self.convert_to_fp16(inputs)

            if idx == self.fault_config['target_decoder_idx']:
                outputs = self._faulty_decode(inputs, idx)
            else:
                outputs = self.decoder.decode(inputs, idx)

            hidden = outputs[self.out_hidden]
            self.pastkeys[idx]   = outputs[self.out_past_key]
            self.pastvalues[idx] = outputs[self.out_past_value]

        hidden = self.decoder.norm_head(hidden)
        return hidden

    def _faulty_decode(self, inputs: dict, idx: int):
        from llama.memory_pool import OrtWrapper
        path = self.fault_config['faulty_decoder_path']
        if path not in self.faulty_decoders:
            self.faulty_decoders[path] = OrtWrapper(path)
        faulty_handler = self.faulty_decoders[path]

        # Augment inputs with the two fault-injection scalars required by the
        # injected model.  hidden_dim is used as a conservative upper bound for
        # the flat element index so the index is always in-bounds even when
        # seq_len=1 (single-token generation step).
        #
        # rand_idx_inject is drawn from a local RNG seeded by inject_seed, which
        # is derived from (layer, fault_model, bit_position, experiment) in the
        # outer loop.  This makes the injected element index fully reproducible
        # across restarts without touching the global np.random state.
        inputs = dict(inputs)
        rng = np.random.default_rng(self.fault_config.get('inject_seed'))
        inputs["rand_idx_inject"] = np.array(
            rng.integers(0, self.hidden_dim), dtype=np.int64)
        inputs["bit_pos_inject"] = np.array(
            self.fault_config['bit_position'], dtype=np.int32)

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
        
        try:
            while True:
                logits = self.decode(next_token)
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
            
            first_token = generated_tokens[0] if generated_tokens else None
            return full_response, first_token
        finally:
            self.pastkeys  = [None] * self.DECODER_COUNT
            self.pastvalues = [None] * self.DECODER_COUNT
    
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
        
        try:
            while True:
                # At the target token generation, use decode_faulty
                if token_count == self.fault_config['target_token_idx']:
                    logger.debug(f"Injecting fault at token position {token_count}")
                    logits = self.decode_faulty(next_token)
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
            
            faulty_token = None
            if self.fault_config['target_token_idx'] < len(generated_tokens):
                faulty_token = generated_tokens[self.fault_config['target_token_idx']]
            return full_response, faulty_token
        finally:
            self.pastkeys  = [None] * self.DECODER_COUNT
            self.pastvalues = [None] * self.DECODER_COUNT

    def process_prompt(self, prompt: str):
        """Golden run for a single prompt string."""
        logger.info("\nGOLDEN RUN RESULTS:")
        logger.info("-" * 40)
        golden_output, first_token = self.sample_golden(prompt)
        model_output = golden_output[len(prompt):].strip()
        logger.info(f"Model Output: {model_output[:100]}{'...' if len(model_output) > 100 else ''}")
        return {
            'prompt': prompt,
            'golden_output': model_output,
            'golden_token': first_token,
        }

    def process_prompt_faulty(self, prompt: str):
        """Faulty run for a single prompt string."""
        logger.info("\nFAULTY RUN RESULTS:")
        logger.info("-" * 40)
        faulty_output, faulty_token = self.sample_faulty(prompt)
        model_output = faulty_output[len(prompt):].strip()
        logger.info(f"Model Output: {model_output[:100]}{'...' if len(model_output) > 100 else ''}")
        return {
            'faulty_output': model_output,
            'faulty_token': faulty_token,
        }

def extract_decoder_idx(path: str, decoder_template: str = 'decoder-merge-{}.onnx') -> int:
    """
    Extract the decoder layer index from the injected ONNX filename.

    Uses the model's decoder_template (e.g. 'decoder-merge-{}.onnx') to
    derive a stable prefix ('decoder-merge-') so this works for any naming
    convention, not just LLaMA.
    """
    filename = os.path.basename(path)
    # Split the template on the format placeholder to get prefix/suffix
    parts = decoder_template.split('{}')
    prefix = parts[0]   # e.g. 'decoder-merge-'
    if prefix and prefix in filename:
        after_prefix = filename.split(prefix, 1)[1]
        # The index is the leading digits (suffix may be '_injected.onnx' etc.)
        idx_str = ''
        for ch in after_prefix:
            if ch.isdigit():
                idx_str += ch
            else:
                break
        if idx_str:
            return int(idx_str)
    raise ValueError(f"Cannot extract decoder index from '{filename}' using template '{decoder_template}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM fault-injection inference")

    # Prompt source — exactly one required
    prompt_src = parser.add_mutually_exclusive_group(required=True)
    prompt_src.add_argument('--csv', metavar='PATH',
                            help='Local CSV file containing prompts (and optionally labels)')
    prompt_src.add_argument('--dataset', metavar='NAME',
                            help='HuggingFace dataset name, e.g. cais/mmlu')

    # Column/field selection — shared by both --csv and --dataset
    parser.add_argument('--prompt_field', default='question',
                        help='Column name to use as the prompt string (default: question)')
    parser.add_argument('--label_field', default=None,
                        help='Column name to record as ground-truth label in the CSV '
                             '(e.g. "answer"). If omitted, Ground_Truth_Label is blank.')

    # HuggingFace-only option
    parser.add_argument('--dataset_split', default='test',
                        help='HuggingFace dataset split to load (default: test)')

    parser.add_argument('--model_config', default='configs/llama_7b.json',
                        help='Path to model spec JSON (default: configs/llama_7b.json)')
    parser.add_argument('--onnxdir',     default='alpaca',
                        help='Directory containing decoder ONNX files (default: alpaca)')
    parser.add_argument('--layer_files', default='injection_llm',
                        help='Directory containing layer injection JSON configs (default: injection_llm)')
    parser.add_argument('--precision',   default='int8', choices=['int8', 'float16', 'float32'],
                        help='Model precision (default: int8)')
    parser.add_argument('--fp16',        action='store_true', default=True,
                        help='Run inference in FP16 (default: true)')
    parser.add_argument('--no_fp16',     action='store_false', dest='fp16',
                        help='Disable FP16 inference')
    parser.add_argument('--temperature', type=float, default=0.001,
                        help='Sampling temperature (default: 0)')
    parser.add_argument('--topp',        type=float, default=0.1,
                        help='Top-p sampling (default: 0.1)')
    parser.add_argument('--max_tokens',  type=int,   default=300,
                        help='Max tokens to generate per inference (default: 300)')
    parser.add_argument('--poolsize',    type=int,   default=44,
                        help='Memory pool size in GB (default: 44)')
    parser.add_argument('--resume',      action='store_true', default=False,
                        help='Skip experiments already recorded in the CSV file')
    parser.add_argument('--seed',        type=int, default=0,
                        help='Global seed mixed into the injection index derivation. '
                             'Change to get a different draw of fault locations '
                             'while keeping all other conditions identical (default: 0)')

    args = parser.parse_args()

    logger.info("Starting fault injection experiments...")

    # Load model spec
    with open(args.model_config) as f:
        model_spec = json.load(f)
    logger.info(f"Loaded model config from {args.model_config}")

    # Load prompts (and optional ground-truth labels)
    prompts, labels = load_prompts(args)
    if not prompts:
        logger.error("No prompts loaded. Exiting.")
        exit(1)

    # Configure runtime options
    llama_config = {
        'temperature': args.temperature,
        'topp':        args.topp,
        'max':         args.max_tokens,
        'poolsize':    args.poolsize,
        'fp16':        args.fp16,
        'precision':   args.precision,
        'onnxdir':     args.onnxdir,
        'layer_files': args.layer_files,
    }
    
    if args.precision == 'float16':
        bit_range = range(16)
    elif args.precision == 'int8':
        bit_range = range(8)
    elif args.precision == 'int4':
        bit_range = range(4)
    else:
        raise ValueError(f"Unsupported precision: {args.precision}")

    # Create model instance
    persistent_llama = Llama(onnxdir=llama_config['onnxdir'], config=llama_config, model_spec=model_spec)

    # Create CSV file for results — name encodes model, precision, and dataset
    # so different experiment configurations don't overwrite each other.
    model_tag   = os.path.basename(args.onnxdir.rstrip('/\\'))
    dataset_tag = (os.path.splitext(os.path.basename(args.csv))[0]
                   if args.csv else args.dataset.split('/')[-1])
    csv_filename = f'results_{model_tag}_{args.precision}_{dataset_tag}.csv'
    file_exists = os.path.isfile(csv_filename)
    fieldnames = [
        'Timestamp', 'Prompt', 
        'Layer_Config', 'Fault_Model', 'Bit_Position',
        'Target_Decoder_Idx', 'Target_Token_Idx', 'Experiment',
        'Ground_Truth_Label','Golden_Raw_Output', 'Faulty_Raw_Output',
        'Golden_Token', 'Faulty_Token',
    ]
    with open(csv_filename, 'a' if file_exists else 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    completed = set()
    if args.resume and file_exists:
        with open(csv_filename, newline='') as f:
            for row in csv.DictReader(f):
                completed.add((row['Layer_Config'], row['Fault_Model'],
                               row['Bit_Position'], row['Experiment']))
        logger.info(f"Resume: {len(completed)} completed experiments loaded from {csv_filename}")

    for layer_file in os.listdir(llama_config['layer_files']):
        config_path = os.path.join(llama_config['layer_files'], layer_file)
        if os.path.isdir(config_path):
            continue

        config = json.load(open(config_path))
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing layer configuration: {layer_file}")
        logger.info(f"{'='*40}")

        for fault_model in ['INPUT', 'WEIGHT', 'INPUT16', 'WEIGHT16']:

            # Skip the entire (layer, fault_model) block if every combination is
            # already recorded — avoids graph build and GPU memory allocation.
            if completed:
                all_done = all(
                    (layer_file, fault_model, str(bp), str(exp)) in completed
                    for bp in bit_range for exp in range(len(prompts))
                )
                if all_done:
                    logger.info(f"Resume: skipping {fault_model} for {layer_file} (all done)")
                    continue

            # Build the injected graph once per (layer_config, fault_model).
            # bit_position is now a runtime feed-dict input (bit_pos_inject), so
            # the same model file covers the whole bit_range without rebuilding.
            logger.info(f"Creating faulty decoder for {fault_model}...")
            faulty_path = modify_onnx_graph(config, llama_config, fault_model)

            target_decoder_idx = extract_decoder_idx(
                faulty_path, model_spec.get('decoder_template', 'decoder-merge-{}.onnx'))

            for bit_position in bit_range:

                print(f"\n{'-'*40}")
                print(f"Layer: {layer_file}, Fault Model: {fault_model}, Bit: {bit_position}")
                print(f"{'-'*40}")

                random_seed = (bit_position * 1000 + 1)
                persistent_llama.seed = random_seed

                # fault_config carries bit_position so _faulty_decode can feed
                # bit_pos_inject into the ORT session at inference time.
                fault_config = {
                    'target_decoder_idx': target_decoder_idx,
                    'target_token_idx': 0,
                    'faulty_decoder_path': faulty_path,
                    'bit_position': bit_position,
                }
                persistent_llama.fault_config = fault_config

                for experiment, (prompt, label) in enumerate(zip(prompts, labels)):
                    print(f"\nRunning experiment {experiment}")

                    if (layer_file, fault_model, str(bit_position), str(experiment)) in completed:
                        print(f"Resume: skipping (already recorded)")
                        continue

                    # Derive a deterministic seed for rand_idx_inject from the
                    # full experiment key so the injected element index is
                    # reproducible across restarts and comparable across conditions.
                    # args.seed lets you run the full suite with a different draw
                    # of fault locations without changing anything else.
                    inject_seed = hash((args.seed, layer_file, fault_model, bit_position, experiment)) & 0xFFFFFFFF
                    persistent_llama.fault_config['inject_seed'] = inject_seed

                    # Golden run
                    print("Running golden inference...")
                    golden_result = persistent_llama.process_prompt(prompt)

                    # Faulty run
                    print("Running faulty inference...")
                    faulty_result = persistent_llama.process_prompt_faulty(prompt)

                    # Display comparison
                    output_changed = golden_result['golden_output'] != faulty_result['faulty_output']
                    print("\nCOMPARISON RESULTS:")
                    print(f"{'='*40}")
                    print(f"Output Changed: {'YES' if output_changed else 'NO'}")

                    # Save to CSV
                    with open(csv_filename, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({
                            'Timestamp': datetime.now().isoformat(),
                            'Prompt': prompt,
                            'Layer_Config': layer_file,
                            'Fault_Model': fault_model,
                            'Bit_Position': bit_position,
                            'Target_Decoder_Idx': fault_config['target_decoder_idx'],
                            'Target_Token_Idx': fault_config['target_token_idx'],
                            'Experiment': experiment,
                            'Ground_Truth_Label': label,
                            'Golden_Raw_Output': golden_result['golden_output'],
                            'Faulty_Raw_Output': faulty_result['faulty_output'],
                            'Golden_Token': golden_result['golden_token'],
                            'Faulty_Token': faulty_result['faulty_token'],
                        })

            # Clean up the faulty decoder after all bit positions are done —
            # explicit del before gc to release GPU memory promptly.
            if faulty_path in persistent_llama.faulty_decoders:
                sess = persistent_llama.faulty_decoders.pop(faulty_path)
                del sess
            if os.path.exists(faulty_path):
                os.remove(faulty_path)

            gc.collect()

    logger.info("\nExperiments completed.")
    logger.info(f"Results saved to {csv_filename}")