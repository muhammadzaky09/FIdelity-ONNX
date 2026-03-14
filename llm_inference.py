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


def load_prompts(args) -> list:
    """
    Load prompts as a flat list of strings from either:
      --prompts_file  path.txt   (one prompt per line)
      --prompts_file  path.json  (JSON list of strings)
      --dataset name --dataset_split split --prompt_field field
                                 (HuggingFace datasets)
    """
    if args.prompts_file:
        path = args.prompts_file
        if path.endswith('.json'):
            with open(path) as f:
                prompts = json.load(f)
            assert isinstance(prompts, list), "--prompts_file JSON must be a list of strings"
        else:
            with open(path) as f:
                prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts from {path}")
        return prompts

    # HuggingFace dataset mode
    ds = load_dataset(args.dataset, split=args.dataset_split)
    field = args.prompt_field
    prompts = [str(ex[field]) for ex in ds]
    logger.info(f"Loaded {len(prompts)} prompts from dataset '{args.dataset}' "
                f"(split={args.dataset_split}, field={field})")
    return prompts

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
            else:
                outputs = self.decoder.decode(inputs, idx)

            hidden = outputs['hidden_out']
            self.pastkeys[idx] = outputs['past_key']
            self.pastvalues[idx] = outputs['past_value']

        hidden = self.decoder.norm_head(hidden)
        return hidden

    def _faulty_decode(self, inputs: dict, idx: int):
        from llama.memory_pool import OrtWrapper
        path = self.fault_config['faulty_decoder_path']
        if path not in self.faulty_decoders:
            self.faulty_decoders[path] = OrtWrapper(path)
        faulty_handler = self.faulty_decoders[path]
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

def extract_decoder_idx(path):
    """Extract decoder index from filename"""
    import os
    filename = os.path.basename(path)
    if 'decoder-merge-' in filename:
        decoder_idx_str = filename.split('decoder-merge-')[1].split('_')[0]
        return int(decoder_idx_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM fault-injection inference")

    # Prompt source — one of the two groups is required
    prompt_src = parser.add_mutually_exclusive_group(required=True)
    prompt_src.add_argument('--prompts_file', metavar='PATH',
                            help='.txt (one per line) or .json (list of strings)')
    prompt_src.add_argument('--dataset', metavar='NAME',
                            help='HuggingFace dataset name, e.g. cais/mmlu')

    parser.add_argument('--dataset_split', default='test',
                        help='Dataset split to use (default: test)')
    parser.add_argument('--prompt_field', default='question',
                        help='Field name to use as the prompt string (default: question)')

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
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (default: 0)')
    parser.add_argument('--topp',        type=float, default=0.1,
                        help='Top-p sampling (default: 0.1)')
    parser.add_argument('--max_tokens',  type=int,   default=300,
                        help='Max tokens to generate per inference (default: 300)')
    parser.add_argument('--poolsize',    type=int,   default=44,
                        help='Memory pool size in GB (default: 44)')

    args = parser.parse_args()

    logger.info("Starting fault injection experiments...")

    # Load prompts
    prompts = load_prompts(args)
    if not prompts:
        logger.error("No prompts loaded. Exiting.")
        exit(1)

    # Configure Llama model
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

    # Create Llama instance
    persistent_llama = Llama(onnxdir=llama_config['onnxdir'], config=llama_config)

    # Create CSV file for results
    csv_filename = 'fault_injection_results.csv'
    file_exists = os.path.isfile(csv_filename)
    fieldnames = [
        'Timestamp', 'Prompt',
        'Layer_Config', 'Fault_Model', 'Bit_Position',
        'Target_Decoder_Idx', 'Target_Token_Idx', 'Experiment',
        'Golden_Raw_Output', 'Faulty_Raw_Output',
        'Golden_Token', 'Faulty_Token',
    ]
    with open(csv_filename, 'a' if file_exists else 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    for layer_file in os.listdir(llama_config['layer_files']):
        config_path = os.path.join(llama_config['layer_files'], layer_file)
        if os.path.isdir(config_path):
            continue

        config = json.load(open(config_path))
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing layer configuration: {layer_file}")
        logger.info(f"{'='*40}")

        for fault_model in ['INPUT', 'WEIGHT', 'INPUT16', 'WEIGHT16']:
            for bit_position in bit_range:

                print(f"\n{'-'*40}")
                print(f"Layer: {layer_file}, Fault Model: {fault_model}, Bit: {bit_position}")
                print(f"{'-'*40}")

                random_seed = (bit_position * 1000 + 1)
                persistent_llama.seed = random_seed

                logger.info(f"Creating faulty decoder for {fault_model} on bit position {bit_position}...")
                faulty_path = modify_onnx_graph(config, llama_config, fault_model, bit_position)

                for experiment, prompt in enumerate(prompts):
                    print(f"\nRunning experiment {experiment}")

                    # Golden run
                    print("Running golden inference...")
                    golden_result = persistent_llama.process_prompt(prompt)

                    # Set up fault config (always target first token)
                    fault_config = {
                        'target_decoder_idx': extract_decoder_idx(faulty_path),
                        'target_token_idx': 0,
                        'faulty_decoder_path': faulty_path,
                    }
                    persistent_llama.fault_config = fault_config

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
                            'Golden_Raw_Output': golden_result['golden_output'],
                            'Faulty_Raw_Output': faulty_result['faulty_output'],
                            'Golden_Token': golden_result['golden_token'],
                            'Faulty_Token': faulty_result['faulty_token'],
                        })

                # Clean up the faulty decoder — explicit del before gc to release GPU memory promptly
                if faulty_path in persistent_llama.faulty_decoders:
                    sess = persistent_llama.faulty_decoders.pop(faulty_path)
                    del sess
                if os.path.exists(faulty_path):
                    os.remove(faulty_path)

                gc.collect()

    logger.info("\nExperiments completed.")
    logger.info(f"Results saved to {csv_filename}")