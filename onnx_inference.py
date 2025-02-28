import os
import onnxruntime as ort
import json
from loguru import logger
from llama.tokenizer import Tokenizer
from llama.decoder import Decoder
from llama.memory_pool import MemoryPoolSimple
from llama.utils import cpsoftmax, cpmultinominal2D, cpgreedy2D
from llama.logits_process import warp_temperature, warp_topk
import argparse
from datasets import load_dataset
import cupy as cp
from find_op_pairs import modify_onnx_graph_input, modify_onnx_graph_weight, modify_onnx_graph_random
import numpy as np
import random

PROMPT_DICT = {
    "prompt_input":
    ("Below is an instruction that describes a task, paired with an input that provides further context. "
     "Write a response that appropriately completes the request.\n\n"
     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
     ),
    "prompt_no_input":
    ("Below is an instruction that describes a task. "
     "Write a response that appropriately completes the request.\n\n"
     "### Instruction:\n{instruction}\n\n### Response:"),
}
PROMPT = PROMPT_DICT['prompt_no_input']

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

        # cache
        self.pastkeys = [None for i in range(self.DECODER_COUNT)]
        self.pastvalues = [None for i in range(self.DECODER_COUNT)]
        
        self.faulty_decoders = {}

        pool.check()

    # Modified transformers.models.llama.modeling_llama._make_causal_mask with np.array
    def _make_causal_mask(self, input_ids_shape, dtype, past_key_values_length: int = 0):
        """
        Make causal mask for self-attention.
        Produces a lower triangular matrix (0 for allowed positions, -inf for masked ones),
        and prepends left-padding if past_key_values_length > 0.
        """
        bsz, tgt_len = input_ids_shape
        # Create a square matrix filled with -inf
        mask = cp.full((tgt_len, tgt_len), cp.finfo(dtype).min, dtype=dtype)
        
        # Create a lower triangular mask: positions where j <= i become True
        mask_cond = cp.arange(tgt_len)
        cond = mask_cond < (mask_cond + 1).reshape(-1, 1)
        # Where the condition is True, set to 0; otherwise keep -inf
        mask = cp.where(cond, 0, mask)
        
        # If past keys exist, pad the mask on the left with zeros
        if past_key_values_length > 0:
            left_padding = cp.zeros((tgt_len, past_key_values_length), dtype=dtype)
            mask = cp.concatenate([left_padding, mask], axis=1)
        
        return mask.reshape(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(self, mask, dtype, tgt_len=None):
        """
        Expands a [bsz, src_len] attention mask into shape 
        [bsz, 1, tgt_len, src_len] and converts 1-> -inf for masked positions.
        """
        mask = cp.asarray(mask)
        bsz, src_len = mask.shape
        if tgt_len is None:
            tgt_len = src_len
        # Expand dimensions to [bsz, 1, tgt_len, src_len]
        expanded_mask = cp.expand_dims(mask, axis=1)
        expanded_mask = cp.broadcast_to(expanded_mask, (bsz, 1, tgt_len, src_len))
        # Invert the mask: allowed positions become 0, disallowed become 1
        inverted_mask = 1.0 - expanded_mask
        # Where inverted_mask > 0, replace with -inf; otherwise keep 0.
        return cp.where(inverted_mask > 0, cp.finfo(dtype).min, inverted_mask)


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

    def decode(self, token: cp.ndarray):
        # embed space
        hidden = self.decoder.embed(token)
        assert hidden.shape[-1] == 4096

        if self.pastkeys[0] is None:
            pastlen = 0
        else:
            pastlen = self.pastkeys[0].shape[-2]
        seqlen = hidden.shape[1]

        position_ids = cp.arange(seqlen, dtype=cp.int64).reshape((1, seqlen))
        position_ids[0][0] = pastlen

        attention_mask = cp.ones((1, seqlen + pastlen), dtype=cp.float16)
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (1, seqlen), hidden, pastlen)

        for idx in range(self.DECODER_COUNT):
            past_key = self.pastkeys[idx]
            past_value = self.pastvalues[idx]

            if past_key is None:
                zero_tensor = cp.zeros((1, 32, 0, 128), dtype=cp.float16)
                print('check!')
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
            # del inputs
            # if 'hidden' in locals():
            #     del hidden

            hidden = outputs[
                'hidden_out']  # [[[ 0.0221,  0.0120,  0.0007,  ..., -0.0614, -0.0625,  0.0494]]]
            self.pastkeys[idx] = outputs['past_key']
            self.pastvalues[idx] = outputs['past_value']
            
            # del outputs
            # Force Cupy to free unused memory blocks
            
        hidden = self.decoder.norm_head(hidden)
        return hidden
    
    def decode_faulty(self, token: cp.ndarray):
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
        position_ids = cp.arange(seqlen, dtype=cp.int64).reshape((1, seqlen))
        position_ids[0][0] = pastlen

        attention_mask = cp.ones((1, seqlen + pastlen), dtype=cp.float16)
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (1, seqlen), hidden, pastlen)

        for idx in range(self.DECODER_COUNT):
            past_key = self.pastkeys[idx]
            past_value = self.pastvalues[idx]

            if past_key is None:
                zero_tensor = cp.zeros((1, 32, 0, 128), dtype=cp.float16)
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

    def apply_warp(self, tensor: np.array):
        tensor = warp_temperature(tensor, self.config['temperature'])
        tensor = warp_topk(tensor, self.config['topk'])
        return tensor

    def sample_golden(self, prompt: str = 'bonjour'):
        """
        Golden run: Runs full inference with no fault injection.
        """
        prompt = prompt.strip()
        format_prompt = PROMPT.format_map({'instruction': prompt})
        # Tokenize on CPU and move tokens to GPU.
        input_ids = self.tokenizer.encode(format_prompt, True, False)
        input_ids = cp.array(input_ids, dtype=cp.int64).reshape((1, len(input_ids)))

        # Reset caches.
        self.pastkeys = [None] * self.DECODER_COUNT
        self.pastvalues = [None] * self.DECODER_COUNT
        next_token = input_ids
        while True:
            # Use the standard (golden) decode.
            logits = self.decode(next_token)
            next_token_scores = next_token[:, -1, :]
            next_token_scores = self.apply_warp(next_token_scores)
            probs = cpsoftmax(next_token_scores.astype(cp.float64), axis=1)
            next_token = cpgreedy2D(probs).astype(input_ids.dtype)
            input_ids = cp.concatenate([input_ids, next_token.reshape((1, 1))], axis=1)

            if input_ids.shape[-1] >= self.config['max'] or next_token[0, 0] == self.FINISH_TOKEN:
                break

        decoded = self.tokenizer.decode(input_ids[0].tolist())
        out = str(decoded.split('Response:')[1])
        logger.debug('Q: {} A: {}'.format(prompt, out))
        return decoded
    
    def sample_faulty(self, prompt: str = 'bonjour'):
        """
        Faulty run: Runs inference with a one-time fault injection.
        For the token at fault_config['target_token_idx'], the decoder layer at
        fault_config['target_decoder_idx'] is processed using the faulty module.
        Subsequent tokens are generated normally.
        """
        prompt = prompt.strip()
        format_prompt = PROMPT.format_map({'instruction': prompt})
        input_ids = self.tokenizer.encode(format_prompt, True, False)
        input_ids = cp.array(input_ids, dtype=cp.int64).reshape((1, len(input_ids)))

        # Reset caches.
        self.pastkeys = [None] * self.DECODER_COUNT
        self.pastvalues = [None] * self.DECODER_COUNT

        token_count = 0  # CPU-based token counter.

        while True:
            # At the target token generation, use decode_faulty().
            if token_count == self.fault_config['target_token_idx']:
                hidden = self.decode_faulty(input_ids)
            else:
                hidden = self.decode(input_ids)

            next_token_scores = hidden[:, -1, :]
            next_token_scores = self.apply_warp(next_token_scores)
            probs = cpsoftmax(next_token_scores.astype(cp.float64), axis=1)
            next_token = cpgreedy2D(probs).astype(input_ids.dtype)
            input_ids = cp.concatenate([input_ids, next_token.reshape((1, 1))], axis=1)

            token_count += 1

            if input_ids.shape[-1] >= self.config['max'] or next_token[0, 0] == self.FINISH_TOKEN:
                break

        decoded = self.tokenizer.decode(input_ids[0].tolist())
        return decoded


    def _faulty_decode(self, inputs: dict, idx: int):
        from llama.memory_pool import OrtWrapper
        path = self.fault_config['faulty_decoder_path']
        if path not in self.faulty_decoders:
            self.faulty_decoders[path] = OrtWrapper(path)
        faulty_handler = self.faulty_decoders[path]
        outputs = faulty_handler.forward(inputs)
        return outputs

    
def parse_args():
    parser = argparse.ArgumentParser(description='llama.onnx onnxruntime demo')
    parser.add_argument('onnxdir', help='llama 7B onnx model directory.')
    args = parser.parse_args()
    return args

 
def prepare_guanaco_prompts(dataset_name="mlabonne/guanaco-llama2-1k", num_samples=None):
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, split="train")
        print('ya')
        # Extract prompts from the dataset
        prompts = []
        for item in dataset:
            # Extract just the instruction part from the text
            text = item['text']
            if '[INST]' in text and '[/INST]' in text:
                # Extract text between [INST] and [/INST]
                instruction = text.split('[INST]')[1].split('[/INST]')[0].strip()
                prompts.append(instruction)
        
        # Sample if requested
        if num_samples and len(prompts) > num_samples:
          
            return random.sample(prompts, num_samples)
            
        return prompts
        
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        return None

def extract_decoder_idx(path):
    import os
    filename = os.path.basename(path)
    if 'decoder-merge-' in filename:
        # Split by 'decoder-merge-' and then take the first part before an underscore.
        decoder_idx_str = filename.split('decoder-merge-')[1].split('_')[0]
        return int(decoder_idx_str)

if __name__ == "__main__":
    # print(extract_decoder_idx('decoders/7B/decoder-merge-20.onnx'))
    # Directory containing per-layer configuration files.
    # Prepare a set of prompts from the Guanaco dataset.
    prompts = prepare_guanaco_prompts("mlabonne/guanaco-llama2-1k", num_samples=1000)
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Starting experiments...")

    # Llama configuration remains the same.
    llama_config = {
        'temperature': 1,
        'topk': None,
        'max': 2000,
        'poolsize': 39,
        'fp16': True
    }

    # Create a persistent Llama instance once.
    persistent_llama = Llama(onnxdir='decoders/7B16', config=llama_config)

    # Loop over each layer configuration.
    for layer_file in os.listdir("input_llm"):
        
        config_path = os.path.join("input_llm", layer_file)
        config = json.load(open(config_path))
        print("Processing layer configuration:", layer_file)
        
        # Loop over different fault models.
        for fault_model in ['INPUT', 'INPUT16', 'WEIGHT', 'WEIGHT16', 'RANDOM']:
            # For each bit position (0-7).
            for bit_position in range(8):
                # Run several experiments for this combination.
                # if fault_model in ['INPUT', 'INPUT16']:
                #     faulty_path = modify_onnx_graph_input(config, fault_model, bit_position)
                # elif fault_model in ['WEIGHT', 'WEIGHT16']:
                #     faulty_path = modify_onnx_graph_weight(config, fault_model, bit_position)
                # else:
                #     faulty_path = modify_onnx_graph_random(config, fault_model, bit_position)
                # Pick a random prompt.
                prompt_index = np.random.randint(0, len(prompts))
                prompt = prompts[prompt_index]
                print("Prompt:", prompt)
                for experiment in range(10):
                    # Choose the appropriate faulty model file.
                    # print("Faulty model path:", faulty_path)
                    # print(extract_decoder_idx(faulty_path))
                    print(f"Layer: {layer_file}, Fault Model: {fault_model}, Bit: {bit_position}, Experiment: {experiment}")
                    
                    # ----- Golden Run (No Fault Injection) -----
                    
                    golden_output = persistent_llama.sample_golden(prompt)
                    print("Golden Output:")
                    print(golden_output)
 
                    
                    # # ----- Faulty Run (One-Time Fault Injection) -----
                    # # Tokenize the prompt to choose a valid target token index.
                    # tokenized_prompt = persistent_llama.tokenizer.encode(prompt, bos=True, eos=False)
                    # target_token = np.random.randint(0, len(tokenized_prompt))
                    # fault_config = {
                    #     'enable_fault_injection': True,
                    #     'target_decoder_idx': extract_decoder_idx(faulty_path),
                    #     'target_token_idx': 2,  # Token index for fault injection.
                    #     'faulty_decoder_path': faulty_path
                    # }
                    # persistent_llama.fault_config = fault_config
                    # persistent_llama.enable_fault_injection = True
                    
                    # faulty_output = persistent_llama.sample_faulty(prompt)
                    # print("Faulty Output:")
                    # print(faulty_output)
                    
                
                    
                    # Evaluation with cosine similarity, etc.
