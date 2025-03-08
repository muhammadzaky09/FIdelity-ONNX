from loguru import logger
from llama.utils import singleton
import onnxruntime as ort
import numpy as np
import os
import psutil
import math
import json
from llama.decoder import Decoder
from llama.utils import npsoftmax, seeded_npmultinomial2D
from llama.logits_process import warp_temperature, warp_topk
from datasets import load_dataset
import csv
from find_op_pairs import modify_onnx_graph_input, modify_onnx_graph_weight, modify_onnx_graph_random
import random
from transformers import AutoTokenizer
from datetime import datetime

class OrtWrapper:
    def __init__(self, onnxfile: str, gpu_id = 0):
        assert os.path.exists(onnxfile)
        self.onnxfile = onnxfile
        provider_options = [{'device_id': gpu_id }]
        self.sess = ort.InferenceSession(onnxfile, [('CUDAExecutionProvider', provider_options[0])])
        self.inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.output_names = [output.name for output in outputs]
        logger.debug('{} loaded'.format(onnxfile))

    def forward(self, _inputs: dict):
        assert len(self.inputs) == len(_inputs)
        output_tensors = self.sess.run(None, _inputs)

        assert len(output_tensors) == len(self.output_names)
        output = dict()
        for i, tensor in enumerate(output_tensors):
            output[self.output_names[i]] = tensor

        return output
    
    def __del__(self):
        logger.debug('{} unload'.format(self.onnxfile))


@singleton
class MemoryPoolSimple:
    def __init__(self, maxGB, gpu_id = 0):
        if maxGB < 0:
            raise Exception('maxGB must > 0, get {}'.format(maxGB))
        self.gpu_id = gpu_id
        
        self.max_size = maxGB * 1024 * 1024 * 1024
        self.wait_map = {}
        self.active_map = {}

    def submit(self, key: str, onnx_filepath: str):
        if not os.path.exists(onnx_filepath):
            raise Exception('{} not exist!'.format(onnx_filepath))

        if key not in self.wait_map:
            self.wait_map[key] = {
                'onnx': onnx_filepath,
                'file_size': os.path.getsize(onnx_filepath)
            }

    def used(self):
        sum_size = 0
        biggest_k = None
        biggest_size = 0
        for k in self.active_map.keys():
            cur_size = self.wait_map[k]['file_size']
            sum_size += cur_size

            if biggest_k is None:
                biggest_k = k
                biggest_size = cur_size
                continue
            
            if cur_size > biggest_size:
                biggest_size = cur_size
                biggest_k = k
        
        return sum_size, biggest_k

    def check(self):
        sum_need = 0
        for k in self.wait_map.keys():
            sum_need = sum_need + self.wait_map[k]['file_size']
            
        sum_need /= (1024 * 1024 * 1024)
        
        total = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        if total > 0 and total < sum_need:
            logger.warning('virtual_memory not enough, require {}, try `--poolsize {}`'.format(sum_need, math.floor(total)))


    def fetch(self, key: str):
        if key in self.active_map:
            return self.active_map[key]
        
        need = self.wait_map[key]['file_size']
        onnx = self.wait_map[key]['onnx']

        # check current memory use
        used_size, biggest_k = self.used()
        while biggest_k is not None and self.max_size - used_size < need:
            # if exceeded once, delete until `max(half_max, file_size)` left
            need = max(need, self.max_size / 2)
            if len(self.active_map) == 0:
                break

            del self.active_map[biggest_k]
            used_size, biggest_k = self.used()
        
        self.active_map[key] = OrtWrapper(onnx, gpu_id=self.gpu_id)
        return self.active_map[key]


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
    def __init__(self, onnxdir='decoders/7B16', config: dict = {}, gpu_id = 0):
        if not os.path.exists(onnxdir):
            logger.error('{} not exist'.format(onnxdir))

        assert os.path.isdir(onnxdir)

        self.DECODER_COUNT = 32
        # EOS token
        self.FINISH_TOKEN = 2
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token="")

        pool = MemoryPoolSimple(config['poolsize'], gpu_id = self.gpu_id)
        self.decoder = Decoder(pool, onnxdir, 'decoder-merge-{}.onnx',
                               self.DECODER_COUNT)
        self.config = config
        self.device = 'cuda'
        self.gpu_id = gpu_id
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

            hidden = outputs[
                'hidden_out']  # [[[ 0.0221,  0.0120,  0.0007,  ..., -0.0614, -0.0625,  0.0494]]]
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
                print('Hm!')
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
        
        format_prompt = PROMPT.format_map({'instruction': prompt})
        input_ids = self.tokenizer(format_prompt, return_tensors="pt").input_ids.to(self.device)
        input_ids = np.array(input_ids.cpu(), dtype=np.int64).reshape(
            (1, input_ids.cpu().shape[1]))
        
        # Reset caches
        self.pastkeys = [None] * self.DECODER_COUNT
        self.pastvalues = [None] * self.DECODER_COUNT
        
        generated_tokens = []
        next_token = input_ids
        
        while True:

            logits = self.decode(next_token)
            next_token_scores = logits[:, -1, :]
            next_token_scores = self.apply_warp(next_token_scores)
            probs = npsoftmax(next_token_scores.astype(np.float64), axis=1)
            # next_token = npmultinominal2D(probs).astype(input_ids.dtype)
            next_token = seeded_npmultinomial2D(probs, self.seed).astype(input_ids.dtype)
            token_id = int(next_token[0, 0])
            generated_tokens.append(token_id)
            input_ids = np.concatenate([input_ids, next_token.reshape((1, 1))], axis=1)

            if input_ids.shape[-1] >= self.config['max'] or next_token[0,0] == self.FINISH_TOKEN:
                break

        decoded = self.tokenizer.decode(input_ids[0].tolist())
        
        # Get the token at target position
        golden_token = None
        if self.fault_config['target_token_idx'] < len(generated_tokens):
            golden_token = generated_tokens[self.fault_config['target_token_idx']]
        out = str(decoded.split('Response:')[1])
        logger.debug('Q: {} A: {}'.format(prompt, out))
            
        return out, golden_token
    
    def sample_faulty(self, prompt: str = 'bonjour'):
        format_prompt = PROMPT.format_map({'instruction': prompt})
        input_ids = self.tokenizer(format_prompt, return_tensors="pt").input_ids.to(self.device)
        input_ids = np.array(input_ids.cpu(), dtype=np.int64).reshape(
            (1, input_ids.cpu().shape[1]))

        # Reset caches
        self.pastkeys = [None] * self.DECODER_COUNT
        self.pastvalues = [None] * self.DECODER_COUNT

        token_count = 0
        generated_tokens = []
        next_token = input_ids
        
        while True:
            # At the target token generation, use decode_faulty
            if token_count == self.fault_config['target_token_idx']:
                logits = self.decode_faulty(next_token)
            else:
                logits = self.decode(next_token)
                
            next_token_scores = logits[:, -1, :]
            next_token_scores = self.apply_warp(next_token_scores)
            probs = npsoftmax(next_token_scores.astype(np.float64), axis=1)
            # next_token = npgreedy2D(probs).astype(input_ids.dtype)
            next_token = seeded_npmultinomial2D(probs, self.seed).astype(input_ids.dtype)
            token_id = int(next_token[0, 0])
            generated_tokens.append(token_id)
            input_ids = np.concatenate([input_ids, next_token.reshape((1, 1))], axis=1)
            token_count += 1

            if input_ids.shape[-1] >= self.config['max'] or token_id == self.FINISH_TOKEN:
                break

        decoded = self.tokenizer.decode(input_ids[0].tolist())
        
        # Get the token at the target position
        faulty_token = None
        if self.fault_config['target_token_idx'] < len(generated_tokens):
            faulty_token = generated_tokens[self.fault_config['target_token_idx']]
        out = str(decoded.split('Response:')[1])
        logger.debug('Q: {} A: {}'.format(prompt, out))
            
        return out, faulty_token


    def _faulty_decode(self, inputs: dict, idx: int):
        from llama.memory_pool import OrtWrapper
        path = self.fault_config['faulty_decoder_path']
        if path not in self.faulty_decoders:
            self.faulty_decoders[path] = OrtWrapper(path)
        faulty_handler = self.faulty_decoders[path]
        outputs = faulty_handler.forward(inputs)
        return outputs

    
# def parse_args():
#     parser = argparse.ArgumentParser(description='llama.onnx onnxruntime demo')
#     parser.add_argument('onnxdir', help='llama 7B onnx model directory.')
#     args = parser.parse_args()
#     return args

 
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
        'temperature': 0.1,
        'topk': 40,
        'max': 1000,
        'poolsize': 39,
        'fp16': True
    }

    # Create a persistent Llama instance once.
    golden_llama = Llama(onnxdir='decoders/7B16', config=llama_config, gpu_id=0)
    faulty_llama = Llama(onnxdir='decoders/7B16', config=llama_config, gpu_id=1)

    # Loop over each layer configuration.
    for layer_file in os.listdir("input_llm"):
        layer_file = "decoder-merge-23_q_proj.json"
        config_path = os.path.join("input_llm", layer_file)
        config = json.load(open(config_path))
        print("Processing layer configuration:", layer_file)
        
        # Loop over different fault models.
        for fault_model in [ 'RANDOM']:
            # For each bit position (0-7).
            for bit_position in range(10):
                # Run several experiments for this combination.
                if fault_model in ['INPUT', 'INPUT16']:
                    faulty_path = modify_onnx_graph_input(config, fault_model, bit_position)
                elif fault_model in ['WEIGHT', 'WEIGHT16']:
                    faulty_path = modify_onnx_graph_weight(config, fault_model, bit_position)
                else:
                    faulty_path = modify_onnx_graph_random(config, fault_model, bit_position)
                    
                # If a faulty decoder is already loaded for this path, unload it.
                # Pick a random prompt.
                prompt_index = np.random.randint(0, len(prompts))
                prompt = prompts[prompt_index]
               
                random_seed = (bit_position * 1000) 
                for experiment in range(2):
                    # Choose the appropriate faulty model file.
                    print(f"Layer: {layer_file}, Fault Model: {fault_model}, Bit: {bit_position}, Experiment: {experiment}")
                    if experiment % 2 == 0:
                        target_token_idx = 0
                    else:
                        target_token_idx = 5
                    
                    fault_config = {
                        'target_decoder_idx': extract_decoder_idx(faulty_path),
                        'target_token_idx': target_token_idx,  
                        'faulty_decoder_path': faulty_path
                    }
                    golden_llama.fault_config = fault_config
                    faulty_llama.fault_config = fault_config
                    
                    golden_llama.seed = random_seed
                    faulty_llama.seed = random_seed
                    # ----- Golden Run (No Fault Injection) -----
                    print("Golden Run")
                    golden_output, golden_token = golden_llama.sample_golden(prompt)
                    # # ----- Faulty Run (One-Time Fault Injection) -----
                    # # Tokenize the prompt to choose a valid target token index.
                    
                    print("Faulty Run")
                    faulty_output, faulty_token= faulty_llama.sample_faulty(prompt)
                    
                    golden_token_text = golden_llama.tokenizer.decode([golden_token]) 
                    faulty_token_text = faulty_llama.tokenizer.decode([faulty_token])
                    print(f"Golden Token: {golden_token_text}, Faulty Token: {faulty_token_text}") 
                    
                    csv_filename = 'fault_injection_results5.csv'
                    file_exists = os.path.isfile(csv_filename)
                    with open(csv_filename, 'a', newline='') as csvfile:
                        fieldnames = [
                            'Timestamp', 'Layer_Config', 'Fault_Model', 'Bit_Position', 
                            'Target_Decoder_Idx', 'Target_Token_Idx',
                            'Golden_Token_ID', 'Golden_Token_Text', 
                            'Faulty_Token_ID','Faulty_Token_Text',
                            'Golden_Output', 'Faulty_Output',
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        
                        # Write header if file is new
                        if not file_exists:
                            writer.writeheader()
                        
                        # Write the row as a dictionary
                        writer.writerow({
                            'Timestamp': datetime.now().isoformat(),
                            'Layer_Config': str(layer_file),
                            'Fault_Model': str(fault_model),
                            'Bit_Position': str(bit_position),
                            'Target_Decoder_Idx': str(fault_config['target_decoder_idx']),
                            'Target_Token_Idx': str(fault_config['target_token_idx']),
                            'Golden_Token_ID': str(golden_token),
                            'Golden_Token_Text': str(golden_token_text),
                            'Faulty_Token_ID': str(faulty_token),
                            'Faulty_Token_Text': str(faulty_token_text),
                            'Golden_Output': str(golden_output),
                            'Faulty_Output': str(faulty_output)
                        })
                        # Record results
                      
                    
                    # Evaluation with cosine similarity, etc.
                if faulty_path is not None and faulty_path in faulty_llama.faulty_decoders:
                    del faulty_llama.faulty_decoders[faulty_path]
                    if os.path.exists(faulty_path):
                        os.remove(faulty_path)
