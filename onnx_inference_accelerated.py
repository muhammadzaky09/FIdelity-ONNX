import torch
from threading import Lock
from loguru import logger
import onnxruntime as ort
import numpy as np
import os
import psutil
import math
from transformers import AutoTokenizer
from datasets import load_dataset
import random
from find_op_pairs import modify_onnx_graph_input, modify_onnx_graph_weight, modify_onnx_graph_random
import json
from datetime import datetime

def singleton(cls):
    _instance = {}
    _instance_lock = Lock()

    def inner(*args, **kwargs):
        if cls not in _instance:
            with _instance_lock:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


def torchsoftmax(x, axis):
    return torch.nn.functional.softmax(x, dim=axis)

def torchgreedy2D(x):
    return torch.argmax(x, dim=1).reshape(x.shape[0], 1)

class OrtWrapper:
    def __init__(self, onnxfile: str):
        assert os.path.exists(onnxfile)
        self.onnxfile = onnxfile
        self.sess = ort.InferenceSession(
            onnxfile, 
            providers=['CUDAExecutionProvider']
        )
        self.inputs = self.sess.get_inputs()
        self.outputs = self.sess.get_outputs()
        self.input_names = [input.name for input in self.inputs]
        self.output_names = [output.name for output in self.outputs]
        logger.debug('{} loaded'.format(onnxfile))

    def forward(self, _inputs: dict):
        if len(self.inputs) != len(_inputs):
            raise ValueError(f"Expected {len(self.inputs)} inputs, got {len(_inputs)}")
        
        # Create IO binding
        io_binding = self.sess.io_binding()
        
        # Bind all input tensors
        for name, tensor in _inputs.items():
            # Convert numpy arrays to PyTorch tensors if needed
            if isinstance(tensor, np.ndarray):
                if tensor.dtype == np.float16:
                    dtype = torch.float16
                elif tensor.dtype == np.float32:
                    dtype = torch.float32
                elif tensor.dtype == np.int64:
                    dtype = torch.int64
                else:
                    dtype = torch.float32
                tensor = torch.from_numpy(tensor).to(device="cuda", dtype=dtype)
            
            # Ensure tensor is contiguous and on CUDA
            tensor = tensor.contiguous().cuda()
            
            # Bind input
            io_binding.bind_input(
                name=name,
                device_type='cuda',
                device_id=0,
                element_type=numpy_dtype_to_ort(tensor.dtype),
                shape=tensor.shape,
                buffer_ptr=tensor.data_ptr()
            )
        
        # Create output bindings on CUDA
        output_names = [output.name for output in self.outputs]
        
        for name in output_names:
            io_binding.bind_output(name, 'cuda')
        
        # Run inference
        self.sess.run_with_iobinding(io_binding)
        
        # Run normal inference instead as a workaround
        # This actually uses the same optimized CUDA kernels but handles tensor conversion for us
        inputs_numpy = {name: tensor.cpu().numpy() for name, tensor in _inputs.items()}
        outputs_numpy = self.sess.run(output_names, inputs_numpy)
        
        # Convert numpy outputs to PyTorch tensors
        outputs = {}
        for i, name in enumerate(output_names):
            outputs[name] = torch.from_numpy(outputs_numpy[i]).cuda()
        
        return outputs
    
    def __del__(self):
        logger.debug('{} unload'.format(self.onnxfile))


def numpy_dtype_to_ort(dtype):
    """Convert PyTorch dtype to ONNX Runtime data type"""
    if dtype == torch.float16:
        return np.float16
    elif dtype == torch.float32:
        return np.float32
    elif dtype == torch.int64:
        return np.int64
    elif dtype == torch.int32:
        return np.int32
    else:
        return np.float32


@singleton
class MemoryPoolSimple:
    def __init__(self, maxGB):
        if maxGB < 0:
            raise Exception('maxGB must > 0, get {}'.format(maxGB))
        
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
        used_size, biggest_k = self.used()
        while biggest_k is not None and self.max_size - used_size < need:
            need = max(need, self.max_size / 2)
            if len(self.active_map) == 0:
                break

            del self.active_map[biggest_k]
            used_size, biggest_k = self.used()
        
        self.active_map[key] = OrtWrapper(onnx)
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

class Decoder:
    
    def __init__(self, pool: MemoryPoolSimple, onnxdir: str, nameformat: str, count: int = 32):
        assert os.path.isdir(onnxdir)
        self._pool = pool
        for idx in range(count):
            filepath = os.path.join(onnxdir, nameformat.format(idx))
            self._pool.submit('decode{}'.format(idx), filepath)
        self._pool.submit('embed', os.path.join(onnxdir, 'embed.onnx'))
        self._pool.submit('norm', os.path.join(onnxdir, 'norm.onnx'))
        self._pool.submit('head', os.path.join(onnxdir, 'head.onnx'))

    def decode(self, _inputs: dict, idx: int):
        key = 'decode{}'.format(idx)
        handler = self._pool.fetch(key)
        outputs = handler.forward(_inputs)
        
        return outputs

    def embed(self, input_ids: torch.Tensor):
        handler = self._pool.fetch('embed')
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids).cuda()
            
        input_embed = handler.forward({'input': input_ids})['embed']
        return input_embed

    def norm_head(self, hidden: torch.Tensor):
        if isinstance(hidden, np.ndarray):
            hidden = torch.from_numpy(hidden).cuda()
            
        handler = self._pool.fetch('norm')
        output = handler.forward({'input': hidden})['output']

        handler = self._pool.fetch('head')
        output = handler.forward({'input': output})['output']
        return output
    
class Llama:
    def __init__(self, onnxdir='decoders/7B16', config: dict = {}):
        if not os.path.exists(onnxdir):
            logger.error('{} not exist'.format(onnxdir))

        assert os.path.isdir(onnxdir)

        self.DECODER_COUNT = 32
        # EOS token
        self.FINISH_TOKEN = 2
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token="")
        if 'poolsize' not in config or config['poolsize'] < 30:
            logger.info("Setting poolsize to 30GB for A100 performance")
            config['poolsize'] = 30
            
        pool = MemoryPoolSimple(config['poolsize'])
        self.decoder = Decoder(pool, onnxdir, 'decoder-merge-{}.onnx',
                               self.DECODER_COUNT)
        self.config = config
        self.device = 'cuda'
        self.pastkeys = [None for i in range(self.DECODER_COUNT)]
        self.pastvalues = [None for i in range(self.DECODER_COUNT)]
        
        self.faulty_decoders = {}

        pool.check()

    def _make_causal_mask(self,
                          input_ids_shape,
                          dtype,
                          past_key_values_length: int = 0):
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, dtype=dtype, device="cuda")
        mask_cond = torch.arange(mask.shape[1], device="cuda")
        mask.masked_fill_(mask_cond < (mask_cond + 1).reshape(-1, 1), 0.0)

        if past_key_values_length > 0:
            mask = torch.cat([
                torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device="cuda"), 
                mask
            ], dim=1)

        return mask.expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(self, mask, dtype, tgt_len=None):
        """  
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.  
        """
        bsz, src_len = mask.shape
        if tgt_len is None:
            tgt_len = src_len
        expanded_mask = mask.unsqueeze(1).unsqueeze(2)
        expanded_mask = expanded_mask.expand(bsz, 1, tgt_len, src_len)
        inverted_mask = 1.0 - expanded_mask
        inverted_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
        
        return inverted_mask

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = self._expand_mask(attention_mask,
                                                  inputs_embeds.dtype,
                                                  tgt_len=input_shape[-1])
            combined_attention_mask = (expanded_attn_mask
                                      if combined_attention_mask is None else
                                      expanded_attn_mask +
                                      combined_attention_mask)

        return combined_attention_mask

    def decode(self, token: torch.Tensor):
        """
        Run a forward pass through the decoder stack using PyTorch tensors
        """
        if isinstance(token, np.ndarray):
            token = torch.from_numpy(token).to(device="cuda")
            

        hidden = self.decoder.embed(token)
        assert hidden.shape[-1] == 4096

        if self.pastkeys[0] is None:
            pastlen = 0
        else:
            pastlen = self.pastkeys[0].shape[-2]
        seqlen = hidden.shape[1]

        
        position_ids = torch.arange(seqlen, dtype=torch.int64, device="cuda").reshape(1, seqlen)
        position_ids[0, 0] = pastlen

        
        attention_mask = torch.ones((1, seqlen + pastlen), dtype=torch.float32, device="cuda")
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (1, seqlen), hidden, pastlen)

        for idx in range(self.DECODER_COUNT):
            past_key = self.pastkeys[idx]
            past_value = self.pastvalues[idx]

            if past_key is None:
                zero_tensor = torch.zeros((1, 32, 0, 128), dtype=torch.float16, device="cuda")
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

            if self.config['fp16'] and hidden.dtype != torch.float16:
                inputs = {k: v.to(dtype=torch.float16) if v.dtype == torch.float32 else v 
                          for k, v in inputs.items()}
                
            outputs = self.decoder.decode(inputs, idx)

            hidden = outputs['hidden_out']
            self.pastkeys[idx] = outputs['past_key']
            self.pastvalues[idx] = outputs['past_value']

        hidden = self.decoder.norm_head(hidden)
        return hidden
    
    def apply_warp(self, tensor: torch.Tensor):
        if self.config['temperature'] != 1.0:
            tensor = tensor / self.config['temperature']
            
        if self.config['topk'] is not None and self.config['topk'] > 0:
            topk = min(self.config['topk'], tensor.shape[-1])
            
            values, _ = torch.topk(tensor, topk, dim=-1)
            min_values = values[:, -1].unsqueeze(-1).expand_as(tensor)
            tensor = torch.where(
                tensor < min_values,
                torch.full_like(tensor, float('-inf')),
                tensor
            )
            
        return tensor

    def sample_golden(self, prompt: str = 'bonjour'):
        """
        Golden run: Runs full inference with no fault injection, using PyTorch tensors.
        """
        format_prompt = PROMPT.format_map({'instruction': prompt})
        

        input_ids = self.tokenizer(format_prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Reset caches
        self.pastkeys = [None] * self.DECODER_COUNT
        self.pastvalues = [None] * self.DECODER_COUNT
        
        generated_tokens = []
        golden_token_logit = None
        token_count = 0
        next_token = input_ids
        
        while True:

            logits = self.decode(next_token)
            next_token_scores = logits[:, -1, :]

            if token_count == self.fault_config['target_token_idx']:
                golden_token_logit = next_token_scores.clone()
            
            next_token_scores = self.apply_warp(next_token_scores)
            probs = torch.softmax(next_token_scores, dim=1)
            next_token = torch.argmax(probs, dim=1).reshape(1, 1)
            token_id = int(next_token[0, 0].item())
            
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            token_count += 1

            if input_ids.shape[-1] >= self.config['max'] or token_id == self.FINISH_TOKEN:
                break

        decoded = self.tokenizer.decode(input_ids[0].tolist())

        golden_token = None
        if self.fault_config['target_token_idx'] < len(generated_tokens):
            golden_token = generated_tokens[self.fault_config['target_token_idx']]
        out = str(decoded.split('Response:')[1])
        logger.debug('Q: {} A: {}'.format(prompt, out))
            
        return out, golden_token, golden_token_logit
    
    def decode_faulty(self, token: torch.Tensor):

        if isinstance(token, np.ndarray):
            token = torch.from_numpy(token).to(device="cuda")
            
        # Embed tokens
        hidden = self.decoder.embed(token)
        assert hidden.shape[-1] == 4096

        # Get dimensions for attention masks
        pastlen = 0 if self.pastkeys[0] is None else self.pastkeys[0].shape[-2]
        seqlen = hidden.shape[1]
        
        # Create position IDs directly on GPU
        position_ids = torch.arange(seqlen, dtype=torch.int64, device="cuda").reshape(1, seqlen)
        position_ids[0, 0] = pastlen

        # Create attention mask directly on GPU
        attention_mask = torch.ones((1, seqlen + pastlen), dtype=torch.float32, device="cuda")
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (1, seqlen), hidden, pastlen)

        for idx in range(self.DECODER_COUNT):
            past_key = self.pastkeys[idx]
            past_value = self.pastvalues[idx]

            if past_key is None:
                # Create zero tensors directly on GPU
                zero_tensor = torch.zeros((1, 32, 0, 128), dtype=torch.float16, device="cuda")
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

            if self.config['fp16'] and hidden.dtype != torch.float16:
                inputs = {k: v.to(dtype=torch.float16) if v.dtype == torch.float32 else v 
                        for k, v in inputs.items()}
            
            # If this is the target decoder layer, call the faulty module
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
        
        path = self.fault_config['faulty_decoder_path']
        if path not in self.faulty_decoders:
            self.faulty_decoders[path] = OrtWrapper(path)
        
        faulty_handler = self.faulty_decoders[path]
        
        # Ensure all inputs are PyTorch tensors on CUDA
        cuda_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, np.ndarray):
                if v.dtype == np.float16:
                    dtype = torch.float16
                elif v.dtype == np.float32:
                    dtype = torch.float32
                elif v.dtype == np.int64:
                    dtype = torch.int64
                else:
                    dtype = torch.float32
                cuda_inputs[k] = torch.from_numpy(v).to(device="cuda", dtype=dtype)
            else:
                cuda_inputs[k] = v.cuda() if not v.is_cuda else v
        
        # Run the faulty model with GPU tensors using IOBinding
        outputs = faulty_handler.forward(cuda_inputs)
        
        # Return the standard outputs needed for the decoder
        return outputs

    def sample_faulty(self, prompt: str = 'bonjour'):
        format_prompt = PROMPT.format_map({'instruction': prompt})
        
        # Get input IDs as PyTorch tensor directly
        input_ids = self.tokenizer(format_prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Reset caches
        self.pastkeys = [None] * self.DECODER_COUNT
        self.pastvalues = [None] * self.DECODER_COUNT

        token_count = 0
        generated_tokens = []
        faulty_token_logit = None  # Will only be set at the target token
        next_token = input_ids
        
        while True:
            # At the target token generation, use decode_faulty
            if token_count == self.fault_config['target_token_idx']:
                logits = self.decode_faulty(next_token)
                faulty_token_logit = logits[:, -1, :].clone()
            else:
                logits = self.decode(next_token)
                
            next_token_scores = logits[:, -1, :]
            next_token_scores = self.apply_warp(next_token_scores)
            probs = torch.softmax(next_token_scores, dim=1)
            next_token = torch.argmax(probs, dim=1).reshape(1, 1)
            token_id = int(next_token[0, 0].item())
            
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)
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
            
        return out, faulty_token, faulty_token_logit

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
        'temperature': 0.001,
        'topk': 1,
        'max': 1000,
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
        for fault_model in ['INPUT','INPUT16','WEIGHT', 'WEIGHT16', 'RANDOM']:
            # For each bit position (0-7).
            for bit_position in range(8):
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
              
                for experiment in range(10):
                    # Choose the appropriate faulty model file.
                    print(f"Layer: {layer_file}, Fault Model: {fault_model}, Bit: {bit_position}, Experiment: {experiment}")
                    target_token_idx = np.random.randint(0, 10)
                    fault_config = {
                        'target_decoder_idx': extract_decoder_idx(faulty_path),
                        'target_token_idx': target_token_idx,  
                        'faulty_decoder_path': faulty_path
                    }
                    persistent_llama.fault_config = fault_config
                    # ----- Golden Run (No Fault Injection) -----
                    print("Golden Run")
                    golden_output, golden_token, golden_logits = persistent_llama.sample_golden(prompt)
                    # # ----- Faulty Run (One-Time Fault Injection) -----
                    # # Tokenize the prompt to choose a valid target token index.
                    
                    print("Faulty Run")
                    faulty_output, faulty_token, faulty_logits = persistent_llama.sample_faulty(prompt)
                    
                    golden_token_text = persistent_llama.tokenizer.decode([golden_token]) 
                    faulty_token_text = persistent_llama.tokenizer.decode([faulty_token])
                    print(f"Golden Token: {golden_token_text}, Faulty Token: {faulty_token_text}") 
                    
                    csv_filename = 'fault_injection_results2.csv'
                    file_exists = os.path.isfile(csv_filename)
                    with open(csv_filename, 'a', newline='') as csvfile:
                        csvfile.write(str(datetime.now().isoformat() + "," + str(layer_file) + "," + str(fault_model) + "," + str(bit_position) + "," + str(fault_config['target_decoder_idx']) + "," + str(fault_config['target_token_idx']) + "," + str(golden_token) + "," + str(golden_token_text) + "," + str(golden_logits) + "," + str(faulty_token) + "," + str(faulty_token_text) + "," + str(faulty_logits) + "," + str(golden_output) + "," + str(faulty_output) + "\n"))
                        # Record results
                    # Evaluation with cosine similarity, etc.
                if faulty_path is not None and faulty_path in persistent_llama.faulty_decoders:
                    del persistent_llama.faulty_decoders[faulty_path]
                    if os.path.exists(faulty_path):
                        os.remove(faulty_path)
