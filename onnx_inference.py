import os
import onnxruntime as ort
import json
import re
from loguru import logger
from llama.decoder import Decoder
from llama.memory_pool import MemoryPoolSimple
from llama.utils import npsoftmax, npmultinominal2D, npgreedy2D, seeded_npmultinomial2D
from llama.logits_process import warp_temperature, warp_topp
from llama.tokenizer import Tokenizer
import argparse
from datasets import load_dataset
import csv
from find_op_pairsfp16 import modify_onnx_graph_input_fp16, modify_onnx_graph_random_fp16, modify_onnx_graph_weight_fp16
import numpy as np
import random
from datetime import datetime
import gc


def load_mmlu_dataset():
    """Load MMLU dataset with all examples for each subject"""
    try:
        # Load dev set (for few-shot examples)
        dev_dataset = load_dataset("cais/mmlu", "all", split="dev")
            
        # Load test set (for evaluation)
        test_dataset = load_dataset("cais/mmlu", "all", split="test")
            
        # Convert to lists for easier handling
        dev_list = [ex for ex in dev_dataset]
        test_list = [ex for ex in test_dataset]
        
        # Get all unique subjects
        subjects = sorted(list(set(ex['subject'] for ex in test_list)))
        logger.info(f"Found {len(subjects)} subjects in MMLU dataset")
        
        # Organize all examples by subject (no pre-sampling)
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
    """Create a few-shot prompt for MMLU with examples from dev set"""
    # Get examples of the same subject for the few-shot context
    subject = test_example['subject']
    subject_examples = [ex for ex in dev_examples if ex['subject'] == subject]
    
    # Select shot examples
    if len(subject_examples) <= num_shots:
        shot_examples = subject_examples
    else:
        # Use deterministic sampling for reproducibility
        random.seed(42)
        shot_examples = random.sample(subject_examples, num_shots)
    
    # Build the few-shot prompt with examples
    prompt = ""
    for example in shot_examples:
        question = example['question']
        choices = example['choices']
        answer_idx = example['answer']
        correct_letter = chr(65 + answer_idx)  # Convert 0,1,2,3 to A,B,C,D
        
        prompt += f"Question: {question}\n"
        prompt += f"A. {choices[0]}\n"
        prompt += f"B. {choices[1]}\n"
        prompt += f"C. {choices[2]}\n"
        prompt += f"D. {choices[3]}\n"
        prompt += f"Answer: {correct_letter}\n\n"
    
    # Add the test question
    prompt += f"Question: {test_example['question']}\n"
    prompt += f"A. {test_example['choices'][0]}\n"
    prompt += f"B. {test_example['choices'][1]}\n"
    prompt += f"C. {test_example['choices'][2]}\n"
    prompt += f"D. {test_example['choices'][3]}\n"
    prompt += "Answer:"
    
    return prompt

def extract_answer(full_response, prompt):
    """Extract the model's answer from its response"""
    # The model's answer is after the prompt
    model_answer = full_response[len(prompt):].strip()
    
    # First try to find a direct pattern
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
    
    # Fallback: just look for the first occurrence of A, B, C, or D
    for char in model_answer:
        if char in "ABCD":
            return char
    
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
        
        # Add token length to return values
        token_length = len(generated_tokens)
        
        # Get token at the first position for experiment 0
        first_token = generated_tokens[0] if generated_tokens else None

            
        return full_response, first_token
    
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
        
        # Get the token at the target position
        faulty_token = None
        if self.fault_config['target_token_idx'] < len(generated_tokens):
            faulty_token = generated_tokens[self.fault_config['target_token_idx']]
            
        return full_response, faulty_token

    # Add MMLU-specific methods
    def process_mmlu_example(self, test_example, dev_examples, num_shots=3):
        """Run MMLU inference and extract results"""
        # Create few-shot prompt
        prompt = create_few_shot_prompt(dev_examples, test_example, num_shots)
        
        # Get golden output
        golden_output, first_token = self.sample_golden(prompt)
        
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
        }
    
    def process_mmlu_example_faulty(self, test_example, dev_examples, prompt, num_shots=5):
        """Run faulty MMLU inference and extract results"""
        # Get faulty output
        faulty_output, faulty_token = self.sample_faulty(prompt)
        
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
            'model_output': model_output,  # Just the answer part without prompt
            'predicted': predicted_letter,
            'correct': correct_letter,
            'is_correct': (predicted_letter == correct_letter) if predicted_letter else False,
            'faulty_token': faulty_token
        }

def extract_decoder_idx(path):
    """Extract decoder index from filename"""
    import os
    filename = os.path.basename(path)
    if 'decoder-merge-' in filename:
        decoder_idx_str = filename.split('decoder-merge-')[1].split('_')[0]
        return int(decoder_idx_str)

if __name__ == "__main__":
    logger.info("Starting MMLU fault injection experiments...")

    # Load dataset
    dev_examples, subject_to_examples, subjects = load_mmlu_dataset()
    if not dev_examples or not subject_to_examples or not subjects:
        logger.error("Failed to load MMLU dataset. Exiting.")
        exit(1)

    # Configure Llama model with low temperature for multiple choice
    llama_config = {
        'temperature': 0.01,
        'topp': 0.9,
        'max': 650,
        'poolsize': 48,
        'fp16': True
    }

    # Create Llama instance
    persistent_llama = Llama(onnxdir='alpaca', config=llama_config)

    # Create CSV file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'mmlu_fault_injection_results.csv'
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, 'a' if file_exists else 'w', newline='') as csvfile:
        fieldnames = [
            'Timestamp', 'Subject', 
            'Question', 'Option_A', 'Option_B', 'Option_C', 'Option_D',
            'Layer_Config', 'Fault_Model', 'Bit_Position', 
            'Target_Decoder_Idx', 'Target_Token_Idx', 'Experiment',
            'Golden_Answer', 'Faulty_Answer', 'Correct_Answer',
            'Golden_Correct', 'Faulty_Correct', 'Answer_Changed',
            'Correctness_Changed', 'Golden_Raw_Output', 'Faulty_Raw_Output', 'Golden_Token', 'Faulty_Token'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    # Track subject usage to ensure all subjects are covered
    subject_index = 0
    subjects_used = set()
    
    # Start experiment loop
    for layer_file in os.listdir("injection_llm"):
        config_path = os.path.join("injection_llm", layer_file)
        # Skip directories
        if os.path.isdir(config_path):
            continue
            
        config = json.load(open(config_path))
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing layer configuration: {layer_file}")
        logger.info(f"{'='*40}")
        
        # For each fault model
        for fault_model in ['INPUT','WEIGHT','INPUT16','WEIGHT16','RANDOM','RANDOM_BITFLIP']: 
            # For each bit position (0-7)
            for bit_position in range(8):
                # Select a subject in rotation
                curr_subject = subjects[subject_index % len(subjects)]
                subjects_used.add(curr_subject)
                examples = subject_to_examples[curr_subject]
                
                # Sample 2 different questions from this subject
                if len(examples) >= 2:
                    test_examples = random.sample(examples, 2)
                else:
                    # Fallback if subject has only one example
                    test_examples = [examples[0], examples[0]]
                
                print(f"\n{'-'*40}")
                print(f"Layer: {layer_file}, Fault Model: {fault_model}, Bit: {bit_position}")
                print(f"Using subject: {curr_subject} (Subject {subject_index % len(subjects) + 1}/57)")
                print(f"{'-'*40}")
                
                # Set seed for deterministic results
                random_seed = (bit_position * 1000 + 1)
                persistent_llama.seed = random_seed
                
                # Create faulty decoder for this configuration
                logger.info(f"Creating faulty decoder for {fault_model} on bit position {bit_position}...")
                if fault_model in ['INPUT', 'INPUT16']:
                    faulty_path = modify_onnx_graph_input_fp16(config, fault_model, bit_position)
                elif fault_model in ['WEIGHT', 'WEIGHT16']:
                    faulty_path = modify_onnx_graph_weight_fp16(config, fault_model, bit_position)
                else:
                    faulty_path = modify_onnx_graph_random_fp16(config, fault_model, bit_position)
                
                # Run both experiments with different questions
                for experiment in range(2):
                    test_example = test_examples[experiment]
                    print(f"\nRunning experiment {experiment} with token position 0 (first token)")
                    print(f"Question: {test_example['question']}")
                    
                    # Run golden inference
                    print("Running golden inference...")
                    golden_result = persistent_llama.process_mmlu_example(test_example, dev_examples)
                    
                    # Set up fault config (always target token 0)
                    fault_config = {
                        'target_decoder_idx': extract_decoder_idx(faulty_path),
                        'target_token_idx': 0,  # Always target first token
                        'faulty_decoder_path': faulty_path
                    }
                    persistent_llama.fault_config = fault_config
                    
                    # Run faulty inference
                    print("Running faulty inference...")
                    faulty_result = persistent_llama.process_mmlu_example_faulty(
                        test_example, dev_examples, golden_result['prompt']
                    )
                    
                    # Analyze changes
                    answer_changed = golden_result['predicted'] != faulty_result['predicted']
                    correctness_changed = golden_result['is_correct'] != faulty_result['is_correct']
                    
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
                            'Faulty_Token': faulty_result['faulty_token']
                        })
                
                # Clean up the faulty decoder
                if faulty_path in persistent_llama.faulty_decoders:
                    del persistent_llama.faulty_decoders[faulty_path]
                if os.path.exists(faulty_path):
                    os.remove(faulty_path)
                
                # Move to next subject and run garbage collection
                subject_index += 1
                gc.collect()

    # Print coverage statistics
    logger.info(f"\nExperiments completed.")
    logger.info(f"Used {len(subjects_used)} out of {len(subjects)} subjects")
    
    if len(subjects_used) < len(subjects):
        unused_subjects = set(subjects) - subjects_used
        logger.info(f"Unused subjects: {', '.join(unused_subjects)}")
    else:
        logger.info("All MMLU subjects were used in the experiments.")
    
    logger.info(f"Results saved to {csv_filename}")