import os
import json
import re
from loguru import logger
from llama.decoder import Decoder
from llama.memory_pool import MemoryPoolSimple
from llama.utils import npsoftmax, npmultinominal2D
from llama.logits_process import warp_temperature, warp_topk, warp_topp
from llama.tokenizer import Tokenizer
from datasets import load_dataset
import csv
import numpy as np
import random
from datetime import datetime

# You can uncomment this to use your full EVALUATION_SUBJECTS list
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

    def apply_warp(self, tensor: np.array):
        tensor = warp_temperature(tensor, self.config['temperature'])
        tensor = warp_topp(tensor, self.config['top_p'])

        return tensor

    def sample(self, prompt):
        """Simple inference with the model without instruction wrapping"""
        logger.debug(f"PROMPT END: {prompt[-100:].strip()}")  # Show the end of the prompt
        
        input_ids = self.tokenizer.encode(prompt, True, False)
        input_ids = np.array(input_ids, dtype=np.int64).reshape((1, len(input_ids)))
        
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
            next_token = npmultinominal2D(probs).astype(input_ids.dtype)
            token_id = int(next_token[0, 0])
            generated_tokens.append(token_id)
            input_ids = np.concatenate([input_ids, next_token.reshape((1, 1))], axis=1)
        
            if input_ids.shape[-1] >= self.config['max'] or next_token[0,0] == self.FINISH_TOKEN:
                break
        
        response = self.tokenizer.decode(input_ids[0].tolist())
        
        # Show just the model's response (not the full text with prompt)
        model_answer = response[len(prompt):].strip()
        logger.debug(f"MODEL ANSWER: '{model_answer}'")
        
        return response

def load_datasets(subjects=None, num_samples=None):
    """Load development and test sets for MMLU evaluation"""
    try:
        # Load dev set (for few-shot examples)
        dev_dataset = load_dataset("cais/mmlu", "all", split="dev")
        if subjects:
            dev_dataset = dev_dataset.filter(lambda x: x['subject'] in subjects)
            
        # Load test set (for evaluation)
        test_dataset = load_dataset("cais/mmlu", "all", split="test")
        if subjects:
            test_dataset = test_dataset.filter(lambda x: x['subject'] in subjects)
            
        # Convert to lists for easier handling
        dev_list = [ex for ex in dev_dataset]
        test_list = [ex for ex in test_dataset]
        
        # Limit samples if requested
        if num_samples and len(test_list) > num_samples:
            random.seed(42)
            subjects = list(set(ex['subject'] for ex in test_list))
            samples_per_subject = max(1, num_samples // len(subjects))
            
            sampled_test = []
            for subject in subjects:
                subject_exs = [ex for ex in test_list if ex['subject'] == subject]
                if len(subject_exs) > samples_per_subject:
                    subject_exs = random.sample(subject_exs, samples_per_subject)
                sampled_test.extend(subject_exs)
                
            test_list = sampled_test[:num_samples]
            
        print(f"Loaded {len(dev_list)} development examples")
        print(f"Loaded {len(test_list)} test examples")
        
        return dev_list, test_list
        
    except Exception as e:
        logger.error(f"Error loading MMLU dataset: {e}")
        return None, None

def create_few_shot_prompt(dev_examples, test_example, num_shots=5):
    """Create a 5-shot prompt for MMLU with examples from dev set"""
    # Get examples of the same subject for the few-shot context
    subject = test_example['subject']
    subject_examples = [ex for ex in dev_examples if ex['subject'] == subject]
    
    # Select shot examples
    if len(subject_examples) <= num_shots:
        shot_examples = subject_examples
    else:
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



def evaluate_mmlu(model, dev_examples, test_examples, num_shots=5, csv_path=None):
    """Run MMLU evaluation with CSV tracking of each iteration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CSV file for tracking results
    if csv_path is None:
        csv_path = f'mmlu_iterations_{timestamp}.csv'
    
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Subject', 
            'Question', 
            'Correct Answer', 
            'Model Answer', 
            'Correct?', 
            'Raw Model Output'
        ])
    
    results = {}
    
    # Group test examples by subject
    subjects = list(set(ex['subject'] for ex in test_examples))
    
    all_correct = 0
    all_total = 0
    
    for subject in subjects:
        subject_examples = [ex for ex in test_examples if ex['subject'] == subject]
        correct = 0
        total = 0
        
        print(f"\n=== Evaluating subject: {subject} ===")
        print(f"Number of test examples: {len(subject_examples)}")
        
        # For the first subject only, print a sample prompt to verify format
        if subject == subjects[0]:
            sample_prompt = create_few_shot_prompt(dev_examples, subject_examples[0], num_shots)
            print("\nSAMPLE PROMPT FORMAT:")
            print("=============")
            print(sample_prompt)
            print("=============\n")
        
        for i, example in enumerate(subject_examples):
            question = example['question']
            choices = example['choices']
            answer_idx = example['answer']
            correct_letter = chr(65 + answer_idx)  # Convert 0,1,2,3 to A,B,C,D
            
            # Create the few-shot prompt
            prompt = create_few_shot_prompt(dev_examples, example, num_shots)
            
            # Run the model
            print(f"\nQuestion {i+1}: {question}")
            full_response = model.sample(prompt)
            
            # Extract the answer
            predicted_letter = extract_answer(full_response, prompt)
            model_output = full_response[len(prompt):].strip()
            
            # Check if correct
            is_correct = predicted_letter == correct_letter
            if is_correct:
                correct += 1
            total += 1
            
            print(f"Model answer: {predicted_letter or 'None'}, Correct: {correct_letter}, Correct? {is_correct}")
            
            # Save result to CSV
            with open(csv_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([
                    subject,
                    question[:100],  # Truncate long questions for CSV
                    correct_letter,
                    predicted_letter or 'None',
                    'Yes' if is_correct else 'No',
                    model_output[:200]  # Truncate long outputs for CSV
                ])
        
        # Calculate accuracy for this subject
        accuracy = correct / total if total > 0 else 0
        results[subject] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        
        all_correct += correct
        all_total += total
        
        print(f"\nSubject: {subject}, Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        # Save intermediate results after each subject
        intermediate_results = {
            "overall_accuracy_so_far": all_correct / all_total if all_total > 0 else 0,
            "results_by_subject": results
        }
        
        with open(f'mmlu_results_{timestamp}_intermediate.json', 'w') as f:
            json.dump(intermediate_results, f, indent=2)
    
    # Calculate overall accuracy
    overall_accuracy = all_correct / all_total if all_total > 0 else 0
    
    # Create summary CSV
    summary_path = f'mmlu_summary_{timestamp}.csv'
    with open(summary_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Subject', 'Accuracy', 'Correct', 'Total'])
        
        for subject, data in results.items():
            csv_writer.writerow([
                subject,
                f"{data['accuracy']:.4f}",
                data['correct'],
                data['total']
            ])
        
        # Add overall row
        csv_writer.writerow(['OVERALL', f"{overall_accuracy:.4f}", all_correct, all_total])
    
    print(f"\nEvaluation complete.")
    print(f"Iteration details saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    
    return {
        "overall_accuracy": overall_accuracy,
        "results_by_subject": results
    }

if __name__ == "__main__":
    print("Starting MMLU evaluation...")

    # Configuration for the Llama model - using very low temperature for multiple choice
    llama_config = {
        'temperature': 0.01,  # Very low temperature for multiple choice
        'topk': 40,
        'top_p': 0.9,
        'max': 1000,
        'poolsize': 39,
        'fp16': True
    }

    # Create Llama instance
    model = Llama(onnxdir='decoders/alpaca16', config=llama_config)
    
    # Set up timestamped files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iterations_csv = f'mmlu_iterations_{timestamp}.csv'
    results_json = f'mmlu_results_{timestamp}.json'
    
    # Load datasets
    dev_examples, test_examples = load_datasets(subjects=EVALUATION_SUBJECTS, num_samples=200)
    if not dev_examples or not test_examples:
        print("Failed to load datasets. Exiting.")
        exit(1)
    
    # Balance dev examples by answer to reduce bias


    
    # Run evaluation
    results = evaluate_mmlu(model, dev_examples, test_examples, num_shots=5, csv_path=iterations_csv)
    
    # Save final results
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nOverall MMLU accuracy: {results['overall_accuracy']:.4f}")
    print(f"Final results saved to {results_json}")