from typing import List, Optional
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin
from smoothquant.fake_quant import quantize_llama_like
from smoothquant.smooth import smooth_lm
import torch
import sys

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
PROMPT = PROMPT_DICT['prompt_no_input']

#CACHE_DIR = 'alpaca_out'

class CustomLlamaModel(LlamaForCausalLM, GenerationMixin):
    pass

class Predictor():
    def __init__(self):
        self.device = 'cuda'
        self.model = CustomLlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf", token="", trust_remote_code=True)
        act_scales = torch.load("../act_scales/llama-2-7b.pt")
        smooth_lm(self.model, act_scales, 0.85)
        print("QUANTIZING MODEL:")
        self.model = quantize_llama_like(self.model)
        print("DONE QUANTIZATION:")
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token="", trust_remote_code=True)

    def predict(
            self,
            prompt: str = "bonjour",
            n: int = 1,
            total_tokens: int = 2000,
            temperature: float = 0.1, 
            top_p: float = 1.0,
            repetition_penalty: float = 1) -> List[str]:

        format_prompt = PROMPT.format_map({'instruction': prompt})
        _input = self.tokenizer(format_prompt, return_tensors="pt").input_ids.to(self.device)

        print("GENERATING ONNX")
        outputs = self.model.generate(
            _input,
            num_return_sequences=n,
            max_length=total_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=40,
            repetition_penalty=repetition_penalty
        )
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # removing prompt b/c it's returned with every input 
        out = [val.split('Response:')[1] for val in out]
        print('Q: {} A: {}'.format(prompt, out))
        return out

#x = Predictor(sys.argv[1])
x = Predictor()
x.predict()
