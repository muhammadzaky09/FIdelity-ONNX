# in llama/tokenizer.py
from transformers import AutoTokenizer
from typing import List
from loguru import logger
import os

class Tokenizer:
    def __init__(self, model_path: str):
        # Ignore model_path and use HuggingFace tokenizer
        logger.info("Using HuggingFace AutoTokenizer instead of model file")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            token=""
        )
        
        # Map properties to match original interface
        self.n_words = len(self.tokenizer.vocab)
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_id
        
        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
    
    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        # Get token IDs without special tokens
        t = self.tokenizer.encode(s, add_special_tokens=False, return_tensors=None)
        
        # Add special tokens manually as required
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t
    
    def decode(self, t: List[int]) -> str:
        # Handle numpy arrays
        if hasattr(t, 'tolist'):
            t = t.tolist()
        return self.tokenizer.decode(t, skip_special_tokens=False)