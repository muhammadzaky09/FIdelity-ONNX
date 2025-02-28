from transformers import AutoTokenizer
from loguru import logger

class Tokenizer:
    def __init__(self, model_path: str = None):
        logger.info("Loading HuggingFace AutoTokenizer for Llama-2-7b-hf")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
        logger.info(f"Vocabulary size: {len(self.tokenizer.get_vocab())}")
        logger.info(f"FINISH tokens: { self.tokenizer.eos_token_id}")
    
    def encode(self, s: str, ):
        return self.tokenizer.encode(s, add_special_tokens=True)
    
    def decode(self, token_ids):
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
