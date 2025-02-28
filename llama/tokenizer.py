from transformers import AutoTokenizer
from loguru import logger

class Tokenizer:
    def __init__(self, model_path: str = None):
        # We ignore model_path and load the pretrained AutoTokenizer directly.
        logger.info("Loading HuggingFace AutoTokenizer for Llama-2-7b-hf")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
        # Optionally, you can log vocabulary info:
        logger.info(f"Vocabulary size: {len(self.tokenizer.get_vocab())}")
    
    def encode(self, s: str, **kwargs):
        """
        Encodes a string into token IDs.
        By default, special tokens are added automatically.
        """
        # AutoTokenizer by default inserts special tokens if its configuration requires it.
        # return_tensors="pt" returns a PyTorch tensor.
        input_ids = self.tokenizer(s, return_tensors="pt", **kwargs).input_ids
        # Convert tensor to list (squeeze to remove batch dimension)
        return input_ids.squeeze(0).tolist()
    
    def decode(self, token_ids):
        """
        Decodes a list or tensor of token IDs back to a string.
        """
        # Convert tensor to list if necessary.
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
