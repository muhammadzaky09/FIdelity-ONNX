import numpy as np
from loguru import logger
from utils import npsoftmax
# refers to https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py 
def warp_topk(tensor: np.array, topk: int, fill_value = -float("Inf")):
    if topk is None or topk <= 0:
        return tensor
    assert len(tensor.shape) == 2
    
    if topk > tensor.shape[-1]:
        logger.warning('topk value {} bigger than tensor shape {}, updated'.format(topk, tensor.shape))
        topk = min(topk, tensor.shape[-1])

    for idx, pval in enumerate(tensor):
        # for each row, loop
        non_topk_idx = np.argpartition(pval, -topk)[0:-topk]
        tensor[idx][non_topk_idx] = fill_value

    return tensor

def warp_topp(tensor: np.array, top_p: float, fill_value = -float("Inf")):
    if top_p is None or top_p >= 1.0:
        return tensor
    assert len(tensor.shape) == 2
    
    for idx, pval in enumerate(tensor):
        # Sort probabilities in descending order
        sorted_indices = np.argsort(pval)[::-1]
        sorted_logits = pval[sorted_indices]
        
        # Convert to probabilities
        sorted_probs = npsoftmax(sorted_logits[None, :], axis=1)[0]
        
        # Compute cumulative probabilities
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Find cutoff index where cumulative probability exceeds p
        cutoff_idx = np.searchsorted(cumulative_probs, top_p, side='right')
        cutoff_idx = max(1, cutoff_idx)  # Always keep at least one token
        
        # Create mask for tokens to keep (those within the top-p nucleus)
        mask = np.zeros_like(pval, dtype=bool)
        mask[sorted_indices[:cutoff_idx]] = True
        
        # Set scores for tokens outside the nucleus to fill_value
        tensor[idx, ~mask] = fill_value
    
    return tensor

def warp_temperature(tensor: np.array, temperature: float):
    EPSILON = 1e-4
    if abs(temperature - 1.0) <= EPSILON:
        return tensor
    
    if temperature <= EPSILON:
        raise Exception('bad temperature {}, make sure `0.0 < temperature < 1.0`')
    
    return tensor / temperature

# copy from github.com/BLinkDL/ChatRWKV
def sample_logits(probs, temperature=1.0, top_p=0.85):
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out