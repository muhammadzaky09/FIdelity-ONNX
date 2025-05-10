import torch
import torch.nn.functional as F

def softmax_fp32_clipped(logits, scale=255):
    p = F.softmax(logits, dim=-1)
    q = (p * scale).round().clamp(0, scale)
    return q.div(scale)

# random test
logits = torch.randn(4, 10)
q = softmax_fp32_clipped(logits, scale=255)

# 1) All values are in [0,1]:
assert q.min() >= 0.0 and q.max() <= 1.0

# 2) Multiply back gives integers:
scaled = q * 255
assert torch.equal(scaled.round(), scaled), "Not all entries landed on integer grid!"

print("Fake‑quant softmax is snapping correctly to 1/255 steps.")
