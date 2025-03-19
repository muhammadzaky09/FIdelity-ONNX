import numpy as np

def extract_exponent_fp16(value):
    # For a nonzero normalized FP16 value:
    abs_val = np.abs(np.float16(value))
    if abs_val == 0:
        return None  # Exponent is undefined for zero.
    unbiased = np.floor(np.log2(abs_val))
    biased = unbiased + 15  # FP16 bias is 15.
    return biased

# Example:
print(extract_exponent_fp16(0.1))  # Expected: floor(log2(256)) + 15 = 8 + 15 = 23.
