import numpy as np

def flip_exponent_bit_fp16(fp16_value, bit_pos):
    """
    Flips a single bit in the exponent of an FP16 value.
    
    Args:
        fp16_value (np.float16): Input FP16 value
        bit_pos (int): Bit position in the exponent (0-4, where 0=LSB)
        
    Returns:
        np.float16: New FP16 value after bitflip
    """
    # Convert FP16 to 16-bit unsigned integer representation
    uint16_repr = fp16_value.view(np.uint16)
    
    # Mask to isolate exponent bits (bits 10-14)
    EXPONENT_MASK = 0x7C00  # 0111110000000000
    
    # Extract exponent bits and shift to LSB
    exponent = (uint16_repr & EXPONENT_MASK) >> 10
    
    # Flip the specified bit in the exponent
    new_exponent = exponent ^ (1 << bit_pos)
    
    # Clear original exponent bits and set new ones
    new_uint16 = (uint16_repr & ~EXPONENT_MASK) | (new_exponent << 10)
    
    # Convert back to FP16
    return new_uint16.view(np.float16)

# Example usage
if __name__ == "__main__":
    # Test value: 1.0 in FP16 (0x3C00)
    original = np.float16(1.0)
    
    # Flip MSB of exponent (bit position 4)
    modified = flip_exponent_bit_fp16(original, 4)
    
    print(f"Original: {original} (0x{original.view(np.uint16).item():04x})")
    print(f"Modified: {modified} (0x{modified.view(np.uint16).item():04x})")