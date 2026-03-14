import os
import numpy as np
import onnxruntime as ort
from bit_flip_module import BitFlipOp
import struct

def display_binary(value):
    """Return the 16-bit binary representation of a float16 value."""
    return format(np.float16(value).view(np.uint16), '016b')

def compute_expected_delta(orig_val, bit_pos):
    """
    Compute expected delta by:
      1. Converting orig_val to FP16.
      2. Flipping the specified bit in the underlying 16-bit representation.
      3. Converting both original and flipped values to float32.
      4. Returning the difference, converted back to FP16.
    """
    orig_fp16 = np.float16(orig_val)
    orig_bits = orig_fp16.view(np.uint16)
    flipped_bits = orig_bits ^ (1 << bit_pos)
    flipped_fp16 = np.frombuffer(struct.pack('H', flipped_bits), dtype=np.float16)[0]
    delta = np.float32(flipped_fp16) - np.float32(orig_fp16)
    return np.float16(delta)

def test_large_tensor(bit):
    # Create a larger tensor: for example, (3, 10, 10) with random FP16 values.
    shape = (3, 10, 10)
    input_array = np.random.uniform(low=-1000, high=1000, size=shape).astype(np.float16)

    # Instantiate your custom op.
    bit_flip = BitFlipOp()

    # Run the op.
    output = bit_flip.run(input_array, bit, use_gpu=True)

    # Check that exactly one element in output is nonzero.
    nonzero_indices = np.nonzero(output)
    # Use all indices from the nonzero result.
    injected_index = tuple(idx[0] for idx in nonzero_indices)
    count_nonzero = np.prod([len(arr) for arr in nonzero_indices])
    print(f"Tensor shape: {shape}, bit position: {bit}")
    print(f"Nonzero count in output: {count_nonzero}")
    if count_nonzero != 1:
        print("FAIL: Expected exactly one nonzero element in the output.")
        return

    # Compute the flattened index using all dimensions.
    flat_index = np.ravel_multi_index(injected_index, shape)

    # Read the original FP16 value at that index.
    orig_val = input_array.flatten()[flat_index]
    expected_value_bits = np.float16(orig_val).view(np.uint16) ^ (1 << bit)
    expected_delta = np.frombuffer(struct.pack('H', expected_value_bits), dtype=np.float16)[0] - orig_val

    result_val = output.flatten()[flat_index]

    print(f"Injected at flat index: {flat_index}")
    print(f"Original FP16 value: {orig_val} (bin: {display_binary(orig_val)})")
    print(f"Computed delta from op: {result_val} (bin: {display_binary(result_val)})")
    print(f"Expected delta:       {expected_delta} (bin: {display_binary(expected_delta)})")

    if np.isclose(result_val, expected_delta, atol=1e-3):
        print("MATCH")
    else:
        print("FAIL: Mismatch in delta.")


def test_large_tensor_all_bits():
    # For each bit position, test the large tensor fault injection.
    for bit in range(16):
        test_large_tensor(bit)
        print("-" * 80)

if __name__ == '__main__':
    test_large_tensor_all_bits()
