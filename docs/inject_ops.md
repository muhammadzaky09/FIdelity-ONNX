# inject_ops.py

Low-level ONNX subgraph constructors. Each function returns a list of
`onnx.NodeProto` objects that `graph.py` inserts into the graph.
No file I/O or graph loading — pure node building.

---

## Fault model functions

### INT8 / INT4 / quantized models

#### `create_quantized_fault_injection`
```python
create_quantized_fault_injection(input_name, output_name,
                                  fp16=False, is_signed=True,
                                  rand_idx_name="rand_idx_inject",
                                  bit_pos_name="bit_pos_inject")
```
Used by **INPUT**, **WEIGHT**, **INPUT16**, **WEIGHT16** on INT8 and INT4
campaigns. The helper builds an INT8 or UINT8 bit-flip delta; `graph.py` selects
signedness from the requested precision or from the Q/DQ source tensor type.

Produces a delta tensor representing the effect of flipping the bit at
`bit_pos_inject` in the integer representation of one element at `rand_idx_inject`:

```
bitmask = Cast(BitShift(1_u8, Cast(bit_pos_inject, UINT8), "LEFT"), INT8)
delta   = Cast(XOR(Cast(src, INT8), bitmask), FLOAT) - Cast(src, FLOAT)
```

Both `rand_idx_inject` and `bit_pos_inject` are external graph inputs supplied
at inference time. The delta is later added to the cloned path output by `graph.py`.

---

### FP16 models

#### `create_fp16_fault_injection`
```python
create_fp16_fault_injection(input_name, output_name,
                             fp32=False, rand_idx_name="rand_idx_inject",
                             bit_pos_name="bit_pos_inject")
```
Used by **INPUT**, **WEIGHT**, **INPUT16**, **WEIGHT16** on FP16 models.

Uses the `BitFlip` custom op (`custom.bitflip`) to flip one bit, then computes
the delta in FP32 to avoid catastrophic cancellation:

```
perturbed = BitFlip(src, bit_pos_inject, rand_idx_inject)  # bit-exact FP16 flip
delta_f32 = Cast(perturbed, F32) - Cast(src, F32)          # subtract in FP32
delta     = Cast(delta_f32, F16)                           # cast back
```

`bit_pos_inject` (INT32 scalar) is an external graph input supplied at inference
time. Set `fp32=True` when the source tensor is FP32 (adds Cast in/out around the FP16 op).

---

### Mask functions (INPUT16 / WEIGHT16)

#### `create_input16_mask` / `create_conv_input16_mask`
Restricts the delta to a contiguous block of 16 output channels, as specified
by FIdelity's INPUT16 fault model.

#### `create_weight16_mask` / `create_conv_weight16_mask`
Restricts the delta to a contiguous block of 16 spatial positions in the weight
dimension, as specified by FIdelity's WEIGHT16 fault model.

#### `create_fc_input16_mask` / `create_fc_weight16_mask`
Restricts FC-like layer deltas when `graph.py` sees `Linear`,
`FullyConnected`, `Gemm`, or a config with `"layer_type": "FC"`.

For Conv outputs, `create_conv_input16_mask` keeps 16 consecutive output
channels at one `(N, H, W)` position, while `create_conv_weight16_mask` keeps up
to 16 consecutive positions along the output width for one `(N, C, H)` group.

---

### RANDOM models

#### `create_random_fault_injection`
```python
create_random_fault_injection(output_name, random_value, fp16=True,
                               rand_idx_name="rand_idx_inject")
```
Used by **RANDOM**. Overwrites one element of the output tensor with `random_value`:

```
flat   = Reshape(output, [-1])
faulty = ScatterND(flat, [[rand_idx]], [random_value])
output_faulty = Reshape(faulty, orig_shape)
```

Works on both FP16 and FP32 tensors. `random_value` is baked in as a constant;
`rand_idx` is supplied at inference time.

---

#### `create_random_bitflip_injection`
```python
create_random_bitflip_injection(output_name, fp16=True,
                                 rand_idx_name="rand_idx_inject",
                                 bit_pos_name="bit_pos_inject")
```
Used by **RANDOM_BITFLIP** on FP16. Delegates directly to the `BitFlip` custom op:

```
faulty = BitFlip(output, bit_pos_inject, rand_idx_inject)   # domain: custom.bitflip
```

Both `rand_idx_inject` and `bit_pos_inject` are external graph inputs supplied at
inference time. Requires `llama/onnx_bitflip.so` registered at session creation.

---

#### `create_random_bitflip_fp32`
```python
create_random_bitflip_fp32(output_name,
                            rand_idx_name="rand_idx_inject",
                            bit_pos_name="bit_pos_inject")
```
Used by **RANDOM_BITFLIP** on FP32. Uses the Python custom op `DirectBitToggleFp32`
(`ai.onnx.contrib` domain) from `onnxruntime_extensions`:

```
elem          = GatherND(flat, [[rand_idx_inject]])
flipped       = DirectBitToggleFp32(elem, bit_pos_inject)
faulty        = ScatterND(flat, [[rand_idx_inject]], flipped)
output_faulty = Reshape(faulty, orig_shape)
```

`bit_pos_inject` (INT32 scalar) is an external graph input supplied at inference time.

---

## Custom operators required at runtime

| Function | Domain | Library |
|----------|--------|---------|
| `create_fp16_fault_injection` | `custom.bitflip` | `llama/onnx_bitflip.so` |
| `create_random_bitflip_injection` | `custom.bitflip` | `llama/onnx_bitflip.so` |
| `create_random_bitflip_fp32` | `ai.onnx.contrib` | `onnxruntime_extensions` |

`DirectBitToggleFp32` (Python) is also registered via `onnxruntime_extensions` for
FP32 bit-flip operations on CPU.
