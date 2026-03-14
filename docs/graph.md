# graph.py

Modifies a single ONNX decoder file by inserting fault injection subgraph nodes
around the output of a target MatMul/Conv layer.

---

## Entry point

```python
modify_onnx_graph(config, model_config, fault_model, bit_position) -> str
```

Saves the injected model as `<original>_injected.onnx` (or `config["output_path"]`)
and returns the output path.

### `config` keys

| Key | Required for | Description |
|-----|-------------|-------------|
| `model_name` | all | Path to the source ONNX file |
| `target_layer` | all | Node name substring **or** output tensor name of the target MatMul/Conv |
| `input_tensor` | INPUT / INPUT16 / WEIGHT / WEIGHT16 | Name of the activation tensor fed into the target layer |
| `weight_tensor` | WEIGHT / WEIGHT16 | Name of the weight tensor (initializer or node output) |
| `output_path` | optional | Override output file path |

### `model_config` keys

| Key | Values |
|-----|--------|
| `precision` | `"int8"` / `"float16"` / `"float32"` |

---

## How it works

### RANDOM / RANDOM_BITFLIP

1. Find target node by matching `target_layer` against node names or output tensor names.
2. Detect whether the target output is FP16.
3. Call the appropriate `inject_ops` function to build injection nodes.
4. Insert injection nodes into the graph; re-wire all consumers of the original output
   tensor to consume the faulty output instead.
5. Add `rand_idx_inject` (INT64 scalar) as a new graph input — caller supplies it at
   inference time to control which element is targeted.

### INPUT / WEIGHT / INPUT16 / WEIGHT16

1. Use `analyze_paths_gs` to find all nodes on the path from source (input or weight)
   to target, plus the source output tensor name.
2. Clone the path: each node is duplicated with renamed tensors so the original
   forward pass is untouched.
3. Call the appropriate `inject_ops` function to build the fault subgraph
   (bit-flip + delta computation) on the cloned path's source tensor.
4. The delta (faulty − clean) propagates through the cloned path and is added to
   the original output: `faulty_out = original_out + delta`.

### Graph patching

- `patch_reduce_ops` rewrites opset-13 ReduceMean/ReduceMax to use explicit `axes`
  inputs (required for some ORT versions).
- `move_initializers_to_constant_for_matmul` converts weight initializers consumed by
  MatMul into explicit `Constant` nodes so GraphSurgeon can handle them.

---

## Custom opset imports

| Fault path | Domain added |
|------------|-------------|
| RANDOM_BITFLIP (FP16) | `custom.bitflip` |
| RANDOM_BITFLIP (FP32) | `ai.onnx.contrib` |
| INPUT/WEIGHT/INPUT16/WEIGHT16 with FP16 precision | `custom.bitflip` |

---

## Helper utilities

| Function | Purpose |
|----------|---------|
| `bin2fp32` / `bin2fp16` | Convert binary string → float value |
| `delta_init` | Generate a random valid FP32 or FP16 value for RANDOM injection |
| `_is_fp16_tensor` | Check if a named tensor is FP16 in the GraphSurgeon graph |
| `analyze_paths_gs` | BFS to find all nodes on the path from src to target node |

---

## Example

```python
from graph import modify_onnx_graph

config = {
    "model_name":    "decoders/7B16/decoder-merge-8.onnx",
    "target_layer":  "/self_attn/v_proj/MatMul",
    "input_tensor":  "/self_attn/q_proj/Round_output_0",
    "weight_tensor": "/self_attn/v_proj/Round_output_0",
}
out = modify_onnx_graph(config, {"precision": "int8"}, "INPUT", bit_position=7)
print(out)  # decoders/7B16/decoder-merge-8_injected.onnx
```
