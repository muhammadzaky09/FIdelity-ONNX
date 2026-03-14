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

### Pre-processing

Before any injection, two patches are applied to the raw ONNX model:

- **`patch_reduce_ops`** — rewrites `ReduceMean`/`ReduceMax` nodes to pass `axes` as an
  explicit input tensor instead of an attribute (required for opset 18 / some ORT versions).
- **`move_initializers_to_constant_for_matmul`** — promotes weight initializers consumed
  by `MatMul` into explicit `Constant` nodes so GraphSurgeon can traverse and clone them for fp16.

The patched model is then loaded into a GraphSurgeon (`gs`) graph for manipulation.

---

### RANDOM / RANDOM_BITFLIP — direct output replacement

**Target lookup**

The target node is found by iterating `graph.nodes` and matching `config["target_layer"]`
as either a **substring of `node.name`** or an **exact match of any output tensor name**.
Only `MatMul`, `Conv`, `Linear`, and `FullyConnected` ops are considered.

**Injection node conversion**

`inject_ops` functions return plain `onnx.NodeProto` objects. These are converted to
GraphSurgeon `gs.Node` objects using a shared `_get_var` helper that ensures every
intermediate tensor is represented by exactly one `gs.Variable` object (avoiding
"ghost variable" disconnection bugs). The faulty output variable inherits `dtype` and
`shape` from the original output so GraphSurgeon can export a valid `ValueInfoProto`.

**Graph insertion — position**

Injection nodes are inserted immediately after the target node in `graph.nodes`:

```
graph.nodes.insert(target_idx + 1 + i, injection_node_i)
```

**Re-wiring consumers**

After insertion, all downstream nodes that previously consumed `tgt_out` are updated
to consume `faulty_tensor` instead:

```
for node in graph.nodes:
    if node is NOT one of the injection nodes:   # avoid cycle
        for i, inp in node.inputs:
            if inp == tgt_out:
                node.inputs[i] = faulty_tensor
```

Injection nodes themselves are **excluded** from re-wiring — they intentionally read
from `tgt_out` as their source. Re-wiring them would create a cycle
(`faulty → inject → faulty`).

If `tgt_out` was a graph output (e.g. the MatMul is the final node), `graph.outputs`
is also updated so `faulty_tensor` is reachable and `graph.cleanup()` does not prune
the injection subgraph as dead code.

**`rand_idx_inject` graph input**

A new `gs.Variable("rand_idx_inject", dtype=int64, shape=[])` is appended to
`graph.inputs` before any node conversion. This makes it a named feed-dict input that
the caller must supply at inference time (e.g. `np.array(42, dtype=np.int64)`).

---

### INPUT / WEIGHT / INPUT16 / WEIGHT16 — delta propagation

**Path discovery (`analyze_paths_gs`)**

BFS from the source tensor's producer node to the target node, restricted to nodes
reachable by both directions. Returns the ordered node list `[src_node, ..., tgt_node]`.

Special case — **initializer weights**: if the weight tensor is a `gs.Constant`
(no producer node), `analyze_paths_gs` sets `weight_node = target_node` to signal a
single-element path. The clone loop then processes all nodes including `tgt_node`
(not `path[1:]`), and the `gs.Constant` input is replaced with a new `gs.Variable`
so the perturbed weight flows through.

**Path cloning**

A cloned copy of every node on the path is created with tensor names suffixed
`_fault_injected`:

```
src_out       →  src_out_fault_injected
intermediate  →  intermediate_fault_injected
tgt_out       →  tgt_out_fault_injected
```

Inputs not on the path (e.g. biases, other activations) are shared unchanged between
the original and cloned nodes. All created `gs.Variable` objects are stored in
`created_tensors` to ensure uniqueness.

**Fault injection subgraph**

`create_quantized_fault_injection` (INT8) or `create_fp16_fault_injection` (FP16) is
called on `(src_out, src_out_fault_injected)`. These nodes compute:

```
delta = perturbed(src_out) − src_out
```

producing `src_out_fault_injected` as a delta tensor with the same dtype as `src_out`.

**Graph insertion — position**

Injection nodes are inserted right after `src_node`:

```
src_idx = graph.nodes.index(src_node)
graph.nodes.insert(src_idx + 1 + i, injection_node_i)    # fault nodes first
graph.nodes.insert(src_idx + 1 + len(inj) + i, cloned_node_i)  # then cloned path
```

**Mask nodes (INPUT16 / WEIGHT16)**

For the `16`-variant models, mask nodes from `create_input16_mask` /
`create_weight16_mask` (or their Conv equivalents) are appended at the end of
`graph.nodes` via `graph.nodes.append`. They zero out all but 16 contiguous elements
of `tgt_out_fault_injected`, producing `tgt_out_fault_injected_masked`.

**Final Add node**

An `Add` node is appended:

```
faulty_out_final = Add(orig_tgt_out, tgt_out_fault_injected[_masked])
```

All downstream consumers of `orig_tgt_out` are then re-wired to `faulty_out_final`.
The original forward pass (`orig_tgt_out`) remains intact; only consumers see the
injected output.

---

### Finalisation

```python
graph.cleanup()    # remove dead nodes / tensors
graph.toposort()   # ensure topological order for ORT
model = gs.export_onnx(graph)
# opset domains added as needed (custom.bitflip / ai.onnx.contrib)
onnx.save(model, output_path)
```

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
