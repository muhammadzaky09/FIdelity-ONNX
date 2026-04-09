# llm_inference.py

End-to-end fault injection runner for LLaMA ONNX models. For each combination of
`(layer config × fault model × bit position × prompt)` it runs a golden inference
and a faulty inference and saves the comparison to CSV.

---


## Arguments

### Prompt source (mutually exclusive, one required)

| Argument | Description |
|----------|-------------|
| `--prompts_file PATH` | `.txt` (one prompt per line) or `.json` (list of strings) |
| `--dataset NAME` | HuggingFace dataset name, e.g. `cais/mmlu` |
| `--dataset_split` | Split to load (default: `test`) |
| `--prompt_field` | Field name used as the prompt string (default: `question`) |

> **Note:** `--dataset` pulls only one field as a plain string. For multi-field
> prompts (e.g. question + answer choices), pre-format them and use `--prompts_file`.

### Model / inference config

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_config PATH` | `configs/llama_7b.json` | Path to the model spec JSON (see below) |
| `--onnxdir` | `alpaca` | Directory containing decoder ONNX files |
| `--layer_files` | `injection_llm` | Directory with layer injection JSON configs |
| `--precision` | `int8` | Model precision: `int8`, `float16`, `float32` |
| `--fp16` / `--no_fp16` | `--fp16` | Enable/disable FP16 inference |
| `--temperature` | `0.0` | Sampling temperature |
| `--topp` | `0.1` | Top-p nucleus sampling |
| `--max_tokens` | `300` | Max tokens to generate per inference |
| `--poolsize` | `44` | Memory pool size in GB for decoder sessions |

#### Model spec JSON (`--model_config`)

All model-architecture constants are externalised into a JSON file so the same
script can drive any per-layer-split ONNX model.  The default
`configs/llama_7b.json` encodes the LLaMA-7B values.

| Field | Description |
|-------|-------------|
| `decoder_count` | Number of decoder layer ONNX files |
| `eos_token_id` | Token ID that terminates generation |
| `hidden_dim` | Hidden state width (used for shape assertion) |
| `n_heads` / `head_dim` | Dimensions for the KV cache zero tensor |
| `decoder_template` | Filename pattern, e.g. `"decoder-merge-{}.onnx"` |
| `tokenizer_file` | Tokenizer filename inside `onnxdir` |
| `embed_file` / `norm_file` / `head_file` | Separate ONNX filenames for embedding, norm, and LM head |
| `input_names` | Dict mapping logical roles (`hidden`, `attn_mask`, `position_ids`, `past_key`, `past_value`) to actual ONNX input tensor names |
| `output_names` | Dict mapping logical roles (`hidden`, `past_key`, `past_value`) to actual ONNX output tensor names |
| `embed_input` / `embed_output` | Tensor names for the embedding ONNX |
| `norm_input` / `norm_output` | Tensor names for the norm ONNX |
| `head_input` / `head_output` | Tensor names for the LM head ONNX |

To support a new model, copy `configs/llama_7b.json`, fill in the new values, and
pass `--model_config configs/my_model.json`.

---

## Experiment loop

```
for each layer_file in layer_files/:
    for each fault_model in [INPUT, WEIGHT, INPUT16, WEIGHT16]:
        inject → faulty ONNX  (via graph.py's modify_onnx_graph)   # built ONCE
        for each bit_position in 0..15:
            for each prompt:
                golden_result  = process_prompt(prompt)
                faulty_result  = process_prompt_faulty(prompt)
                    # _faulty_decode injects rand_idx_inject + bit_pos_inject
                    # into the feed-dict at inference time
                append row to CSV
        delete faulty ONNX + session
```

The faulty model is built once per `(layer, fault_model)` combination. `bit_position`
is now a runtime feed-dict input (`bit_pos_inject`) so the same ONNX file is reused
across the entire bit range, which avoids repeated graph rebuilds and reduces GPU
memory churn.

---

## CSV output

Results are appended to `fault_injection_results.csv`.

| Column | Description |
|--------|-------------|
| `Timestamp` | ISO timestamp of the row |
| `Prompt` | Input prompt string |
| `Layer_Config` | JSON filename from `layer_files/` |
| `Fault_Model` | `INPUT` / `WEIGHT` / `INPUT16` / `WEIGHT16` |
| `Bit_Position` | 0–15 |
| `Target_Decoder_Idx` | Decoder index extracted from the injected filename |
| `Target_Token_Idx` | Always `0` (fault injected at first generated token) |
| `Experiment` | Prompt index within the current run |
| `Golden_Raw_Output` | Model's text output for the clean run |
| `Faulty_Raw_Output` | Model's text output for the faulty run |
| `Golden_Token` | Token ID of the first generated token (clean) |
| `Faulty_Token` | Token ID of the first generated token (faulty) |

---

## Key classes and methods

### `Llama`

Constructed with `Llama(onnxdir, config, model_spec)` where `model_spec` is the
loaded `configs/*.json` dict.  All architecture constants (layer count, tensor
names, KV cache shape, etc.) are read from `model_spec` at init time with
LLaMA-7B values as fallbacks.

| Method | Description |
|--------|-------------|
| `decode(token)` | One full forward pass through all decoder layers (clean), using tensor names from `model_spec` |
| `decode_faulty(token)` | Same, but substitutes the faulty session at `fault_config['target_decoder_idx']` |
| `sample_golden(prompt)` | Full autoregressive generation, no fault injection |
| `sample_faulty(prompt)` | Full autoregressive generation, fault injected at `target_token_idx` |
| `process_prompt(prompt)` | Wrapper around `sample_golden`; returns dict with output text + first token |
| `process_prompt_faulty(prompt)` | Wrapper around `sample_faulty`; returns dict with faulty output + faulty token |

KV cache (`pastkeys` / `pastvalues`) is reset at the start **and** end of every
`sample_golden` / `sample_faulty` call via `try/finally` to avoid GPU memory leaks
across thousands of experiments.

### `load_prompts(args)`

Returns a flat `list[str]` from either a local file or a HuggingFace dataset.

---

## Memory management

- KV cache is explicitly cleared after each `sample_golden` / `sample_faulty` call.
- The faulty `OrtWrapper` session is explicitly deleted (`del sess`) before
  `gc.collect()` so CUDA memory is released promptly.
- Lower `--poolsize` if OOM occurs between experiments.
- Lower `--max_tokens` to reduce peak KV cache size (50 is usually enough for
  fault injection studies that compare first-token outputs).
