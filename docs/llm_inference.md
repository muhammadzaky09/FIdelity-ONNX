# llm_inference.py

End-to-end fault injection runner for LLaMA ONNX models. For each combination of
`(layer config × fault model × bit position × prompt)` it runs a golden inference
and a faulty inference and saves the comparison to CSV.

---

## Usage

```bash
# Prompts from a local file
python llm_inference.py --prompts_file prompts.txt [options]

# Prompts from a HuggingFace dataset
python llm_inference.py --dataset cais/mmlu --dataset_split test \
                        --prompt_field question [options]
```

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
| `--onnxdir` | `alpaca` | Directory containing decoder ONNX files |
| `--layer_files` | `injection_llm` | Directory with layer injection JSON configs |
| `--precision` | `int8` | Model precision: `int8`, `float16`, `float32` |
| `--fp16` / `--no_fp16` | `--fp16` | Enable/disable FP16 inference |
| `--temperature` | `0.0` | Sampling temperature |
| `--topp` | `0.1` | Top-p nucleus sampling |
| `--max_tokens` | `300` | Max tokens to generate per inference |
| `--poolsize` | `44` | Memory pool size in GB for decoder sessions |

---

## Experiment loop

```
for each layer_file in layer_files/:
    for each fault_model in [INPUT, WEIGHT, INPUT16, WEIGHT16]:
        for each bit_position in 0..15:
            inject → faulty ONNX  (via graph.py's modify_onnx_graph)
            for each prompt:
                golden_result  = process_prompt(prompt)
                faulty_result  = process_prompt_faulty(prompt)
                append row to CSV
            delete faulty ONNX + session
```

The faulty model is re-generated and loaded fresh for each `(layer, fault_model,
bit_position)` combination, then immediately destroyed to free GPU memory.

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

| Method | Description |
|--------|-------------|
| `decode(token)` | One full forward pass through all 32 decoders (clean) |
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
