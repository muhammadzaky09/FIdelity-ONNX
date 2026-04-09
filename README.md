# ONNX Transformer Fault Injection — Overview

Fault injection framework for LLaMA-based ONNX models, implementing the
[FIdelity](https://ieeexplore.ieee.org/document/9251852/) software fault models for NVDLA-style accelerators.

---

## Repository layout

```
.
├── graph.py              # Injects fault nodes into an ONNX file
├── inject_ops.py         # Building blocks: ONNX subgraph constructors per fault model
├── llm_inference.py      # End-to-end inference runner: golden + faulty, saves CSV
├── parser.py             # Parses any ONNX model → injection JSON configs (auto-detects quantized vs float)
├── axes_parser.py        # ONNX graph patches (ReduceMean/Max axes, initializer handling)
├── configs/
│   └── llama_7b.json     # Model spec: layer count, tensor names, file layout, KV cache dims
├── injection_llm/        # JSON layer configs produced by parser.py
├── decoders/
│   ├── 7B16/             # INT8-quantised decoder ONNX files
│   └── fp16/             # FP16 decoder ONNX files
├── llama/
│   ├── onnx_bitflip.so   # Custom CUDA op: BitFlip (custom.bitflip domain)
│   └── ...               # Runtime helpers (decoder, tokenizer, memory pool)
├── setup.sh              # Environment setup (pip install + LD_LIBRARY_PATH)
└── requirements.txt
```

---

## Validated fault models (NVDLA, spatial parallelism k = 4, reuse factor t = 16)

| Fault model   | Description | Precision |
|---------------|-------------|-----------|
| `INPUT`       | Bit-flip in one input activation before SRAM, delta propagated to all neurons that use it | INT8 / FP16 |
| `WEIGHT`      | Bit-flip in one weight value before SRAM, delta propagated to all neurons that use it | INT8 / FP16 |
| `INPUT16`     | Same as INPUT but only 16 contiguous output neurons in datapath between SRAM and MAC units are affected | INT8 / FP16 |
| `WEIGHT16`    | Same as WEIGHT but only 16 contiguous output neurons between SRAM and MAC units are affected | INT8 / FP16 |
| `RANDOM`      | Error in local control units causing one neuron outputs random erroneous value | FP32 / FP16 |
| `RANDOM_BITFLIP` | Bitflip in one neuron value inside/after MAC units  | FP32 / FP16 |

---

## End-to-end workflow

### 1. Install dependencies

```bash
source setup.sh        # installs requirements + exports LD_LIBRARY_PATH for onnx_bitflip.so
```

Requires **CUDA 12** and **cuDNN 9** for GPU inference with FP16 models.

### 2. Prepare ONNX models

Place decoder ONNX files in `decoders/7B16/` (INT8) or `decoders/fp16/` (FP16).
Expected filename pattern: `decoder-merge-{idx}.onnx`
Also Prepare the config spec for the model. See docs/llm_inference.md and configs/llama_7b.json for example

### 3. Parse layer configs

Run `parser.py` on a directory of ONNX files to generate one JSON injection config
per MatMul layer.  It auto-detects whether the model is quantized (INT8 — contains
`Round` nodes) or float (FP16/FP32) and resolves the correct injection starting
points accordingly.

```bash
python parser.py decoders/7B16/ --output_dir injection_llm
```

Each run produces `injection_llm/<model_stem>_<layer>.json` files, for example:
```
injection_llm/decoder-merge-8__self_attn_q_proj_MatMul.json
```

Each JSON has the form:
```json
{
    "input_tensor":  "<tensor name>",
    "target_layer":  "<MatMul node name or output tensor name>",
    "weight_tensor": "<tensor name>",
    "model_name":    "decoders/7B16/decoder-merge-8.onnx"
}
```


### 4. Run bulk fault injection experiments (llm_inference.py)

Runs golden + faulty inference for every combination of
`(layer config × fault model × bit position × prompt)` and saves results to CSV.

```bash
python llm_inference.py --prompts_file prompts.txt --onnxdir decoders/7B16
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompts_file PATH` | — | `.txt` (one per line) or `.json` (list of strings) |
| `--dataset NAME` | — | HuggingFace dataset name (mutually exclusive with `--prompts_file`) |
| `--dataset_split` | `test` | Dataset split to load |
| `--prompt_field` | `question` | Field used as the prompt string |
| `--model_config PATH` | `configs/llama_7b.json` | Model spec JSON — copy and edit for a different model |
| `--onnxdir` | `alpaca` | Directory containing decoder ONNX files |
| `--layer_files` | `injection_llm` | Directory with layer injection JSON configs |
| `--precision` | `int8` | weight precision `int8`, `float16`, or `float32` |
| `--fp16` / `--no_fp16` | `--fp16` | Enable/disable matrix multiplication with FP16 precision |
| `--temperature` | `0.0` | Sampling temperature |
| `--topp` | `0.1` | Top-p nucleus sampling |
| `--max_tokens` | `300` | Max tokens to generate per inference |
| `--poolsize` | `44` | Memory pool size in GB |

Results are appended to `fault_injection_results.csv`.
See `docs/llm_inference.md` for full details.

---

## Notes

- `rand_idx_inject` (INT64 scalar graph input) controls which element is targeted.
  Pass it at inference time for reproducibility.
- `bit_pos_inject` (INT32 scalar graph input, all fault models except `RANDOM`) controls
  which bit to flip. The same injected ONNX file covers all bit positions — no rebuild
  needed when sweeping. Bit 0 = LSB. For FP16: bit 15 = sign, bits 10-14 = exponent,
  bits 0-9 = mantissa.
- FP16 models require `CUDAExecutionProvider` — they cannot run on CPU.
