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
├── int8/                 # INT8 Quantized export helpers + SmoothQuant flow (see int8/README.md)
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

For the patched `transformers`-based INT8 export flow, see
[int8/README.md](int8/README.md).

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
#### a. Formatting ONNX Files
Place decoder ONNX files in intended files (e.g. `decoders/7B16/` or `decoders/fp16/`).
Expected filename pattern: `decoder-merge-{idx}.onnx`

#### b. (Optional) Converting to FP16 model
NVDLA doesnt natively support FP32 matrix multiplication. To do FP16 matrix operation, export FP32 models using [convert-fp32-to-fp16.py](tools/convert-fp32-to-fp16.py)

### 3. Making model configs
Prepare the config spec for the model. See [llm_inference.md](docs/llm_inference.md) and [llama_7b.json](configs/llama_7b.json) for example. Netron is helpful

### 4. Parse layer configs

Run `parser.py` on a directory of ONNX files to generate one JSON injection config
per MatMul layer (should also work for conv as well).  It auto-detects whether the model has its weight/activations quantized (INT8 — contains
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
### 5. Prepare Prompt (Dataset)
FIdelity-ONNX supports two types of prompt source: Local CSVs and HuggingFace dataset. Check [llm_inference.md](docs/llm_inference.md) for additional information.

### 6. Run bulk fault injection experiments (llm_inference.py)

Runs golden + faulty inference for every combination of
`(layer config × fault model × bit position × prompt)` and saves results to CSV.

```bash
python llm_inference.py --prompts_file prompts.txt --onnxdir decoders/7B16
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv PATH` | — | Local CSV file containing prompts (mutually exclusive with `--dataset`) |
| `--dataset NAME` | — | HuggingFace dataset name (mutually exclusive with `--csv`) |
| `--prompt_field` | `question` | Column name to use as the prompt string (applies to both sources) |
| `--label_field` | *(none)* | Column name to record as ground-truth label in the CSV output |
| `--dataset_split` | `test` | HuggingFace dataset split to load |
| `--model_config PATH` | `configs/llama_7b.json` | Model spec JSON — copy and edit for a different model |
| `--onnxdir` | `alpaca` | Directory containing decoder ONNX files |
| `--layer_files` | `injection_llm` | Directory with layer injection JSON configs |
| `--precision` | `int8` | weight precision `int8`, `float16`, or `float32` |
| `--fp16` / `--no_fp16` | `--fp16` | Enable/disable matrix multiplication with FP16 precision |
| `--temperature` | `0.0` | Sampling temperature |
| `--topp` | `0.1` | Top-p nucleus sampling |
| `--max_tokens` | `300` | Max tokens to generate per inference |
| `--poolsize` | `44` | Memory pool size in GB |
| `--resume` | *(off)* | Skip experiments already recorded in the CSV; safe to restart interrupted runs |
| `--seed` | `0` | Global seed mixed into the injection index derivation; change to get a different draw of fault locations |

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
