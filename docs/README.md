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
├── parser.py             # Parses INT8 decoder ONNX → injection JSON configs
├── parser_fp16.py        # Parses FP16 decoder ONNX → injection JSON configs
├── axes_parser.py        # ONNX graph patches (ReduceMean/Max axes, initializer handling)
├── injection_llm/        # JSON layer configs produced by parser.py / parser_fp16.py
├── decoders/
│   ├── 7B16/             # INT8-quantised decoder ONNX files
│   └── fp16/             # FP16 decoder ONNX files
├── llama/
│   ├── onnx_bitflip.so   # Custom CUDA op: BitFlip (custom.bitflip domain)
│   └── ...               # LLaMA runtime helpers (decoder, tokenizer, memory pool)
├── setup.sh              # Environment setup (pip install + LD_LIBRARY_PATH)
└── requirements.txt
```

---

## Supported fault models

| Fault model   | Description | Precision |
|---------------|-------------|-----------|
| `INPUT`       | Bit-flip in one input activation, delta propagated through the layer | INT8 / FP16 |
| `WEIGHT`      | Bit-flip in one weight value, delta propagated through the layer | INT8 / FP16 |
| `INPUT16`     | Same as INPUT but only 16 contiguous output neurons are affected | INT8 / FP16 |
| `WEIGHT16`    | Same as WEIGHT but only 16 contiguous output neurons are affected | INT8 / FP16 |
| `RANDOM`      | One output neuron overwritten with a random value | FP32 / FP16 |
| `RANDOM_BITFLIP` | One bit flipped in one output neuron | FP32 / FP16 |

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

### 3. Parse layer configs

For each decoder, run the appropriate parser to generate the injection JSON configs
that tell `graph.py` which tensors to inject into.

**INT8 models** (quantised — contain `Round` nodes):
```bash
python parser.py decoders/7B16/
# produces injection_llm/decoder-merge-{idx}_{layer}.json for each MatMul layer
```

**FP16 models** (non-quantised):
```bash
python parser_fp16.py decoders/fp16/
# produces injection_llm/decoder-merge-{idx}-fp16_{layer}.json
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
# From a text file (one prompt per line)
python llm_inference.py --prompts_file prompts.txt --onnxdir decoders/7B16

# From a HuggingFace dataset
python llm_inference.py \
    --dataset cais/mmlu --dataset_split test --prompt_field question \
    --onnxdir decoders/7B16 --precision int8 \
    --max_tokens 50 --poolsize 20
```

Results are appended to `fault_injection_results.csv`.

---

## Notes

- `rand_idx_inject` (INT64 scalar graph input) controls which element is targeted
  for `RANDOM`, `RANDOM_BITFLIP`. Pass it at inference time for reproducibility.
- `bit_position` is 0-indexed from LSB. For FP16: bit 15 = sign, bits 10-14 = exponent,
  bits 0-9 = mantissa. Use bit 14 (MSB of exponent) for a fault that survives INT8 quantisation.
- FP16 models require `CUDAExecutionProvider` — they cannot run on CPU.
