# ONNX Transformer Fault Injection — Overview

Fault injection framework for LLaMA and CNN ONNX models, implementing the
[FIdelity](https://ieeexplore.ieee.org/document/9251852/) software fault models
for NVDLA-style accelerators.

---

## Repository layout

```
.
├── graph.py              # Injects fault nodes into an ONNX file
├── inject_ops.py         # Building blocks: ONNX subgraph constructors per fault model
├── llm_inference.py      # End-to-end inference runner: golden + faulty, saves CSV
├── cnn_inference.py      # CNN single-image runner: golden + faulty, saves CSV
├── parser.py             # Parses ONNX MatMul/Conv/FC-like ops → injection JSON configs
├── axes_parser.py        # ONNX graph patches (ReduceMean/Max axes, initializer handling)
├── int8/                 # INT8 quantized export helpers + SmoothQuant flow
├── Dockerfile            # CUDA 12.4 runtime image with built onnx_bitflip.so
├── docker_setup.sh       # Downloads ONNX Runtime dev files and builds the image
├── onnx-cuda-bitflip/    # Custom CUDA BitFlip op source
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
[docs/quant.md](docs/quant.md).

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

## Setup and run paths

There are two separate inference entry points:

- Use `llm_inference.py` for per-layer-split transformer decoder models such as
  LLaMA. This path needs decoder ONNX files, a model spec JSON, prompt data, and
  LLM layer configs.
- Use `cnn_inference.py` for single-image CNN experiments. This path needs a CNN
  ONNX model, image dataset selection, and CNN layer configs.

Both paths use the same parser and graph injection code, but the model
preparation and runtime arguments are different.

### Shared environment setup

#### Local environment

Activate the project environment, then run the setup script:

```bash
source ~/miniconda3/bin/activate fidelity-onnx
source setup.sh
```

`setup.sh` installs `requirements.txt`, adds `llama/` to `LD_LIBRARY_PATH` so
ONNX Runtime can load `onnx_bitflip.so`, and installs PyTorch from the CUDA 12.4
wheel index.

GPU inference with FP16 models requires **CUDA 12** and **cuDNN 9**.

#### Docker environment

The Docker flow downloads ONNX Runtime GPU 1.20.1 development files, builds the
CUDA `BitFlip` custom op from `onnx-cuda-bitflip/`, copies the resulting
`onnx_bitflip.so` into `llama/`, installs Python dependencies, and builds the
`onnx-transformer:local` image.

```bash
bash docker_setup.sh
docker run --gpus all -it onnx-transformer:local
```

The image is based on CUDA 12.4.1 and installs PyTorch with the CUDA 12.4 wheel
index.

---

## LLM inference setup (`llm_inference.py`)

Follow this path for transformer decoder experiments.

### 1. Prepare the LLM ONNX directory

Place the model files in one directory such as `decoders/7B16/` or
`decoders/fp16/`. The default LLaMA config expects:

- Split decoder files named by `decoder_template`, for example
  `decoder-merge-0.onnx`, `decoder-merge-1.onnx`, ...
- `tokenizer.model`
- `embed.onnx`
- `norm.onnx`
- `head.onnx`

The exact filenames come from the selected `--model_config` JSON.

If you need FP16 ONNX models, convert FP32 exports with
[tools/convert-fp32-to-fp16.py](tools/convert-fp32-to-fp16.py). NVDLA does not
natively support FP32 matrix multiplication.

### 2. Prepare the LLM model config

Copy [configs/llama_7b.json](configs/llama_7b.json) and edit it for the model.
This file defines the decoder count, filename template, tokenizer filename,
embedding/norm/head filenames, tensor names, and KV-cache dimensions.

Netron is useful for checking the tensor names. See
[docs/llm_inference.md](docs/llm_inference.md) for the field reference.

### 3. Generate LLM layer configs

Run `parser.py` on the LLM ONNX directory. For transformer decoder files, target
`MatMul` unless you intentionally need other ops.

```bash
python parser.py decoders/7B16/ --output_dir injection_llm --ops MatMul
```

Each run produces JSON files like:

```text
injection_llm/decoder-merge-8__self_attn_q_proj_MatMul.json
```

Each JSON has this shape:

```json
{
    "input_tensor": "<tensor name>",
    "target_layer": "<MatMul node name or output tensor name>",
    "weight_tensor": "<tensor name>",
    "model_name": "decoders/7B16/decoder-merge-8.onnx",
    "layer_type": "MatMul"
}
```

### 4. Prepare prompts

`llm_inference.py` requires exactly one prompt source:

- `--csv prompts.csv` for a local CSV.
- `--dataset cais/mmlu` for a Hugging Face dataset.

Use `--prompt_field` to select the prompt column. Use `--label_field` only when
you want a ground-truth label copied into the output CSV.

### 5. Run LLM fault injection

```bash
python llm_inference.py \
  --csv prompts.csv \
  --prompt_field question \
  --model_config configs/llama_7b.json \
  --onnxdir decoders/7B16 \
  --layer_files injection_llm \
  --precision int8
```

`llm_inference.py` runs golden and faulty inference for every combination of
`(layer config, fault model, bit position, prompt)` and appends results to
`results_<onnxdir>_<precision>_<dataset>.csv`.

Common LLM arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv PATH` | required unless `--dataset` is used | Local CSV file containing prompts |
| `--dataset NAME` | required unless `--csv` is used | Hugging Face dataset name |
| `--prompt_field` | `question` | Column name to use as the prompt string |
| `--label_field` | *(none)* | Column name to record as ground-truth label |
| `--dataset_split` | `test` | Hugging Face dataset split |
| `--model_config PATH` | `configs/llama_7b.json` | Model spec JSON |
| `--onnxdir` | `alpaca` | Directory containing tokenizer, embedding, decoder, norm, and head ONNX files |
| `--layer_files` | `injection_llm` | Directory with LLM layer config JSON files |
| `--precision` | `int8` | `int8`, `float16`, or `float32` |
| `--fp16` / `--no_fp16` | `--fp16` | Enable or disable FP16 inference |
| `--temperature` | `0.001` | Sampling temperature |
| `--topp` | `0.1` | Top-p nucleus sampling |
| `--max_tokens` | `300` | Max tokens to generate per inference |
| `--poolsize` | `44` | Memory pool size in GB |
| `--resume` | *(off)* | Skip experiments already recorded in the CSV |
| `--seed` | `0` | Seed mixed into deterministic injection index selection |

See [docs/llm_inference.md](docs/llm_inference.md) for the full LLM behavior,
CSV columns, resume logic, and model config reference.

---

## CNN inference setup (`cnn_inference.py`)

Follow this path for CNN single-image experiments.

### 1. Prepare the CNN ONNX model

Put the CNN ONNX file in a directory by itself or in a directory of CNN ONNX
files that should all be parsed. `cnn_inference.py` reads the original model path
from each generated JSON config.

### 2. Generate CNN layer configs

Run `parser.py` on the CNN ONNX directory. For CNNs, target `Conv` and FC-like
ops such as `Gemm`.

```bash
python parser.py path/to/cnn_onnx --output_dir injection_cnn --ops Conv Gemm
```

The parser writes one JSON config per target layer. CNN configs include
`layer_type` so `graph.py` can choose the Conv, FC, or MatMul mask logic for
`INPUT16` and `WEIGHT16`.

### 3. Choose the image dataset

`cnn_inference.py` supports:

- `mnist`, downloaded through `torchvision` under `./data`.
- `cifar10`, downloaded through `torchvision` under `./data`.
- `imagenet`, loaded from local Arrow shards matching
  `data/**/imagenet-1k-validation-*.arrow`.

Use `--sample_idx` to choose the test-set image.

### 4. Run CNN fault injection

```bash
python cnn_inference.py \
  --config_dir injection_cnn \
  --dataset cifar10 \
  --sample_idx 0 \
  --precision int8 \
  --fault_models INPUT WEIGHT INPUT16 WEIGHT16 \
  --provider CPUExecutionProvider
```

`cnn_inference.py` runs one image through the golden model and each injected
model, then appends prediction and L-infinity difference results to
`cnn_results_<dataset>_<precision>.csv` unless `--output_csv` is set.

Common CNN arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--config_dir` | required | Directory containing CNN layer JSON configs from `parser.py` |
| `--dataset` | required | `mnist`, `cifar10`, or `imagenet` |
| `--sample_idx` | `0` | Test-set image index |
| `--precision` | `int8` | `int8`, `int4`, `float16`, or `float32` |
| `--fault_models` | `INPUT WEIGHT INPUT16 WEIGHT16` | Fault models to run |
| `--bit_position` | *(none)* | If omitted, sweep all bits for the selected precision |
| `--provider` | `CPUExecutionProvider` | `CPUExecutionProvider` or `CUDAExecutionProvider` |
| `--seed` | `0` | Seed mixed into deterministic `rand_idx_inject` selection |
| `--output_csv` | *(none)* | Output CSV path |

For `float16` precision, use `CUDAExecutionProvider` so the CUDA `BitFlip`
custom op can be registered. `RANDOM_BITFLIP` with `float32` uses
`onnxruntime_extensions`.

See [docs/cnn_inference.md](docs/cnn_inference.md) for full CNN behavior,
precision requirements, dataset details, and CSV columns.

---

## Notes

- `rand_idx_inject` (INT64 scalar graph input) controls which element is targeted.
  Pass it at inference time for reproducibility.
- `bit_pos_inject` (INT32 scalar graph input, all fault models except `RANDOM`) controls
  which bit to flip. The same injected ONNX file covers all bit positions — no rebuild
  needed when sweeping. Bit 0 = LSB. For FP16: bit 15 = sign, bits 10-14 = exponent,
  bits 0-9 = mantissa.
- FP16 models require `CUDAExecutionProvider` — they cannot run on CPU.
