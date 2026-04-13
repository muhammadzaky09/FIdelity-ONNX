# `int8/` ONNX Export Workflow

This folder contains the INT8-oriented LLaMA ONNX export path used by this
repository. The flow is based on
[tpoisonooo/llama.onnx](https://github.com/tpoisonooo/llama.onnx), but the local
scripts already integrate the SmoothQuant and fake-INT8 steps needed by this
repo.

The main difference from the rest of the project is that export is driven from a
patched Hugging Face `transformers` installation. The patched
[`modeling_llama.py`](./modeling_llama.py) exports ONNX files as a side effect of
running one generation pass.

## Folder contents

| File | Purpose |
|------|---------|
| [`export-onnx.py`](./export-onnx.py) | Entry point that loads LLaMA, applies smoothing + fake quantization, then triggers ONNX export |
| [`modeling_llama.py`](./modeling_llama.py) | Replacement for the installed Hugging Face LLaMA implementation; contains the actual ONNX export hooks |
| [`smoothquant/`](./smoothquant/) | SmoothQuant helpers for smoothing, fake quantization, calibration, and evaluation |
| [`llama-2-7b.pt`](./llama-2-7b.pt) | Precomputed activation scales used by the default export flow |

## What the exporter does

The default export flow is:

1. Load `meta-llama/Llama-2-7b-hf` through `transformers`
2. Load SmoothQuant activation scales
3. Apply `smooth_lm(..., alpha=0.85)`
4. Replace LLaMA linear layers with the local fake-INT8 wrappers from
   [`smoothquant.fake_quant`](./smoothquant/fake_quant.py)
5. Run one `generate()` call
6. Let the patched [`modeling_llama.py`](./modeling_llama.py) export:
   - `embed.onnx`
   - `decoder-merge-<layer>.onnx`
   - `norm.onnx`
   - `head.onnx`

Because SmoothQuant is already called from [`export-onnx.py`](./export-onnx.py),
you do not need to run a separate smoothing step for the default LLaMA-2 7B flow.

## Prerequisites

- A working PyTorch environment for this repo
- Access to the Hugging Face model `meta-llama/Llama-2-7b-hf`
- A local installation of Hugging Face `transformers`
- A tokenizer-compatible install, typically with `sentencepiece`

Authenticate to Hugging Face with your usual method before export, for example
through `huggingface-cli login` or environment variables. Do not commit access
tokens into this repository.

## 1. Install `transformers`

Install `transformers` first, then replace the installed LLaMA implementation
with this repo's patched file.

```bash
python -m pip install transformers sentencepiece
python -c "import inspect, transformers.models.llama.modeling_llama as m; print(inspect.getsourcefile(m))"
```

The second command prints the installed `modeling_llama.py` path, usually
something like:

```text
.../site-packages/transformers/models/llama/modeling_llama.py
```

Replace that file with [`int8/modeling_llama.py`](./modeling_llama.py).

```bash
cp int8/modeling_llama.py /path/to/site-packages/transformers/models/llama/modeling_llama.py
```

This patch is required because the stock Hugging Face file does not contain the
local ONNX export hooks used by this repo. If you upgrade `transformers`, repeat
the replacement step.

## 2. Prepare the activation scales and export directory

[`export-onnx.py`](./export-onnx.py) currently loads activation scales from
`../act_scales/llama-2-7b.pt`, so run it from inside `int8/` and place the scale
file where the script expects it.

From the repository root:

```bash
mkdir -p act_scales
cp int8/llama-2-7b.pt act_scales/llama-2-7b.pt
```

The patched [`modeling_llama.py`](./modeling_llama.py) also writes ONNX files to
`/workspace/llama.onnx/7B` by default. Create that directory first, or edit the
hardcoded `onnx_filepath` values in the patched file if you want a different
destination.

## 3. Run the export

Run the exporter from inside the `int8/` directory:

```bash
cd int8
python export-onnx.py
```

`export-onnx.py` is not a general CLI wrapper. It instantiates `Predictor`,
loads the model, applies SmoothQuant plus fake INT8 conversion, then calls one
sample `generate()` with the default prompt `"bonjour"`. The ONNX files are
exported during that forward pass.

Expected outputs under the default export path:

```text
/workspace/llama.onnx/7B/embed.onnx
/workspace/llama.onnx/7B/decoder-merge-0.onnx
/workspace/llama.onnx/7B/decoder-merge-1.onnx
...
/workspace/llama.onnx/7B/norm.onnx
/workspace/llama.onnx/7B/head.onnx
```

The patched exporter skips files that already exist. Delete old ONNX files if
you want to force a fresh export.

## SmoothQuant notes

SmoothQuant is already integrated into the default export path:

- [`export-onnx.py`](./export-onnx.py) loads activation scales and calls
  `smooth_lm(self.model, act_scales, 0.85)`
- [`smoothquant/smooth.py`](./smoothquant/smooth.py) applies the smoothing to the
  LLaMA RMSNorm and linear layers
- [`smoothquant/fake_quant.py`](./smoothquant/fake_quant.py) swaps the relevant
  modules to `W8A8Linear` wrappers

For the default flow, there is no separate manual "run SmoothQuant first" step.
That work is already wired into the exporter.

## When the helper scripts are useful

The files under [`smoothquant/`](./smoothquant/) are still useful when you want
to change the calibration data or validate the quantized model:

- [`smoothquant/calibration.py`](./smoothquant/calibration.py) collects
  activation scales from a calibration dataset. The helper expects a JSON dataset
  whose records contain a `text` field.
- [`smoothquant/ppl_eval.py`](./smoothquant/ppl_eval.py) evaluates perplexity and
  can be used to compare the base model, the smoothed model, and the quantized
  model.

Use those helpers when you are exporting a different checkpoint, changing the
domain of the prompts, or regenerating activation scales.

## Current assumptions and caveats

- The default model ID is hardcoded to `meta-llama/Llama-2-7b-hf` in
  [`export-onnx.py`](./export-onnx.py).
- The exporter assumes CUDA and moves the model to `cuda`.
- The patched [`modeling_llama.py`](./modeling_llama.py) is a direct override of
  `transformers` internals, so it may need rebasing if the installed
  `transformers` version changes significantly.
- Export paths are hardcoded inside the patched model implementation, not exposed
  as CLI flags.

## Upstream references

- Upstream export idea:
  [`tpoisonooo/llama.onnx`](https://github.com/tpoisonooo/llama.onnx)
- Hugging Face LLaMA implementation being overridden:
  [`huggingface/transformers`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
