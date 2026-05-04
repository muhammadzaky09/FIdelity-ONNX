# cnn_inference.py

Single-image fault injection runner for CNN ONNX models. For each layer config,
fault model, and bit position, it runs a golden inference and a faulty inference,
then appends prediction changes and L-infinity output difference to CSV.

---

## Workflow

### 1. Generate layer configs

Use `parser.py` on the directory that contains the CNN ONNX model:

```bash
python parser.py path/to/cnn_onnx --output_dir injection_cnn --ops Conv Gemm
```

The parser targets `MatMul`, `Conv`, `Gemm`, `Linear`, and `FullyConnected` by
default. CNN configs include `layer_type` so `graph.py` can select Conv, FC, or
MatMul-specific masks for the `INPUT16` and `WEIGHT16` models.

### 2. Run inference

```bash
python cnn_inference.py \
  --config_dir injection_cnn \
  --dataset cifar10 \
  --sample_idx 0 \
  --precision int8 \
  --fault_models INPUT WEIGHT INPUT16 WEIGHT16 \
  --provider CPUExecutionProvider
```

If `--output_csv` is omitted, results are written to
`cnn_results_<dataset>_<precision>.csv`.

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config_dir` | required | Directory containing JSON configs from `parser.py` |
| `--dataset` | required | `mnist`, `cifar10`, or `imagenet` |
| `--sample_idx` | `0` | Test-set image index |
| `--precision` | `int8` | `int8`, `int4`, `float16`, or `float32` |
| `--fault_models` | `INPUT WEIGHT INPUT16 WEIGHT16` | Fault models to run; choices are `INPUT`, `WEIGHT`, `INPUT16`, `WEIGHT16`, `RANDOM`, `RANDOM_BITFLIP` |
| `--bit_position` | *(none)* | If omitted, sweep all bits for the selected precision |
| `--provider` | `CPUExecutionProvider` | ONNX Runtime provider |
| `--seed` | `0` | Mixed into deterministic `rand_idx_inject` selection |
| `--output_csv` | *(none)* | Output CSV path |

---

## Datasets

- `mnist` downloads the test split through `torchvision.datasets.MNIST` under
  `./data`.
- `cifar10` downloads the test split through `torchvision.datasets.CIFAR10`
  under `./data`.
- `imagenet` expects local Arrow shards under `data/` matching
  `**/imagenet-1k-validation-*.arrow`.

ImageNet rows are expected to contain `image` and `label` fields.

---

## Runtime behavior

For each layer config:

1. Run the original model once to get `Golden_Pred`.
2. Build a fault-injected ONNX file with `modify_onnx_graph`.
3. Create a faulty ONNX Runtime session.
4. Pick a deterministic flat `rand_idx_inject` from
   `hash((seed, layer_file, fault_model, bit_position, sample_idx))`.
5. Feed `rand_idx_inject` and, when the model exposes it, `bit_pos_inject`.
6. Save the prediction comparison and `Linf_Diff`.

`rand_idx_inject` is bounded by the target tensor size. For `INPUT` and
`INPUT16` that target is `input_tensor`; for `WEIGHT` and `WEIGHT16` it is
`weight_tensor`; for `RANDOM` and `RANDOM_BITFLIP` it is the target layer output.

`RANDOM` and `RANDOM_BITFLIP` use `graph.py`'s direct output replacement path,
which currently looks up `MatMul`, `Conv`, `Linear`, and `FullyConnected` nodes.
For `Gemm` configs, use the `INPUT`, `WEIGHT`, `INPUT16`, or `WEIGHT16` fault
models unless the model has an equivalent `Linear`/`FullyConnected` op name.

---

## Precision and custom ops

| Precision / fault path | Runtime requirement |
|------------------------|---------------------|
| INT8 / INT4 `INPUT`, `WEIGHT`, `INPUT16`, `WEIGHT16` | Standard ONNX Runtime ops |
| FP16 `INPUT`, `WEIGHT`, `INPUT16`, `WEIGHT16` | `CUDAExecutionProvider` and `llama/onnx_bitflip.so` |
| FP16 `RANDOM_BITFLIP` | `CUDAExecutionProvider` and `llama/onnx_bitflip.so` |
| FP32 `RANDOM_BITFLIP` | `onnxruntime_extensions` custom op library |

`cnn_inference.py` registers the needed custom op library when the selected
precision and fault model require it.

---

## CSV columns

| Column | Meaning |
|--------|---------|
| `Timestamp` | Row write time |
| `Layer_Config` | JSON config filename |
| `Layer_Type` | `Conv`, `FC`, or `MatMul` when present in the config |
| `Fault_Model` | Fault model used for the row |
| `Bit_Position` | Bit position used for bit-flip models |
| `Sample_Idx` | Dataset sample index |
| `Rand_Idx` | Flat target tensor index supplied as `rand_idx_inject` |
| `Golden_Pred` | Prediction from the original model |
| `Faulty_Pred` | Prediction from the injected model |
| `Label` | Dataset label |
| `Prediction_Changed` | Whether `Golden_Pred != Faulty_Pred` |
| `Golden_Correct` | Whether `Golden_Pred == Label` |
| `Faulty_Correct` | Whether `Faulty_Pred == Label` |
| `Linf_Diff` | `max(abs(golden_out - faulty_out))` after casting to FP32 |
