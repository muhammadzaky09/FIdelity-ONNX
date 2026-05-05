# Test layout

The pytest suite is split by validation scope:

- `unit/`: small CPU-safe tests for injection subgraph builders, mask geometry,
  precision-specific bit math, and structural custom-op nodes.
- `integration/`: CPU-safe tests that run `modify_onnx_graph`, parser wiring,
  CNN inference runtime setup, all fault models, and custom-op library
  registration.
- `cuda/`: opt-in runtime tests for CUDAExecutionProvider and
  `llama/onnx_bitflip.so`.
- `legacy/`: older manual validation scripts kept for reference. These are not
  collected by pytest.

Default CPU-safe validation:

```bash
source ~/miniconda3/bin/activate fidelity-onnx
python -m pytest
```

CUDA validation:

```bash
source ~/miniconda3/bin/activate fidelity-onnx
python -m pytest tests/cuda --run-cuda
```

Focused validation:

```bash
python -m pytest tests/integration/test_fault_model_matrix.py
python -m pytest -m custom_ops
```
