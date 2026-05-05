from pathlib import Path

import onnxruntime as ort
import pytest


@pytest.mark.custom_ops
def test_onnx_bitflip_shared_library_registers():
    repo_root = Path(__file__).resolve().parents[2]
    library_path = repo_root / "llama" / "onnx_bitflip.so"

    assert library_path.exists(), f"Missing custom op library: {library_path}"

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(str(library_path))
