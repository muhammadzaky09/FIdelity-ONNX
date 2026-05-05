import struct
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest


pytestmark = [pytest.mark.cuda, pytest.mark.slow]

RAND_IDX = 42


def register_available_custom_ops(repo_root):
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3

    bitflip = repo_root / "llama" / "onnx_bitflip.so"
    if not bitflip.exists():
        pytest.skip(f"custom op library not found: {bitflip}")
    session_options.register_custom_ops_library(str(bitflip))

    perturb = repo_root / "llama" / "onnx_perturb.so"
    if perturb.exists():
        session_options.register_custom_ops_library(str(perturb))

    return session_options


def test_decoder_random_bitflip_changes_outputs_on_cuda():
    repo_root = Path(__file__).resolve().parents[2]
    clean_model = repo_root / "decoders" / "7B16" / "decoder-merge-8.onnx"
    injected_model = repo_root / "decoders" / "7B16" / "decoder-merge-8_injected.onnx"
    if not clean_model.exists():
        pytest.skip(f"clean decoder model not found: {clean_model}")
    if not injected_model.exists():
        pytest.skip(f"injected decoder model not found: {injected_model}")

    session_options = register_available_custom_ops(repo_root)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        clean_session = ort.InferenceSession(str(clean_model), session_options, providers=providers)
        injected_session = ort.InferenceSession(str(injected_model), session_options, providers=providers)
    except Exception as exc:
        pytest.skip(f"could not create decoder sessions: {exc}")

    if "CUDA" not in clean_session.get_providers()[0]:
        pytest.skip("CUDAExecutionProvider is not active")

    np.random.seed(0)
    seq = 1
    past = 0
    heads = 32
    head_dim = 128
    hidden = 4096
    base_feed = {
        "hidden_in": np.random.randn(1, seq, hidden).astype(np.float16),
        "attn_mask": np.zeros((1, 1, seq, past + seq), dtype=np.float16),
        "position_ids": np.arange(seq, dtype=np.int64).reshape(1, seq),
        "past_key_in": np.zeros((1, heads, past, head_dim), dtype=np.float16),
        "past_value_in": np.zeros((1, heads, past, head_dim), dtype=np.float16),
    }

    clean_outputs = clean_session.run(None, dict(base_feed))
    injected_outputs = injected_session.run(
        None,
        dict(base_feed, rand_idx_inject=np.array(RAND_IDX, dtype=np.int64)),
    )

    total_changed = 0
    for clean, injected in zip(clean_outputs, injected_outputs):
        assert clean.shape == injected.shape
        changed = clean.flatten() != injected.flatten()
        total_changed += int(changed.sum())

        if changed.sum() == 1:
            idx = np.where(changed)[0][0]
            clean_bits = struct.unpack("<H", struct.pack("<e", clean.flatten()[idx]))[0]
            injected_bits = struct.unpack("<H", struct.pack("<e", injected.flatten()[idx]))[0]
            assert clean_bits ^ injected_bits != 0

    assert total_changed > 0
