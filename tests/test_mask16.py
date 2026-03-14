"""
Simple correctness tests for create_input16_mask and create_weight16_mask.

What we verify for INPUT16:
  - Output shape is unchanged
  - Exactly min(block_length, H) consecutive features (last dim) survive
  - All other features are zero
  - The surviving window is the same across every token/batch row

What we verify for WEIGHT16:
  - Output shape is unchanged
  - Exactly min(block_length, S) consecutive token rows survive
    (S = product of all dims except the last, i.e. the "sequence" axis)
  - All other rows are entirely zero
  - The surviving rows are contiguous
"""
import sys, os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inject_ops import create_input16_mask, create_weight16_mask

PROVIDERS = ['CPUExecutionProvider']

# ──────────────────────────────────────────── helpers ────────────────────────

def build_model(nodes, shape, fp16, name):
    onnx_dtype = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    graph = helper.make_graph(
        nodes, name,
        inputs  = [helper.make_tensor_value_info('y', onnx_dtype, list(shape))],
        outputs = [helper.make_tensor_value_info('y_masked', onnx_dtype, list(shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    onnx.checker.check_model(model)
    return model

def run_model(model, x):
    sess = ort.InferenceSession(model.SerializeToString(), providers=PROVIDERS)
    return sess.run(None, {'y': x})[0]

def check_shape(out, expected):
    assert out.shape == expected, f"Shape mismatch: {out.shape} != {expected}"

def check_input16_structure(out, block_length=16):
    """
    INPUT16 keeps a contiguous window of <=block_length features (last dim).
    Every (batch, token) row must keep the same window.
    """
    H = out.shape[-1]
    expected = min(block_length, H)
    flat = out.reshape(-1, H).astype(np.float32)

    nz_per_row = [np.where(row != 0)[0] for row in flat]

    # All rows must have exactly `expected` nonzeros at the same positions
    nz0 = nz_per_row[0]
    assert len(nz0) == expected, \
        f"Expected {expected} nonzero features, got {len(nz0)}  (H={H})"
    for i, nz in enumerate(nz_per_row[1:], 1):
        assert np.array_equal(nz, nz0), \
            f"Row {i} nonzero pattern differs from row 0"

    # Contiguous check
    if len(nz0) > 1:
        assert np.all(np.diff(nz0) == 1), \
            f"Nonzero features are not contiguous: {nz0}"

def check_weight16_structure(out, block_length=16):
    """
    WEIGHT16 keeps a contiguous window of <=block_length token-rows (second-to-last
    dim, after flattening to 2-D).  All elements in those rows survive; everything
    else is zero.
    """
    H  = out.shape[-1]
    flat = out.reshape(-1, H).astype(np.float32)  # (S, H)
    S  = flat.shape[0]
    expected = min(block_length, S)

    nonzero_rows = [r for r in range(S) if np.any(flat[r] != 0)]
    assert len(nonzero_rows) == expected, \
        f"Expected {expected} nonzero rows, got {len(nonzero_rows)}  (S={S})"

    if len(nonzero_rows) > 1:
        assert nonzero_rows == list(range(nonzero_rows[0], nonzero_rows[0] + len(nonzero_rows))), \
            f"Nonzero rows not contiguous: {nonzero_rows}"

# ──────────────────────────────────────────── INPUT16 tests ──────────────────

def test_input16_fp32():
    B, S, H = 1, 4, 32
    x = np.ones((B, S, H), dtype=np.float32)
    nodes = create_input16_mask('y', 'y_masked', block_length=16, fp16=False)
    model = build_model(nodes, x.shape, fp16=False, name='inp16_fp32')
    out   = run_model(model, x)
    check_shape(out, x.shape)
    check_input16_structure(out, block_length=16)
    print(f"  [INPUT16  fp32 {list(x.shape)}]  PASS  (16 consecutive features kept)")

def test_input16_fp16():
    B, S, H = 2, 6, 40
    x = np.ones((B, S, H), dtype=np.float16)
    nodes = create_input16_mask('y', 'y_masked', block_length=16, fp16=True)
    model = build_model(nodes, x.shape, fp16=True,  name='inp16_fp16')
    out   = run_model(model, x)
    check_shape(out, x.shape)
    check_input16_structure(out, block_length=16)
    print(f"  [INPUT16  fp16 {list(x.shape)}]  PASS  (16 consecutive features kept)")

def test_input16_small_h():
    # H < block_length → all H features should survive
    B, S, H = 1, 3, 10
    x = np.ones((B, S, H), dtype=np.float32)
    nodes = create_input16_mask('y', 'y_masked', block_length=16, fp16=False)
    model = build_model(nodes, x.shape, fp16=False, name='inp16_small_h')
    out   = run_model(model, x)
    check_shape(out, x.shape)
    check_input16_structure(out, block_length=16)
    print(f"  [INPUT16  H<16 {list(x.shape)}]  PASS  (all {H} features kept, H < block_length)")

# ──────────────────────────────────────────── WEIGHT16 tests ─────────────────

def test_weight16_fp32():
    B, S, H = 1, 32, 8
    x = np.ones((B, S, H), dtype=np.float32)
    nodes = create_weight16_mask('y', 'y_masked', block_length=16, fp16=False)
    model = build_model(nodes, x.shape, fp16=False, name='w16_fp32')
    out   = run_model(model, x)
    check_shape(out, x.shape)
    check_weight16_structure(out, block_length=16)
    print(f"  [WEIGHT16 fp32 {list(x.shape)}]  PASS  (16 consecutive rows kept)")

def test_weight16_fp16():
    B, S, H = 1, 24, 64
    x = np.ones((B, S, H), dtype=np.float16)
    nodes = create_weight16_mask('y', 'y_masked', block_length=16, fp16=True)
    model = build_model(nodes, x.shape, fp16=True,  name='w16_fp16')
    out   = run_model(model, x)
    check_shape(out, x.shape)
    check_weight16_structure(out, block_length=16)
    print(f"  [WEIGHT16 fp16 {list(x.shape)}]  PASS  (16 consecutive rows kept)")

def test_weight16_small_s():
    # S < block_length → all S rows should survive
    B, S, H = 1, 10, 8
    x = np.ones((B, S, H), dtype=np.float32)
    nodes = create_weight16_mask('y', 'y_masked', block_length=16, fp16=False)
    model = build_model(nodes, x.shape, fp16=False, name='w16_small_s')
    out   = run_model(model, x)
    check_shape(out, x.shape)
    check_weight16_structure(out, block_length=16)
    print(f"  [WEIGHT16 S<16 {list(x.shape)}]  PASS  (all {S} rows kept, S < block_length)")

# ──────────────────────────────────────────── main ───────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('INPUT16 / WEIGHT16 mask tests')
    print('=' * 60)
    test_input16_fp32()
    test_input16_fp16()
    test_input16_small_h()
    print()
    test_weight16_fp32()
    test_weight16_fp16()
    test_weight16_small_s()
    print('=' * 60)
    print('All tests PASSED')
