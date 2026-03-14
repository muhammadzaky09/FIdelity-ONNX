"""
Compare activations between clean and RANDOM_BITFLIP-injected decoder.

Both models run on the same dummy inputs. Differences in hidden_out,
past_key, and past_value reveal how far the single bit-flip propagates.

Run:
    conda run -n fidelity-onnx python tests/test_decoder_injected.py
"""
import sys, os, struct
import numpy as np
import onnxruntime as ort

REPO         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CLEAN  = os.path.join(REPO, "decoders", "7B16", "decoder-merge-8.onnx")
MODEL_INJECT = os.path.join(REPO, "decoders", "7B16", "decoder-merge-8_injected.onnx")
SO_BITFLIP   = os.path.join(REPO, "llama", "onnx_bitflip.so")
SO_PERTURB   = os.path.join(REPO, "llama", "onnx_perturb.so")

RAND_IDX = 42   # flat index of the element to bit-flip in v_proj/MatMul output

# ── register custom op libraries ─────────────────────────────────────────────
opts = ort.SessionOptions()
opts.log_severity_level = 3
for so in [SO_BITFLIP, SO_PERTURB]:
    if os.path.exists(so):
        opts.register_custom_ops_library(so)
        print(f"[OK] Registered {os.path.basename(so)}")
    else:
        print(f"[WARN] Not found: {so}")

# ── create sessions ───────────────────────────────────────────────────────────
EPS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
try:
    sess_clean  = ort.InferenceSession(MODEL_CLEAN,  opts, providers=EPS)
    sess_inject = ort.InferenceSession(MODEL_INJECT, opts, providers=EPS)
except Exception as e:
    print(f"SKIP: could not create session: {e}")
    sys.exit(0)

ep = sess_clean.get_providers()[0]
print(f"[Session] EP: {ep}")
if "CUDA" not in ep:
    print("SKIP: CUDA EP not active — BitFlip requires GPU (CUDA 12 + cuDNN 9).")
    sys.exit(0)

# ── shared dummy inputs ───────────────────────────────────────────────────────
np.random.seed(0)
SEQ = 1; PAST = 0; HEADS = 32; HEAD_DIM = 128; HIDDEN = 4096

base_feed = {
    "hidden_in":     np.random.randn(1, SEQ, HIDDEN).astype(np.float16),
    "attn_mask":     np.zeros((1, 1, SEQ, PAST + SEQ), dtype=np.float16),
    "position_ids":  np.arange(SEQ, dtype=np.int64).reshape(1, SEQ),
    "past_key_in":   np.zeros((1, HEADS, PAST, HEAD_DIM), dtype=np.float16),
    "past_value_in": np.zeros((1, HEADS, PAST, HEAD_DIM), dtype=np.float16),
}

feed_clean  = dict(base_feed)
feed_inject = dict(base_feed, rand_idx_inject=np.array(RAND_IDX, dtype=np.int64))

# ── run both ──────────────────────────────────────────────────────────────────
out_clean  = sess_clean.run(None, feed_clean)
out_inject = sess_inject.run(None, feed_inject)

out_names = [o.name for o in sess_clean.get_outputs()]  # hidden_out, past_key, past_value

# ── compare ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Fault: RANDOM_BITFLIP  |  rand_idx={RAND_IDX}  |  target: v_proj/MatMul")
print(f"{'='*60}")

total_changed = 0
for name, clean, inject in zip(out_names, out_clean, out_inject):
    flat_c = clean.flatten()
    flat_i = inject.flatten()
    diff_mask  = flat_c != flat_i
    n_changed  = diff_mask.sum()
    total_changed += n_changed

    abs_err = np.abs(flat_c.astype(np.float32) - flat_i.astype(np.float32))
    max_err = abs_err[diff_mask].max() if n_changed > 0 else 0.0
    mean_err = abs_err[diff_mask].mean() if n_changed > 0 else 0.0

    print(f"\n[{name}]  shape={clean.shape}")
    print(f"  Changed elements : {n_changed} / {flat_c.size}  "
          f"({100.0 * n_changed / flat_c.size:.4f}%)")
    print(f"  Max  |Δ|         : {max_err:.6f}")
    print(f"  Mean |Δ|         : {mean_err:.6f}")

    if n_changed > 0 and n_changed <= 10:
        # show all changed positions
        for idx in np.where(diff_mask)[0]:
            c_val = flat_c[idx];  i_val = flat_i[idx]
            c_bits = struct.unpack('<H', struct.pack('<e', c_val))[0]
            i_bits = struct.unpack('<H', struct.pack('<e', i_val))[0]
            print(f"    [{idx:6d}]  clean={c_val:+.5f} ({c_bits:016b})  "
                  f"faulty={i_val:+.5f} ({i_bits:016b})  "
                  f"XOR={c_bits^i_bits:016b}")
    elif n_changed > 10:
        # show top-5 largest differences
        top5 = np.argsort(abs_err)[::-1][:5]
        top5 = top5[diff_mask[top5]]   # keep only changed ones
        print(f"  Top-5 largest Δ:")
        for idx in top5:
            c_val = float(flat_c[idx]); i_val = float(flat_i[idx])
            print(f"    [{idx:6d}]  clean={c_val:+.5f}  faulty={i_val:+.5f}  Δ={i_val-c_val:+.5f}")

print(f"\n{'='*60}")
print(f"  Total changed across all outputs: {total_changed}")
print(f"{'='*60}")
print("\nPASS")
