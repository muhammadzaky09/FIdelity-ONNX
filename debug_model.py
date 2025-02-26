import numpy as np
import onnx
import onnxruntime as ort
from onnx import utils

# Load and prepare model
model = onnx.load('try/decoder-merge-20-patched_injected.onnx')
model.opset_import[0].version = 18
model = utils.polish_model(model)
onnx.checker.check_model(model)
model_bytes = model.SerializeToString()


providers = ['CPUExecutionProvider']
session = ort.InferenceSession(model_bytes, providers=providers)

N = 10           
lastN = 5        
totalN = N + lastN  

attn_mask = np.zeros((1, 1, N, totalN), dtype=np.float16)
for i in range(N):
    attn_mask[0, 0, i, :lastN+i+1] = 1.0

position_ids = np.arange(lastN, lastN + N, dtype=np.int64).reshape(1, N)

inputs = {
    'hidden_in': np.random.rand(1, N, 4096).astype(np.float16),
    'attn_mask': attn_mask,
    'position_ids': position_ids,
    'past_key_in': np.random.rand(1, 32, lastN, 128).astype(np.float16),
    'past_value_in': np.random.rand(1, 32, lastN, 128).astype(np.float16)
}

try:
    outputs = session.run(None, inputs)
    print(outputs)
    hidden_out, past_key, past_value = outputs
    
    print(f"hidden_out shape: {hidden_out.shape}")
    print(f"past_key shape: {past_key.shape}")
    print(f"past_value shape: {past_value.shape}")
except Exception as e:
    print(f"Error during inference: {e}")
    
    # Additional debugging info
    print("\nInput shapes:")
    for name, value in inputs.items():
        print(f"{name}: {value.shape}")