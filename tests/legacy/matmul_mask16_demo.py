import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inject_ops import create_input16_mask, create_weight16_mask

def make_test_graph(tensor, mask_type='INPUT16'):
    """
    tensor      – numpy array (float32 or float16)
    mask_type   – 'INPUT16'  or  'WEIGHT16'
    returns     – in-memory ONNX ModelProto
    """
    fp16 = tensor.dtype == np.float16
    y_dtype = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT

    # 1. Declare graph input
    y_value_info = helper.make_tensor_value_info('y', y_dtype, tensor.shape)

    # 2. Masking nodes
    if mask_type == 'INPUT16':
        nodes = create_input16_mask(
            matmul_output='y',
            masked_output='y_masked',
            block_length=16             # any value: algorithm will min() with len(axis)
        )
    elif mask_type == 'WEIGHT16':
        nodes = create_weight16_mask(
            matmul_output='y',
            masked_output='y_masked',
            block_length=16,
            fp16=fp16                   # keep dtype consistent
        )
    else:
        raise ValueError(mask_type)

    # 3. Declare graph output
    out_value_info = helper.make_tensor_value_info('y_masked',
                                               y_dtype,
                                               list(tensor.shape))

    graph = helper.make_graph(
        nodes       = nodes,
        name        = f'test_{mask_type}',
        inputs      = [y_value_info],
        outputs     = [out_value_info],
        initializer = []
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
    onnx.checker.check_model(model)
    return model

def run(model, tensor):
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=['CPUExecutionProvider'])
    out, = sess.run(None, {'y': tensor})
    return out

# ------------------------------------------------------------------
# EXAMPLE 1 – INPUT16
# ------------------------------------------------------------------
B,S,H = 1, 5, 18          # arbitrary, >16 not needed for the demo
x = np.zeros((B,S,H), dtype=np.float16)
x[0, 2, :] = 1.0         # one full *row* of ones
print("x non-zeros:", np.count_nonzero(x))
print("x shape:", x.shape)

m_inp16 = make_test_graph(x, 'INPUT16')
out_inp16 = run(m_inp16, x)

print('INPUT16  non-zeros:', np.count_nonzero(out_inp16))
print('Should be <= 16 ; shape:', out_inp16.shape)

# ------------------------------------------------------------------
# EXAMPLE 2 – WEIGHT16
# ------------------------------------------------------------------
B,S,H = 1, 3, 50, 100
y = np.zeros((B,S,H), dtype=np.float32)
y[:, :, 1] = 2.0         # one full *column* of twos

m_w16 = make_test_graph(y, 'WEIGHT16')
out_w16 = run(m_w16, y)

print('WEIGHT16 non-zeros:', np.count_nonzero(out_w16))
print('Should be <= 16 ; shape:', out_w16.shape)