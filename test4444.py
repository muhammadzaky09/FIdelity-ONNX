import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto
from inject_ops import create_quantized_fault_injection, create_iw16_fault_injection

def build_and_run_onnx_model(nodes, input_shape, input_dtype):
    input_name = "input_tensor"
    output_name = "output_tensor"

    # Build an ONNX graph with a single input and single output.
    graph = helper.make_graph(
        nodes=nodes,
        name="InjectionTestGraph",
        inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, list(input_shape))],
        outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, list(input_shape))]
    )
    model = helper.make_model(graph, producer_name="InjectionTest")
    onnx.checker.check_model(model)
    
    # Run inference using ONNX Runtime.
    sess = ort.InferenceSession(model.SerializeToString())
    input_data = np.random.randn(*input_shape).astype(input_dtype)
    output = sess.run(None, {input_name: input_data})[0]
    return output

# Test INPUT injection (using create_quantized_fault_injection)
print("Testing INPUT injection:")
# For INPUT injection, the function assumes a 3D tensor (e.g. [batch, sequence, hidden])
input_injection_nodes = create_quantized_fault_injection("input_tensor", "output_tensor", bit_position=3)
output_input = build_and_run_onnx_model(input_injection_nodes, input_shape=(1, 512, 4096), input_dtype=np.float16)
nonzero_indices_input = np.argwhere(output_input != 0)
nonzero_values_input = output_input[output_input != 0]
print("INPUT Nonzero Indices:\n", nonzero_indices_input)
print("INPUT Nonzero Values:\n", nonzero_values_input)

# Test INPUT16 injection (using create_iw16_fault_injection)
print("\nTesting INPUT16 injection:")
# For INPUT16 injection, the function is designed for a 3D tensor as well (e.g. [batch, sequence, hidden])
# but it injects the fault into the activation in a slightly different manner.
input16_injection_nodes = create_iw16_fault_injection("input_tensor", "output_tensor", bit_position=3, block_size=16, target_axis_offset=1)
output_input16 = build_and_run_onnx_model(input16_injection_nodes, input_shape=(1, 512, 4096), input_dtype=np.float16)
nonzero_indices_input16 = np.argwhere(output_input16 != 0)
nonzero_values_input16 = output_input16[output_input16 != 0]
print("INPUT16 Nonzero Indices:\n", nonzero_indices_input16)
print("INPUT16 Nonzero Values:\n", nonzero_values_input16)
