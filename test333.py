import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto
from inject_ops import create_quantized_fault_injection_weight, create_fault_injection_weight16

def build_and_run_onnx_model(nodes, input_shape=(40, 40), input_dtype=np.float16):
    input_name = "input_tensor"
    output_name = "output_tensor"

    # Generate input tensor
    input_tensor = np.random.randn(*input_shape).astype(input_dtype)

    # Define the ONNX graph
    graph = helper.make_graph(
        nodes=nodes,
        name="FaultInjectionGraph",
        inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, list(input_shape))],
        outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, list(input_shape))]
    )

    # Create model
    model = helper.make_model(graph, producer_name="FaultInjectionTest")
    onnx.checker.check_model(model)

    # Run inference using ONNX Runtime
    ort_session = ort.InferenceSession(model.SerializeToString())
    output = ort_session.run(None, {input_name: input_tensor})[0]

    # Get nonzero indices and values
    nonzero_indices = np.argwhere(output != 0)
    nonzero_values = output[output != 0]

    print("Nonzero Indices:\n", nonzero_indices)
    print("Nonzero Values:\n", nonzero_values)

# Test create_quantized_fault_injection_weight function
print("Testing create_quantized_fault_injection_weight...\n")
quant_nodes = create_quantized_fault_injection_weight("input_tensor", "output_tensor", bit_position=3)
build_and_run_onnx_model(quant_nodes)

# Test create_fault_injection_weight16 function
print("\nTesting create_fault_injection_weight16...\n")
weight16_nodes = create_fault_injection_weight16("input_tensor", "output_tensor", bit_position=7, block_size=12, target_axis_offset=2)
build_and_run_onnx_model(weight16_nodes)
