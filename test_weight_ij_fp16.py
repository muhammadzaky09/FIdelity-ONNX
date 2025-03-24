import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnxruntime as ort
from inject_ops import create_fp16_fault_injection_weight

def test_weight_injection():
    # Create a simple FP16 weight tensor (2D for simplicity).
    weight_value = np.array([[1.0, 2.0],
                             [3.0, 4.0]], dtype=np.float16)
    
    # Create an initializer from the weight.
    weight_initializer = numpy_helper.from_array(weight_value, name="weight_fp16")
    
    # Create a Constant node that holds the weight.
    weight_node = helper.make_node(
        'Constant',
        inputs=[],  # No inputs for a constant.
        outputs=["weight_fp16"],
        value=weight_initializer
    )
    
    # Use the injection function to generate the fault injection subgraph.
    # Here, we flip the least-significant bit (bit_position = 0).
    # The output of the injection subgraph is named "perturbation_fp16".
    injection_nodes = create_fp16_fault_injection_weight(
        input_name="weight_fp16",
        output_name="fault_perturbation_fp16",
        bit_position=0
    )
    
    # Build the graph:
    # - The graph contains the weight constant node and then the injection subgraph.
    # - The final output is the perturbation tensor.
    graph = helper.make_graph(
        nodes=[weight_node] + injection_nodes,
        name="weight_injection_graph",
        inputs=[],  # No external inputs (the weight is constant).
        outputs=[helper.make_tensor_value_info("fault_perturbation_fp16", TensorProto.FLOAT16, weight_value.shape)]
    )
    
    # Create the model.
    # Note: We add an opset import for the custom domain "custom.bitflip" (version 1).
    model = helper.make_model(
        graph,
        producer_name="TestWeightInjection",
        opset_imports=[
            helper.make_opsetid("", 14),
            helper.make_opsetid("custom.bitflip", 1)
        ]
    )
    
    # Save the model.
    onnx.save(model, "test_weight_injection.onnx")
    print("Saved model as 'test_weight_injection.onnx'.")
    
    # Create an ONNX Runtime session.
    # Since the model uses the custom op "BitFlip" in domain "custom.bitflip",
    # you must ensure that the shared library (e.g. onnx_bitflip.so) is registered.
    # If you are integrating the custom op into your system, you should have already
    # loaded the library. Here, we assume that the shared library is available at a given path.
    custom_op_lib_path = "llama/onnx_bitflip.so"  # <-- Update with your actual path.
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(custom_op_lib_path)
    sess = ort.InferenceSession("test_weight_injection.onnx", sess_options, providers=['CPUExecutionProvider'])
    
    # Run inference. No external inputs are needed.
    outputs = sess.run(None, {})
    
    print("Original weight:")
    print(weight_value)
    print("Perturbation tensor (fault injection):")
    print(outputs[0])
    
if __name__ == "__main__":
    test_weight_injection()
