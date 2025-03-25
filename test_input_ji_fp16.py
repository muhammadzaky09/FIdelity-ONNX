import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import onnxruntime as ort
from inject_ops import create_fp16_fault_injection

def test_fp16_fault_injection():
    # Define a constant FP16 input tensor (3D). For example, shape [1, 2, 2].
    input_data = np.array(
    [  # Batch dimension (1)
        [  # Dimension 2 (2)
            [  # Dimension 3 (2)
                [1.0, 2.0],  # Dimension 4 (2)
                [3.0, 4.0]
            ],
            [  # Dimension 3 (2)
                [5.0, 6.0],  # Dimension 4 (2)
                [7.0, 8.0]
            ]
        ]
    ], dtype=np.float16)
    
    # Create an initializer (constant node) for the input.
    weight_initializer = numpy_helper.from_array(input_data, name="input_fp16")
    const_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=["input_fp16"],
        value=weight_initializer
    )
    
    # Build the fault injection subgraph using your FP16 fault injection function.
    # This subgraph will use your custom "BitFlip" op (domain "custom.bitflip")
    # and will inject a fault (by flipping bit 0) into one randomly selected element.
    injection_nodes = create_fp16_fault_injection(
        input_name="input_fp16",
        output_name="output_faulty_fp16",
        bit_position=2
    )
    
    # Build the overall graph.
    # The graph has no external inputs because the weight is defined as a constant.
    # The final output is the faulty weight tensor.
    graph = helper.make_graph(
        nodes=[const_node] + injection_nodes,
        name="fp16_fault_injection_graph",
        inputs=[],  # no external inputs
        outputs=[helper.make_tensor_value_info("output_faulty_fp16", TensorProto.FLOAT16, input_data.shape)]
    )
    
    # Create the model.
    # Add opset imports for the default domain and for your custom op domain.
    model = helper.make_model(
        graph,
        producer_name="fp16_fault_injection_test",
        opset_imports=[
            helper.make_opsetid("", 14),
            helper.make_opsetid("custom.bitflip", 1)
        ]
    )
    
    onnx.save(model, "test_fp16_fault_injection.onnx")
    print("Saved model as 'test_fp16_fault_injection.onnx'.")
    
    # --- Create an ONNX Runtime session with the custom op library registered ---
    # Replace "path/to/onnx_bitflip.so" with the actual path to your shared library.
    custom_op_lib_path = "llama/onnx_bitflip.so"
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(custom_op_lib_path)
    
    sess = ort.InferenceSession("test_fp16_fault_injection.onnx", sess_options, providers=['CUDAExecutionProvider'])
    
    # Run inference (no inputs needed because the model is self-contained).
    outputs = sess.run(None, {})
    
    print("Input FP16:")
    print(input_data)
    print("Faulty output (FP16):")
    print(outputs[0])

if __name__ == "__main__":
    test_fp16_fault_injection()
