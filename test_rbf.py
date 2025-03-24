import onnx
from onnx import helper, TensorProto
import numpy as np
import onnxruntime as ort
from inject_ops import create_random_bitflip_injection

def test_random_bitflip_injection():
    # Define a simple FP16 input of shape [1, 2, 2]
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [1, 2, 2])
    # We'll label the final output from the injection subgraph as "output_faulty"
    output_tensor = helper.make_tensor_value_info("output_faulty", TensorProto.FLOAT16, [1, 2, 2])
    
    nodes = []
    # Identity to pass input through (for clarity)
    nodes.append(helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["original_output"],
        name="Identity_original"
    ))
    
    # Insert the random bitflip injection subgraph.
    injection_nodes = create_random_bitflip_injection("original_output", bit_position=15)
    nodes.extend(injection_nodes)
    
    # Mark the final output explicitly.
    nodes.append(helper.make_node(
        "Identity",
        inputs=["original_output_faulty"],
        outputs=["output_faulty"],
        name="Identity_final"
    ))
    
    # Build the graph.
    graph = helper.make_graph(
        nodes,
        "test_random_bitflip_graph",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )
    
    # Create the model with opset imports (including custom op domain).
    model = helper.make_model(
        graph,
    
        opset_imports=[
            helper.make_opsetid("", 14),
            helper.make_opsetid("custom.bitflip", 1)
        ]
    )
    
    onnx.save(model, "test_random_bitflip.onnx")
    print("Saved test model as 'test_random_bitflip.onnx'.")
    
    # --- Register the custom op library when creating the InferenceSession ---
    # Specify the path to your shared library containing the custom op.
    custom_op_lib_path = "llama/onnx_bitflip.so"  # update with your actual path
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(custom_op_lib_path)
    
    # Create InferenceSession with the custom op registered.
    sess = ort.InferenceSession("test_random_bitflip.onnx", sess_options, providers=['CPUExecutionProvider'])
    
    # Create a sample FP16 input tensor.
    input_data = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float16)
    outputs = sess.run(None, {"input": input_data})
    
    print("Input:")
    print(input_data)
    print("Output (fault injected):")
    print(outputs[0])

if __name__ == "__main__":
    test_random_bitflip_injection()
