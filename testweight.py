import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto
import csv
import os

# Import your fault injection function. Adjust the import as needed.
from inject_ops import create_quantized_fault_injection_weight

def build_matmul_with_faulty_weight(input_shape=(3, 16), weight_shape=(16, 16), input_dtype=np.float16):
    """
    Builds an ONNX model that:
      1. Provides a constant weight tensor (of shape weight_shape) in FP16.
      2. Passes this weight through the fault injection subgraph
         (create_quantized_fault_injection_weight) to produce a faulty weight.
      3. Multiplies a dummy input (named "x") with the faulty weight using MatMul.
         (For a 2D input x of shape [1, 40] and weight of shape [40,40],
          the output y will have shape [1,40].)
    """
    # Step 1: Create a constant weight tensor.
    weight_value = np.random.rand(*weight_shape).astype(np.float16)
    weight_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["weight_input"],
        value=helper.make_tensor(
            name="weight_input_tensor",
            data_type=TensorProto.FLOAT16,
            dims=weight_shape,
            vals=weight_value.flatten().tolist()
        )
    )
    
    # Step 2: Apply your fault injection to the weight.
    # This function should take the weight constant (by name) and output a perturbed weight.
    fault_nodes = create_quantized_fault_injection_weight("weight_input", "faulty_weight", bit_position=3)
    
    # Step 3: Create a MatMul node that uses an external input "x" and the faulty weight.
    # Here we assume "x" has shape [1, 40] so that MatMul(x, faulty_weight) gives an output of shape [1,40].
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["x", "faulty_weight"],
        outputs=["y"],
        name="MatMul_with_faulty_weight"
    )
    
    # Build the graph.
    # Graph inputs: "x" is provided externally.
    input_tensor_info = helper.make_tensor_value_info("x", TensorProto.FLOAT16, list(input_shape))
    # Graph outputs: "y"
    output_tensor_info = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [input_shape[0], input_shape[1], weight_shape[1]])

    
    # Combine nodes: first, the weight constant, then the fault injection subgraph, then the MatMul.
    all_nodes = [weight_node] + fault_nodes + [matmul_node]
    
    # Build the graph.
    graph = helper.make_graph(
        nodes=all_nodes,
        name="MatMulFaultInjectionGraph",
        inputs=[input_tensor_info],
        outputs=[output_tensor_info]
    )
    
    model = helper.make_model(graph, producer_name="MatMulFaultInjectionTest")
    onnx.checker.check_model(model)
    return model

def run_model(model):
    # Run the model using ONNX Runtime.
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    # Create a dummy input "x" of shape [1,40].
    x = np.random.randn(1, 10, 16).astype(np.float16)
    y = sess.run(None, {"x": x})[0]
    return x, y

if __name__ == "__main__":
    model = build_matmul_with_faulty_weight(input_shape=(1, 10, 16), weight_shape=(16, 16), input_dtype=np.float16)
    x, y = run_model(model)
    print("Input x:")
    print(x)
    print("\nOutput y (result of MatMul with faulty weight):")
    print(y)
    
    csv_filename = "input_shape_and_nonzero_indices.csv"
    file_exists = os.path.exists(csv_filename)
    fault_model = "WEIGHT"

    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header only if file didn't exist.
        if not file_exists:
            writer.writerow(["Input Shape", "Nonzero Indices"])
        # Write a blank row as separator.
        writer.writerow([])
        writer.writerow([str(fault_model), ""])
        writer.writerow([str(x.shape), ""])
        writer.writerow(["", "Nonzero Index (tuple)"])
        for idx in np.argwhere(y != 0):
            writer.writerow(["", tuple(idx.tolist())])
