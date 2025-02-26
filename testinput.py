import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto
import csv
import os

# Import your INPUT fault injection function.
# It is assumed to compute x_faulty = x - delta and output it as the given output_name.
from inject_ops import create_quantized_fault_injection

def build_input_matmul_model(input_shape=(1, 5, 8), weight_shape=(8, 8), input_dtype=np.float16, bit_position=3):
    """
    Builds an ONNX model that:
      1. Accepts an external input "x" (a 3D tensor).
      2. Applies your fault injection subgraph (which subtracts delta from x) so that
         the fault-injected input is produced as "x_faulty".
      3. Multiplies the fault-injected input with a dummy weight (a constant) via MatMul.
         (The MatMul here follows standard ONNX broadcasting rules.)
    """
    # Define tensor names.
    input_name = "x"
    faulty_input_name = "x_faulty"  # output of your injection subgraph
    weight_name = "W"
    matmul_output = "y"
    
    # 1. External input info.
    input_tensor_info = helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, list(input_shape))
    
    # 2. Fault injection subgraph.
    # This function computes the perturbation and subtracts it from x to produce x_faulty.
    injection_nodes = create_quantized_fault_injection(input_name, faulty_input_name, bit_position)
    
    # 3. Dummy weight constant.
    weight_value = np.random.rand(*weight_shape).astype(input_dtype)
    weight_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[weight_name],
        value=helper.make_tensor(
            name="weight_tensor",
            data_type=TensorProto.FLOAT16,
            dims=weight_shape,
            vals=weight_value.flatten().tolist()
        )
    )
    
    # 4. MatMul node.
    # The faulty input "x_faulty" has shape [1,5,8] and weight W has shape [8,8],
    # so the MatMul output "y" will have shape [1,5,8] (via broadcasting rules).
    matmul_node = helper.make_node(
        "MatMul",
        inputs=[faulty_input_name, weight_name],
        outputs=[matmul_output],
        name="MatMul_fault_injected"
    )
    
    # 5. Build the graph.
    all_nodes = injection_nodes + [weight_node, matmul_node]
    output_tensor_info = helper.make_tensor_value_info(matmul_output, TensorProto.FLOAT16, [input_shape[0], input_shape[1], weight_shape[1]])
    
    graph = helper.make_graph(
        nodes=all_nodes,
        name="InputFaultInjectionMatMulGraph",
        inputs=[input_tensor_info],
        outputs=[output_tensor_info]
    )
    
    model = helper.make_model(graph, producer_name="FaultInjectionInputTest")
    onnx.checker.check_model(model)
    return model

def run_model(model):
    # Create a dummy input with the given shape.
    x = np.random.randn(1, 5, 8).astype(np.float16)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    y = sess.run(None, { "x": x })[0]
    return x, y

if __name__ == "__main__":
    model = build_input_matmul_model(input_shape=(1, 5, 8), weight_shape=(8, 8), input_dtype=np.float16, bit_position=3)
    x, y = run_model(model)
    
    print("Input x:")
    print(x)
    print("\nOutput y (result of MatMul with fault-injected input):")
    print(y)
    
    # Get the input shape.
    input_shape = x.shape  # e.g., (1, 5, 8)
    
    # Find the indices where y is nonzero.
    nonzero_indices = np.argwhere(y != 0)
    
    csv_filename = "input_shape_and_nonzero_indices.csv"
    file_exists = os.path.exists(csv_filename)
    fault_model = "INPUT"

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
