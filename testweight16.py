import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto
import csv
import os

# Assume create_quantized_fault_injection_weight is defined elsewhere.
# It takes an input tensor name (e.g. "orig_weight"), an output tensor name (e.g. "perturbation"),
# and a bit_position, and produces a perturbation tensor (of the same shape as the weight).
from inject_ops import create_quantized_fault_injection_weight


def create_weight16_mask(matmul_output="y", masked_output="y_masked", block_length=4):
    """
    Create a mask that keeps only 'block_length' consecutive rows in the sequence dimension.
    Fixed to properly handle broadcasting with 3D tensors of shape [batch, sequence, hidden].
    """
    nodes = []
    suffix = "_mask"
    
    # 1. Get the shape of the input tensor
    nodes.append(helper.make_node(
        "Shape",
        inputs=[matmul_output],
        outputs=["y_shape" + suffix]
    ))
    
    # 2. Get the sequence length (dimension 1)
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["dim1_idx" + suffix],
        value=helper.make_tensor(
            name="dim1_idx_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]  # Second dimension (index 1)
        )
    ))
    
    nodes.append(helper.make_node(
        "Gather",
        inputs=["y_shape" + suffix, "dim1_idx" + suffix],
        outputs=["seq_len_tensor" + suffix],
        axis=0
    ))
    
    # 3. Squeeze to scalar
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["squeeze_axes" + suffix],
        value=helper.make_tensor(
            name="squeeze_axes_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))
    
    nodes.append(helper.make_node(
        "Squeeze",
        inputs=["seq_len_tensor" + suffix, "squeeze_axes" + suffix],
        outputs=["seq_len" + suffix]
    ))
    
    # 4. Create scalar constants
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero_scalar" + suffix],
        value=helper.make_tensor(
            name="zero_scalar_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],  # Empty dims = scalar
            vals=[0]
        )
    ))
    
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["one_scalar" + suffix],
        value=helper.make_tensor(
            name="one_scalar_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],  # Empty dims = scalar
            vals=[1]
        )
    ))
    
    # 5. Create a range of indices (0 to seq_len-1)
    nodes.append(helper.make_node(
        "Range",
        inputs=["zero_scalar" + suffix, "seq_len" + suffix, "one_scalar" + suffix],
        outputs=["seq_indices" + suffix]
    ))
    
    # 6. Create block length constant
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["block_len" + suffix],
        value=helper.make_tensor(
            name="block_len_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[block_length]
        )
    ))
    
    # 7. Calculate valid block length (min of block_length and seq_len)
    nodes.append(helper.make_node(
        "Min",
        inputs=["block_len" + suffix, "seq_len" + suffix],
        outputs=["valid_block" + suffix]
    ))
    
    # 8. Calculate max start index
    nodes.append(helper.make_node(
        "Sub",
        inputs=["seq_len" + suffix, "valid_block" + suffix],
        outputs=["max_start" + suffix]
    ))
    
    # 9. Generate random start index
    nodes.append(helper.make_node(
        "Cast",
        inputs=["max_start" + suffix],
        outputs=["max_start_float" + suffix],
        to=TensorProto.FLOAT
    ))
    
    nodes.append(helper.make_node(
        "RandomUniform",
        inputs=[],
        outputs=["rand_tensor" + suffix],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0,
        shape=[1]
    ))
    
    nodes.append(helper.make_node(
        "Squeeze",
        inputs=["rand_tensor" + suffix, "squeeze_axes" + suffix],
        outputs=["rand_scalar" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Mul",
        inputs=["rand_scalar" + suffix, "max_start_float" + suffix],
        outputs=["rand_scaled" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Floor",
        inputs=["rand_scaled" + suffix],
        outputs=["rand_floor" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Cast",
        inputs=["rand_floor" + suffix],
        outputs=["start_idx" + suffix],
        to=TensorProto.INT64
    ))
    
    # 10. Calculate end index
    nodes.append(helper.make_node(
        "Add",
        inputs=["start_idx" + suffix, "valid_block" + suffix],
        outputs=["end_idx" + suffix]
    ))
    
    # 11. Create boolean mask
    nodes.append(helper.make_node(
        "GreaterOrEqual",
        inputs=["seq_indices" + suffix, "start_idx" + suffix],
        outputs=["ge_mask" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Less",
        inputs=["seq_indices" + suffix, "end_idx" + suffix],
        outputs=["lt_mask" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "And",
        inputs=["ge_mask" + suffix, "lt_mask" + suffix],
        outputs=["bool_mask_1d" + suffix]
    ))
    
    # 12. Create shape for reshaping the mask to 3D
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["reshape_shape" + suffix],
        value=helper.make_tensor(
            name="reshape_shape_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[3],
            vals=[1, -1, 1]  # [1, seq_len, 1] with second dim inferred
        )
    ))
    
    # 13. Reshape the boolean mask to 3D for proper broadcasting
    nodes.append(helper.make_node(
        "Reshape",
        inputs=["bool_mask_1d" + suffix, "reshape_shape" + suffix],
        outputs=["bool_mask_3d" + suffix]
    ))
    
    # 14. Create zeros tensor for masked values
    nodes.append(helper.make_node(
        "ConstantOfShape",
        inputs=["y_shape" + suffix],
        outputs=["zeros" + suffix],
        value=helper.make_tensor(
            name="zeros_value" + suffix,
            data_type=TensorProto.FLOAT16,
            dims=[1],
            vals=[0.0]
        )
    ))
    
    # 15. Use Where instead of Mul for proper broadcasting
    # Where(condition, x, y) returns x where condition is true, y otherwise
    nodes.append(helper.make_node(
        "Where",
        inputs=["bool_mask_3d" + suffix, matmul_output, "zeros" + suffix],
        outputs=[masked_output]
    ))
    
    return nodes

def build_matmul_with_faulty_weight(input_shape=(2, 16), weight_shape=(16, 16), input_dtype=np.float16,
                                    block_length=4, mask_start=0):
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

    # Step 2: Apply the fault injection to the weight.
    fault_nodes = create_quantized_fault_injection_weight("weight_input", "faulty_weight", bit_position=3)

    # Step 3: Create a MatMul node that uses an external input "x" and the faulty weight.
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["x", "faulty_weight"],
        outputs=["y"],
        name="MatMul_with_faulty_weight"
    )

    # Step 4: Create the mask subgraph that restricts the nonzero outputs.
    mask_nodes = create_weight16_mask(matmul_output="y", masked_output="y_masked",
                                               block_length=block_length)

    # Combine all nodes: weight constant, fault injection subgraph, MatMul, then mask.
    all_nodes = [weight_node] + fault_nodes + [matmul_node] + mask_nodes

    # Graph inputs: "x" is provided externally.
    input_tensor_info = helper.make_tensor_value_info("x", TensorProto.FLOAT16, list(input_shape))
    # Graph outputs: now use "y_masked" as final output.
    output_tensor_info = helper.make_tensor_value_info("y_masked", TensorProto.FLOAT16,
                                                 [input_shape[0], input_shape[1], weight_shape[1]])

    graph = helper.make_graph(
        nodes=all_nodes,
        name="MatMulFaultInjectionGraph",
        inputs=[input_tensor_info],
        outputs=[output_tensor_info]
    )

    model = helper.make_model(graph, producer_name="MatMulFaultInjectionTest")
    onnx.checker.check_model(model)
    model.opset_import[0].version = 18
    return model

def run_model(model):
    # Run the model using ONNX Runtime.
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    # Create a dummy input "x" with the defined shape.
    x = np.random.randn(1, 10, 16).astype(np.float16)
    y = sess.run(None, {"x": x})[0]
    return x, y

if __name__ == "__main__":
    size=2
    # For example, with a block length of 4 in each column.
    model = build_matmul_with_faulty_weight(input_shape=(1, 10, 16), weight_shape=(16, 16),
                                            input_dtype=np.float16, block_length=size)
    x, y_masked = run_model(model)
    print("Input x:")
    print(x)
    print("\nOutput y_masked (result of MatMul with faulty weight and mask applied):")
    print(y_masked)
    
    csv_filename = "input16_weight16.csv"
    file_exists = os.path.exists(csv_filename)
    fault_model = "WEIGHT16"

    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header only if file didn't exist.
        if not file_exists:
            writer.writerow(["Input Shape", "Block Length","Nonzero Indices"])
        # Write a blank row as separator.
        writer.writerow([])
        writer.writerow([str(fault_model), ""])
        writer.writerow([str(x.shape), ""])
        writer.writerow([str(size), ""])
        writer.writerow(["", "Nonzero Index (tuple)"])
        for idx in np.argwhere(y_masked != 0):
            writer.writerow(["", tuple(idx.tolist())])