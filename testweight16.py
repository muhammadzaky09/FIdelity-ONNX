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

def create_mask_for_matmul_output(matmul_output="y", masked_output="y_masked",
                                  block_length=4):
    nodes = []
    suffix = "_mask"

    # 1. Get the shape of the MatMul output 'y'
    # Expected shape: [B, seq_length, hidden_size]
    nodes.append(helper.make_node(
        "Shape",
        inputs=[matmul_output],
        outputs=["y_shape_mask"]
    ))

    # --- Extract seq_length (the dimension to mask) and hidden_size ---
    # For seq_length, we want the second element of y_shape_mask.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["M_starts"],
        value=helper.make_tensor(
            name="M_starts_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]  # index 1 for seq_length
        )
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["M_ends"],
        value=helper.make_tensor(
            name="M_ends_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[2]  # slice out element at index 1 only
        )
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["M_axes"],
        value=helper.make_tensor(
            name="M_axes_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]  # always slice along axis 0 of the 1D shape tensor
        )
    ))

    # For hidden_size, get the third element (index 2) of y_shape_mask.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["N_starts"],
        value=helper.make_tensor(
            name="N_starts_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[2]  # index 2 for hidden_size
        )
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["N_ends"],
        value=helper.make_tensor(
            name="N_ends_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[3]  # slice out element at index 2 only
        )
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["N_axes"],
        value=helper.make_tensor(
            name="N_axes_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))

    # 2. Slice out seq_length from y_shape_mask.
    nodes.append(helper.make_node(
        "Slice",
        inputs=["y_shape_mask", "M_starts", "M_ends", "M_axes"],
        outputs=["M_value"]
    ))
    # Squeeze to convert from shape [1] to a scalar.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["squeeze_axes"],
        value=helper.make_tensor(
            name="squeeze_axes_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))
    nodes.append(helper.make_node(
        "Squeeze",
        inputs=["M_value", "squeeze_axes"],
        outputs=["M_scalar"]
    ))

    # 3. Slice out hidden_size.
    nodes.append(helper.make_node(
        "Slice",
        inputs=["y_shape_mask", "N_starts", "N_ends", "N_axes"],
        outputs=["N_value"]
    ))

    # 4. Create constants 0 and 1 for use with Range.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero_const_mask"],
        value=helper.make_tensor(
            name="zero_tensor_mask",
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[0]
        )
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["one_const_mask"],
        value=helper.make_tensor(
            name="one_tensor_mask",
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[1]
        )
    ))

    # 5. Generate a vector of row indices: Range(0, M_scalar, 1)
    nodes.append(helper.make_node(
        "Range",
        inputs=["zero_const_mask", "M_scalar", "one_const_mask"],
        outputs=["row_indices"]
    ))

    # 6. --- Compute dynamic (random) start index ---
    # Create constant for block_length.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["block_length_const"],
        value=helper.make_tensor(
            name="block_length_tensor",
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[block_length]
        )
    ))
    # Compute: M_minus_block = M_scalar - block_length_const.
    nodes.append(helper.make_node(
        "Sub",
        inputs=["M_scalar", "block_length_const"],
        outputs=["M_minus_block"]
    ))
    # Compute range_size = M_minus_block + 1.
    nodes.append(helper.make_node(
        "Add",
        inputs=["M_minus_block", "one_const_mask"],
        outputs=["range_size"]
    ))
    # Cast range_size to FLOAT.
    nodes.append(helper.make_node(
        "Cast",
        inputs=["range_size"],
        outputs=["range_size_float"],
        to=TensorProto.FLOAT
    ))
    # Generate a random FLOAT value in [0,1) with shape [1].
    nodes.append(helper.make_node(
        "RandomUniform",
        inputs=[],
        outputs=["rand_val_temp"],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0,
        shape=[1]
    ))
    # Squeeze the random value to obtain a scalar.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["squeeze_axes_rand"],
        value=helper.make_tensor(
            name="squeeze_axes_rand_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))
    nodes.append(helper.make_node(
        "Squeeze",
        inputs=["rand_val_temp", "squeeze_axes_rand"],
        outputs=["rand_val"]
    ))
    # Scale the random value: rand_scaled = rand_val * range_size_float.
    nodes.append(helper.make_node(
        "Mul",
        inputs=["rand_val", "range_size_float"],
        outputs=["rand_scaled"]
    ))
    # Floor the scaled value.
    nodes.append(helper.make_node(
        "Floor",
        inputs=["rand_scaled"],
        outputs=["rand_index_float"]
    ))
    # Cast the floored value to INT64 to get the dynamic start index.
    nodes.append(helper.make_node(
        "Cast",
        inputs=["rand_index_float"],
        outputs=["start_index_dynamic"],
        to=TensorProto.INT64
    ))
    # Compute dynamic end index: end_index_dynamic = start_index_dynamic + block_length_const.
    nodes.append(helper.make_node(
        "Add",
        inputs=["start_index_dynamic", "block_length_const"],
        outputs=["end_index_dynamic"]
    ))

    # 7. --- Build the mask using the dynamic start index ---
    # Compare: row_indices >= start_index_dynamic.
    nodes.append(helper.make_node(
        "GreaterOrEqual",
        inputs=["row_indices", "start_index_dynamic"],
        outputs=["ge_mask"]
    ))
    # Compare: row_indices < end_index_dynamic.
    nodes.append(helper.make_node(
        "Less",
        inputs=["row_indices", "end_index_dynamic"],
        outputs=["lt_mask"]
    ))
    # Combine the two Boolean masks using And.
    nodes.append(helper.make_node(
        "And",
        inputs=["ge_mask", "lt_mask"],
        outputs=["mask_bool"]
    ))
    # Cast Boolean mask to FLOAT16.
    nodes.append(helper.make_node(
        "Cast",
        inputs=["mask_bool"],
        outputs=["mask_float"],
        to=TensorProto.FLOAT16
    ))
    # 8. Unsqueeze mask_float to shape [seq_length, 1] (provide axes as an input).
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_axes_mask"],
        value=helper.make_tensor(
            name="unsqueeze_axes_mask_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]
        )
    ))
    nodes.append(helper.make_node(
        "Unsqueeze",
        inputs=["mask_float", "unsqueeze_axes_mask"],
        outputs=["mask_unsqueezed"]
    ))
    # 9. Create a constant '1' for tiling as a 1D tensor (shape [1]).
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["one_for_tile"],
        value=helper.make_tensor(
            name="one_for_tile_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]
        )
    ))
    # 10. Concatenate [1] with N_value to form tile multiples: [1, hidden_size].
    nodes.append(helper.make_node(
        "Concat",
        inputs=["one_for_tile", "N_value"],
        outputs=["tile_multiples"],
        axis=0
    ))
    # 11. Tile mask_unsqueezed to shape [seq_length, hidden_size].
    nodes.append(helper.make_node(
        "Tile",
        inputs=["mask_unsqueezed", "tile_multiples"],
        outputs=["mask_full"]
    ))
    # 12. Multiply the MatMul output 'y' with the mask.
    # (mask_full has shape [seq_length, hidden_size] and will broadcast to [B, seq_length, hidden_size])
    nodes.append(helper.make_node(
        "Mul",
        inputs=[matmul_output, "mask_full"],
        outputs=[masked_output]
    ))

    return nodes




# === Building the complete graph (fault injection + MatMul + mask) ===

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
    mask_nodes = create_mask_for_matmul_output(matmul_output="y", masked_output="y_masked",
                                               block_length=block_length)

    # Combine all nodes: weight constant, fault injection subgraph, MatMul, then mask.
    all_nodes = [weight_node] + fault_nodes + [matmul_node] + mask_nodes

    # Graph inputs: "x" is provided externally.
    input_tensor_info = helper.make_tensor_value_info("x", TensorProto.FLOAT16, list(input_shape))
    # Graph outputs: now use "y_masked" as final output.
    output_tensor_info = helper.make_tensor_value_info("y_masked", TensorProto.FLOAT16,
                                                       [input_shape[0], weight_shape[1]])

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