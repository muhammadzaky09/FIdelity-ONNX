import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto
import csv
import os

# Import your INPUT fault injection function.
# It is assumed to compute: x_faulty = x - delta, and output it as the given output name.
from inject_ops import create_quantized_fault_injection

###############################################################################
# Masking Subgraph for MatMul Output (applied AFTER MatMul)
###############################################################################
###############################################################################
# Masking Subgraph for MatMul Output (applied AFTER MatMul)
###############################################################################
def create_input16_mask(matmul_output="y", masked_output="y_masked", block_length=16):

    nodes = []
    # 1. Get the shape of the MatMul output.
    nodes.append(helper.make_node("Shape", inputs=[matmul_output], outputs=["y_shape"]))
    
    # 2. Extract hidden dimension H from y_shape (assume y_shape = [B,S,H]).
    const_H_start = helper.make_tensor("H_start", TensorProto.INT64, [1], [2])
    const_H_end   = helper.make_tensor("H_end",   TensorProto.INT64, [1], [3])
    const_H_axes  = helper.make_tensor("H_axes",  TensorProto.INT64, [1], [0])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_starts"], value=const_H_start))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_ends"],   value=const_H_end))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_axes"],   value=const_H_axes))
    nodes.append(helper.make_node("Slice", inputs=["y_shape", "H_starts", "H_ends", "H_axes"], outputs=["H_value"]))
    # Squeeze to get scalar H. (Omit axes so all 1-dim are removed.)
    nodes.append(helper.make_node("Squeeze", inputs=["H_value"], outputs=["H_scalar"]))
    
    # 3. Compute dynamic start index along H.
    const_block = helper.make_tensor("block_length", TensorProto.INT64, [], [block_length])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["block_length_const"], value=const_block))
    nodes.append(helper.make_node("Sub", inputs=["H_scalar", "block_length_const"], outputs=["H_minus_block"]))
    const_one = helper.make_tensor("one_const", TensorProto.INT64, [], [1])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_const"], value=const_one))
    nodes.append(helper.make_node("Add", inputs=["H_minus_block", "one_const"], outputs=["range_size"]))
    nodes.append(helper.make_node("Cast", inputs=["range_size"], outputs=["range_size_float"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("RandomUniform", inputs=[], outputs=["rand_val_temp"], dtype=TensorProto.FLOAT, high=1.0, low=0.0, shape=[1]))
    nodes.append(helper.make_node("Squeeze", inputs=["rand_val_temp"], outputs=["rand_val"]))
    nodes.append(helper.make_node("Mul", inputs=["rand_val", "range_size_float"], outputs=["rand_scaled"]))
    nodes.append(helper.make_node("Floor", inputs=["rand_scaled"], outputs=["rand_index_float"]))
    nodes.append(helper.make_node("Cast", inputs=["rand_index_float"], outputs=["start_index_dynamic"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Add", inputs=["start_index_dynamic", "block_length_const"], outputs=["end_index_dynamic"]))
    
    # 4. Build 1D mask over H.
    const_zero = helper.make_tensor("zero_const", TensorProto.INT64, [], [0])
    const_one_step = helper.make_tensor("one_step", TensorProto.INT64, [], [1])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["zero_const_H"], value=const_zero))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_const_H_step"], value=const_one_step))
    nodes.append(helper.make_node("Range", inputs=["zero_const_H", "H_scalar", "one_const_H_step"], outputs=["indices_H"]))
    nodes.append(helper.make_node("GreaterOrEqual", inputs=["indices_H", "start_index_dynamic"], outputs=["ge_mask_H"]))
    nodes.append(helper.make_node("Less", inputs=["indices_H", "end_index_dynamic"], outputs=["lt_mask_H"]))
    nodes.append(helper.make_node("And", inputs=["ge_mask_H", "lt_mask_H"], outputs=["mask_bool_H"]))
    nodes.append(helper.make_node("Cast", inputs=["mask_bool_H"], outputs=["mask_1d"], to=TensorProto.FLOAT16))
    
    # 5. Unsqueeze mask_1d to shape [1,1,H]. Use a constant axes tensor [0,1].
    const_unsqueeze_axes = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [2], [0,1])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["unsqueeze_axes"], value=const_unsqueeze_axes))
    nodes.append(helper.make_node("Unsqueeze", inputs=["mask_1d", "unsqueeze_axes"], outputs=["mask_unsqueezed"]))
    
    # 6. Tile mask_unsqueezed to shape [B,S,H]. Extract B and S from y_shape.
    const_B_start = helper.make_tensor("B_start", TensorProto.INT64, [1], [0])
    const_B_end   = helper.make_tensor("B_end",   TensorProto.INT64, [1], [1])
    const_B_axes  = helper.make_tensor("B_axes",  TensorProto.INT64, [1], [0])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["B_starts_out"], value=const_B_start))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["B_ends_out"], value=const_B_end))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["B_axes_out"], value=const_B_axes))
    nodes.append(helper.make_node("Slice", inputs=["y_shape", "B_starts_out", "B_ends_out", "B_axes_out"], outputs=["B_value_out"]))
    nodes.append(helper.make_node("Squeeze", inputs=["B_value_out"], outputs=["B_scalar_out"]))
    
    const_S_start = helper.make_tensor("S_start", TensorProto.INT64, [1], [1])
    const_S_end   = helper.make_tensor("S_end",   TensorProto.INT64, [1], [2])
    const_S_axes  = helper.make_tensor("S_axes",  TensorProto.INT64, [1], [0])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["S_starts_out"], value=const_S_start))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["S_ends_out"], value=const_S_end))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["S_axes_out"], value=const_S_axes))
    nodes.append(helper.make_node("Slice", inputs=["y_shape", "S_starts_out", "S_ends_out", "S_axes_out"], outputs=["S_value_out"]))
    nodes.append(helper.make_node("Squeeze", inputs=["S_value_out"], outputs=["S_scalar_out"]))
    
    # Convert B_scalar_out and S_scalar_out (scalars) to 1D tensors.
    const_unsqueeze_axis0 = helper.make_tensor("unsqueeze_axis0", TensorProto.INT64, [1], [0])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["unsqueeze_axis0"], value=const_unsqueeze_axis0))
    nodes.append(helper.make_node("Unsqueeze", inputs=["B_scalar_out", "unsqueeze_axis0"], outputs=["B_1d"]))
    nodes.append(helper.make_node("Unsqueeze", inputs=["S_scalar_out", "unsqueeze_axis0"], outputs=["S_1d"]))
    
    const_one_tile = helper.make_tensor("one_tile", TensorProto.INT64, [1], [1])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_for_tile"], value=const_one_tile))
    nodes.append(helper.make_node("Concat", inputs=["B_1d", "S_1d", "one_for_tile"], outputs=["tile_multiples"], axis=0))
    nodes.append(helper.make_node("Tile", inputs=["mask_unsqueezed", "tile_multiples"], outputs=["mask_full"]))
    
    # 7. Multiply the MatMul output with the mask.
    nodes.append(helper.make_node("Mul", inputs=[matmul_output, "mask_full"], outputs=[masked_output]))
    
    return nodes

###############################################################################
# Build Complete Model: Injection -> MatMul -> Masking (Post-MatMul)
###############################################################################
def build_input_matmul_model_with_mask(input_shape=(1, 40, 25), weight_shape=(25, 25),
                                       input_dtype=np.float16, bit_position=3, block_length=16):
    input_name = "x"
    faulty_input_name = "x_faulty"
    matmul_output = "y"
    masked_output = "y_masked"
    weight_name = "W"
    
    input_tensor_info = helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, list(input_shape))
    
    injection_nodes = create_quantized_fault_injection(input_name, faulty_input_name, bit_position)
    
    weight_value = np.random.rand(*weight_shape).astype(input_dtype)
    weight_node = helper.make_node("Constant", inputs=[], outputs=[weight_name],
                                   value=helper.make_tensor("weight_tensor", TensorProto.FLOAT16, dims=weight_shape,
                                                            vals=weight_value.flatten().tolist()))
    matmul_node = helper.make_node("MatMul", inputs=[faulty_input_name, weight_name], outputs=[matmul_output], name="MatMul_fault_injected")
    
    mask_nodes = create_input16_mask(matmul_output, masked_output, block_length)
    
    all_nodes = injection_nodes + [weight_node, matmul_node] + mask_nodes
    
    output_tensor_info = helper.make_tensor_value_info(masked_output, TensorProto.FLOAT16,
                                                       [input_shape[0], input_shape[1], weight_shape[1]])
    
    graph = helper.make_graph(nodes=all_nodes,
                              name="InputFaultInjectionMatMulMaskGraph",
                              inputs=[input_tensor_info],
                              outputs=[output_tensor_info])
    
    model = helper.make_model(graph, producer_name="FaultInjectionInputMaskTest")
    onnx.checker.check_model(model)
    model.opset_import[0].version = 18
    return model

###############################################################################
# Testing Code
###############################################################################
def run_model(model):
    x = np.random.randn(1, 40, 25).astype(np.float16)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"x": x})[0]
    return x, y

if __name__ == "__main__":
    size=16
    model = build_input_matmul_model_with_mask(input_shape=(1, 40, 25), weight_shape=(25, 25),
                                               input_dtype=np.float16, bit_position=3, block_length=size)
    x, y = run_model(model)
    print("Input x:")
    print(x)
    print("\nOutput y_masked (result of MatMul with fault-injected then masked output):")
    print(y)
    
    csv_filename = "input16_weight16.csv"
    file_exists = os.path.exists(csv_filename)
    fault_model = "INPUT16"

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
        for idx in np.argwhere(y != 0):
            writer.writerow(["", tuple(idx.tolist())])