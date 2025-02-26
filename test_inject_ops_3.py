import onnx
from onnx import helper, TensorProto, save_model
import numpy as np
import onnxruntime as ort

def create_WEIGHT16_fault_model(input_name, output_name, bit_position=0, block_size=16):
    nodes = []
    
    # --- Common Constant Nodes ---
    # Unsqueeze axes for various operations.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_axes_0"],
        value=helper.make_tensor("unsqueeze_axes_0_value", TensorProto.INT64, [1], [0])
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_axes_1"],
        value=helper.make_tensor("unsqueeze_axes_1_value", TensorProto.INT64, [1], [1])
    ))
    # New constant for unsqueezing along the last axis (axis=-1)
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_axes_neg1"],
        value=helper.make_tensor("unsqueeze_axes_neg1_value", TensorProto.INT64, [1], [-1])
    ))
    
    # --- Part A: Bit-flip Perturbation (same as INPUT16) ---
    nodes.append(helper.make_node("Shape", [input_name], ["runtime_shape"]))
    nodes.append(helper.make_node("Cast", ["runtime_shape"], ["runtime_shape_float"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("RandomUniform", inputs=[], outputs=["random_vals"],
                                  dtype=TensorProto.FLOAT, high=1.0, low=0.0, shape=[3]))
    nodes.append(helper.make_node("Mul", ["random_vals", "runtime_shape_float"], ["scaled_indices"]))
    nodes.append(helper.make_node("Floor", ["scaled_indices"], ["floored_indices"]))
    nodes.append(helper.make_node("Cast", ["floored_indices"], ["indices_int64"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Cast", [input_name], ["int8_val"], to=TensorProto.INT8))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["bitmask"],
                                  value=helper.make_tensor("bitmask_val", TensorProto.INT8, [], [1 << bit_position])))
    nodes.append(helper.make_node("ConstantOfShape", ["runtime_shape"], ["zero_base"],
                                  value=helper.make_tensor("zero_value_int8", TensorProto.INT8, [1], [0])))
    nodes.append(helper.make_node("ScatterND", ["zero_base", "indices_int64", "bitmask"], ["bit_mask"]))
    nodes.append(helper.make_node("BitwiseXor", ["int8_val", "bit_mask"], ["flipped_int"]))
    nodes.append(helper.make_node("Cast", ["flipped_int"], ["flipped_int32"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Cast", ["int8_val"], ["int8_val32"], to=TensorProto.INT32))
    nodes.append(helper.make_node("Sub", ["flipped_int32", "int8_val32"], ["perturbation_int32"]))
    nodes.append(helper.make_node("Cast", ["perturbation_int32"], ["delta_temp"], to=TensorProto.FLOAT))
    
    # --- Part B: Block Restriction for WEIGHT16 (target axis = rank-2) ---
    nodes.append(helper.make_node("Shape", ["delta_temp"], ["delta_shape"], name="Shape_delta_temp"))
    nodes.append(helper.make_node("Size", ["delta_shape"], ["delta_rank"], name="Size_delta_shape"))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["two_int64"],
                                  value=helper.make_tensor("two_int64", TensorProto.INT64, [1], [2])))
    nodes.append(helper.make_node("Sub", ["delta_rank", "two_int64"], ["index_target"], name="Compute_index_target"))
    nodes.append(helper.make_node("Gather", ["delta_shape", "index_target"], ["target_dim"], name="Gather_target_dim"))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["block_size_tensor"],
                                  value=helper.make_tensor("block_size_tensor", TensorProto.INT64, [1], [block_size])))
    nodes.append(helper.make_node("Div", ["target_dim", "block_size_tensor"], ["num_blocks"], name="Compute_num_blocks"))
    nodes.append(helper.make_node("Cast", ["num_blocks"], ["num_blocks_float"], name="Cast_num_blocks", to=TensorProto.FLOAT))
    nodes.append(helper.make_node("RandomUniform", inputs=[], outputs=["ru_block"],
                                  name="RandomUniform_block", low=0.0, high=1.0, shape=[1]))
    nodes.append(helper.make_node("Mul", ["ru_block", "num_blocks_float"], ["scaled_block"], name="Mul_block"))
    nodes.append(helper.make_node("Floor", ["scaled_block"], ["floor_block"], name="Floor_block"))
    nodes.append(helper.make_node("Cast", ["floor_block"], ["rand_block_index"], name="Cast_rand_block", to=TensorProto.INT64))
    nodes.append(helper.make_node("Mul", ["rand_block_index", "block_size_tensor"], ["start_index_block"], name="Compute_start_index_block"))
    
    nodes.append(helper.make_node("NonZero", ["delta_temp"], ["nonzero_indices"], name="NonZero_delta_temp"))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["gather_index"],
                                  value=helper.make_tensor("gather_index", TensorProto.INT64, [1], [0])))
    nodes.append(helper.make_node("Gather", ["nonzero_indices", "gather_index"], ["base_coord"],
                                  name="Gather_base_coord", axis=1))
    nodes.append(helper.make_node("Squeeze", ["base_coord", "unsqueeze_axes_1"], ["base_coord_squeezed"],
                                  name="Squeeze_base_coord"))
    # --- Split coordinate vector into prefix and suffix ---
    # For prefix: slice from index 0 to -2 along axis 0.
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["prefix_starts"],
                                  value=helper.make_tensor("prefix_starts", TensorProto.INT64, [1], [0])))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["prefix_ends"],
                                  value=helper.make_tensor("prefix_ends", TensorProto.INT64, [1], [-2])))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["prefix_axes"],
                                  value=helper.make_tensor("prefix_axes", TensorProto.INT64, [1], [0])))
    nodes.append(helper.make_node("Slice", ["base_coord_squeezed", "prefix_starts", "prefix_ends", "prefix_axes"],
                                  ["prefix"], name="Slice_prefix"))
    # For suffix: slice from index -1 to a large positive value.
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["suffix_starts"],
                                  value=helper.make_tensor("suffix_starts", TensorProto.INT64, [1], [-1])))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["suffix_ends"],
                                  value=helper.make_tensor("suffix_ends", TensorProto.INT64, [1], [2147483647])))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["suffix_axes"],
                                  value=helper.make_tensor("suffix_axes", TensorProto.INT64, [1], [0])))
    nodes.append(helper.make_node("Slice", ["base_coord_squeezed", "suffix_starts", "suffix_ends", "suffix_axes"],
                                  ["suffix"], name="Slice_suffix"))
    # Concatenate prefix, start_index_block, and suffix to form new_base.
    nodes.append(helper.make_node("Concat", ["prefix", "start_index_block", "suffix"],
                                  ["new_base"],
                                  name="Concat_new_base",
                                  axis=0))
    # Extract the replaced coordinate (now at position -2) from new_base.
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["base_replaced_starts"],
                                  value=helper.make_tensor("base_replaced_starts", TensorProto.INT64, [1], [-2])))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["base_replaced_ends"],
                                  value=helper.make_tensor("base_replaced_ends", TensorProto.INT64, [1], [-1])))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["base_replaced_axes"],
                                  value=helper.make_tensor("base_replaced_axes", TensorProto.INT64, [1], [0])))
    nodes.append(helper.make_node("Slice", ["new_base", "base_replaced_starts", "base_replaced_ends", "base_replaced_axes"],
                                  ["base_replaced"],
                                  name="Slice_base_replaced"))
    # Create offsets vector: [0, 1, ..., block_size-1]
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["offsets"],
                                  value=helper.make_tensor("offsets", TensorProto.INT64, [block_size], list(range(block_size)))))
    nodes.append(helper.make_node("Add", ["base_replaced", "offsets"], ["block_indices"], name="Add_offsets"))
    
    # --- Assemble final indices ---
    # Tile prefix and suffix so that for each offset we form a complete coordinate.
    nodes.append(helper.make_node("Unsqueeze", ["prefix", "unsqueeze_axes_0"], ["prefix_unsqueezed"], name="Unsqueeze_prefix"))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["tile_multipliers_prefix"],
                                  value=helper.make_tensor("tile_multipliers_prefix", TensorProto.INT64, [2], [block_size, 1])))
    nodes.append(helper.make_node("Tile", ["prefix_unsqueezed", "tile_multipliers_prefix"], ["prefix_tiled"], name="Tile_prefix"))
    nodes.append(helper.make_node("Unsqueeze", ["suffix", "unsqueeze_axes_0"], ["suffix_unsqueezed"], name="Unsqueeze_suffix"))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["tile_multipliers_suffix"],
                                  value=helper.make_tensor("tile_multipliers_suffix", TensorProto.INT64, [2], [block_size, 1])))
    nodes.append(helper.make_node("Tile", ["suffix_unsqueezed", "tile_multipliers_suffix"], ["suffix_tiled"], name="Tile_suffix"))
    # Unsqueeze block_indices using the new constant (axis=-1) so that shape becomes [16,1].
    nodes.append(helper.make_node("Unsqueeze", ["block_indices", "unsqueeze_axes_neg1"], ["block_indices_unsqueezed"], name="Unsqueeze_block_indices"))
    # Concatenate along axis=1 to form final_indices. Expected shape: [16, 3].
    nodes.append(helper.make_node("Concat", ["prefix_tiled", "block_indices_unsqueezed", "suffix_tiled"],
                                  ["final_indices"],
                                  name="Concat_final_indices",
                                  axis=1))
    
    # Create a zero tensor (FLOAT) of the same shape as delta_temp.
    nodes.append(helper.make_node("ConstantOfShape", ["delta_shape"], ["zero_tensor"], name="ConstantOfShape_zero",
                                  value=helper.make_tensor("zero_value_float", TensorProto.FLOAT, [1], [0.0])))
    
    # Extract the perturbation value from delta_temp using ReduceSum.
    nodes.append(helper.make_node("ReduceSum", ["delta_temp"], ["perturb_value"], keepdims=0, name="ReduceSum_delta"))
    nodes.append(helper.make_node("Unsqueeze", ["perturb_value", "unsqueeze_axes_0"], ["perturb_value_unsqueezed"], name="Unsqueeze_perturb_value"))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["tile_multiplier_for_value"],
                                  value=helper.make_tensor("tile_multiplier_for_value", TensorProto.INT64, [1], [block_size])))
    nodes.append(helper.make_node("Tile", ["perturb_value_unsqueezed", "tile_multiplier_for_value"], ["replicated_value"], name="Tile_perturb_value"))
    
    nodes.append(helper.make_node("ScatterND", ["zero_tensor", "final_indices", "replicated_value"],
                                  [output_name],
                                  name="ScatterND_final"))
    
    return nodes

def build_WEIGHT16_model(input_name, output_name, bit_position=0, block_size=16):
    nodes = create_WEIGHT16_fault_model(input_name, output_name, bit_position, block_size)
    # Define input and output value infos.
    # Here we assume a 3D weight tensor (for example, shape [1, 32, 16]).
    input_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, None, None])
    output_vi = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, None, None])
    graph = helper.make_graph(nodes, "WEIGHT16_Perturbation_Graph", [input_vi], [output_vi])
    opset = [helper.make_opsetid("", 18)]
    model = helper.make_model(graph, producer_name="WEIGHT16_Perturbation_Model", opset_imports=opset)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model

if __name__ == "__main__":
    # Build and save the model.
    block_size=
    model = build_WEIGHT16_model("weight_tensor", "delta_out", bit_position=1, block_size=3)
    save_model(model, "WEIGHT16_fault_model.onnx")
    print("Saved WEIGHT16 fault ONNX model as WEIGHT16_fault_model.onnx")
    
    # Test with ONNX Runtime.
    session = ort.InferenceSession("WEIGHT16_fault_model.onnx")
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    print("Input name:", in_name)
    print("Output name:", out_name)
    
    # Create a dummy weight tensor.
    dummy_weight = np.random.randn(1, 32, 16).astype(np.float32)
    outputs = session.run([out_name], {in_name: dummy_weight})
    delta_out = outputs[0]
    print("Delta output:")
    print(delta_out)
    nonzero_indices = np.argwhere(delta_out != 0)
    print("Nonzero indices in the delta tensor:")
    print(nonzero_indices)
    nonzero_values = delta_out[delta_out != 0]
    print("Nonzero values:")
    print(nonzero_values)
