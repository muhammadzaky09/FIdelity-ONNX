import onnx
from onnx import helper, TensorProto, save_model
import numpy as np
import csv
import onnxruntime as ort

def create_quantized_fault_injection(input_name, output_name, bit_position):
    nodes = []
    
    # -------------------------------
    # 1. Index Generation (Simplified)
    # -------------------------------
    
    # Get input shape, e.g. if input is [batch, sequence, hidden] then runtime_shape is [3]
    nodes.append(helper.make_node(
        'Shape',
        inputs=[input_name],
        outputs=['runtime_shape']
    ))
    
    # Cast the whole runtime_shape to FLOAT directly.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['runtime_shape'],
        outputs=['runtime_shape_float'],
        to=TensorProto.FLOAT
    ))
    
    # Generate random uniform values directly with shape [3].
    # (This assumes that the runtime shape has 3 elements.)
    nodes.append(helper.make_node(
        'RandomUniform',
        inputs=[],
        outputs=['random_vals'],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0,
        shape=[3]
    ))
    
    # Multiply the random values with runtime_shape_float.
    # Since both tensors are [3], no reshaping is required.
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals', 'runtime_shape_float'],
        outputs=['scaled_indices']
    ))
    
    # Floor the results.
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_indices'],
        outputs=['floored_indices']
    ))
    
    # Cast to INT64 to obtain indices.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_indices'],
        outputs=['indices_int64'],
        to=TensorProto.INT64
    ))
    
    # -------------------------------
    # 2. Fault Injection Operations
    # -------------------------------
    
    # Cast the original input tensor to INT8.
    nodes.append(helper.make_node(
        'Cast',
        inputs=[input_name],
        outputs=['int8_val'],
        to=TensorProto.INT8
    ))
    
    # Create a constant for the bitmask.
    nodes.append(helper.make_node(
    'Constant',
    inputs=[],
    outputs=['bitmask'],
    value=helper.make_tensor(
        name='bitmask_val',
        data_type=TensorProto.INT8,
        dims=[],  # Change from [1] to [] to produce a scalar
        vals=[1 << bit_position]
        )
    ))

    # Create a zero tensor of the same shape as the input using ConstantOfShape.
    nodes.append(helper.make_node(
        'ConstantOfShape',
        inputs=['runtime_shape'],  # reusing runtime_shape ensures the shape is correct
        outputs=['zero_base'],
        value=helper.make_tensor(
            name='zero_value',
            data_type=TensorProto.INT8,
            dims=[1],
            vals=[0]
        )
    ))
    
    # Scatter the bitmask into the zero tensor at positions given by indices_int64.
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=['zero_base', 'indices_int64', 'bitmask'],
        outputs=['bit_mask']
    ))
    
    # Perform the bit flip using BitwiseXor.
    nodes.append(helper.make_node(
        'BitwiseXor',
        inputs=['int8_val', 'bit_mask'],
        outputs=['flipped_int']
    ))
    
    # Cast both the flipped tensor and the original INT8 tensor to INT32.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['flipped_int'],
        outputs=['flipped_int32'],
        to=TensorProto.INT32
    ))
    
    nodes.append(helper.make_node(
        'Cast',
        inputs=['int8_val'],
        outputs=['int8_val32'],
        to=TensorProto.INT32
    ))
    
    # Subtract the original value from the flipped value (INT32 subtraction).
    nodes.append(helper.make_node(
        'Sub',
        inputs=['flipped_int32', 'int8_val32'],
        outputs=['perturbation_int32']
    ))
    
    # Finally, cast the perturbation to FLOAT.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['perturbation_int32'],
        outputs=[output_name],
        to=TensorProto.FLOAT
    ))
    return nodes

def create_iw16_fault_model(input_name, output_name, bit_position=0, block_size=16, target_axis_offset=1):
    nodes = []
    
    # --- Common Constant Nodes ---
    # For unsqueeze operations.
    nodes.append(helper.make_node(
        "Constant", inputs=[], outputs=["unsqueeze_axes_0"],
        value=helper.make_tensor("unsqueeze_axes_0_value", TensorProto.INT64, [1], [0])
    ))
    nodes.append(helper.make_node(
        "Constant", inputs=[], outputs=["unsqueeze_axes_1"],
        value=helper.make_tensor("unsqueeze_axes_1_value", TensorProto.INT64, [1], [1])
    ))
    # For unsqueezing along the last axis (used in some branches).
    nodes.append(helper.make_node(
        "Constant", inputs=[], outputs=["unsqueeze_axes_neg1"],
        value=helper.make_tensor("unsqueeze_axes_neg1_value", TensorProto.INT64, [1], [-1])
    ))
    
    nodes.extend(create_quantized_fault_injection(input_name, "delta_temp", bit_position))
    
    # --- Part B: Block Restriction and Replication ---
    # Compute dynamic shape and rank of delta_temp.
    nodes.append(helper.make_node("Shape", ["delta_temp"], ["delta_shape"], name="Shape_delta_temp"))
    nodes.append(helper.make_node("Size", ["delta_shape"], ["delta_rank"], name="Size_delta_shape"))
    
    # Compute target axis: target_axis = delta_rank - target_axis_offset.
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["target_axis_offset_const"],
                                  value=helper.make_tensor("target_axis_offset_const", TensorProto.INT64, [1], [target_axis_offset])))
    nodes.append(helper.make_node("Sub", ["delta_rank", "target_axis_offset_const"], ["target_axis"], name="Compute_target_axis"))
    
    # Gather the size of the target dimension.
    nodes.append(helper.make_node("Gather", ["delta_shape", "target_axis"], ["target_dim"], name="Gather_target_dim"))
    
    # Create a constant for block_size.
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["block_size_tensor"],
                                  value=helper.make_tensor("block_size_tensor", TensorProto.INT64, [1], [block_size])))
    # Compute the number of blocks = target_dim // block_size.
    nodes.append(helper.make_node("Div", ["target_dim", "block_size_tensor"], ["num_blocks"], name="Compute_num_blocks"))
    nodes.append(helper.make_node("Cast", ["num_blocks"], ["num_blocks_float"], name="Cast_num_blocks", to=TensorProto.FLOAT))
    nodes.append(helper.make_node("RandomUniform", inputs=[], outputs=["ru_block"],
                                  name="RandomUniform_block", low=0.0, high=1.0, shape=[1]))
    nodes.append(helper.make_node("Mul", ["ru_block", "num_blocks_float"], ["scaled_block"], name="Mul_block"))
    nodes.append(helper.make_node("Floor", ["scaled_block"], ["floor_block"], name="Floor_block"))
    nodes.append(helper.make_node("Cast", ["floor_block"], ["rand_block_index"], name="Cast_rand_block", to=TensorProto.INT64))
    nodes.append(helper.make_node("Mul", ["rand_block_index", "block_size_tensor"], ["start_index_block"], name="Compute_start_index_block"))
    
    # Obtain nonzero coordinate.
    nodes.append(helper.make_node("NonZero", ["delta_temp"], ["nonzero_indices"], name="NonZero_delta_temp"))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["gather_index"],
                                  value=helper.make_tensor("gather_index", TensorProto.INT64, [1], [0])))
    nodes.append(helper.make_node("Gather", ["nonzero_indices", "gather_index"], ["base_coord"],
                                  name="Gather_base_coord", axis=1))
    nodes.append(helper.make_node("Squeeze", ["base_coord", "unsqueeze_axes_1"], ["base_coord_squeezed"],
                                  name="Squeeze_base_coord"))
    
    # Branch based on target_axis_offset.
    if target_axis_offset == 1:
        # ----- INPUT16: target axis is last axis -----
        # Remove the last coordinate.
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["prefix_starts"],
                                  value=helper.make_tensor("prefix_starts", TensorProto.INT64, [1], [0])))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["prefix_ends"],
                                  value=helper.make_tensor("prefix_ends", TensorProto.INT64, [1], [-1])))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["prefix_axes"],
                                  value=helper.make_tensor("prefix_axes", TensorProto.INT64, [1], [0])))
        nodes.append(helper.make_node("Slice", ["base_coord_squeezed", "prefix_starts", "prefix_ends", "prefix_axes"],
                                  ["base_prefix"], name="Slice_base_prefix"))
        # New base coordinate: concat base_prefix and start_index_block.
        nodes.append(helper.make_node("Concat", ["base_prefix", "start_index_block"],
                                  ["new_base"], name="Concat_new_base", axis=0))
        # Extract replaced coordinate: last element of new_base.
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["rep_starts"],
                                  value=helper.make_tensor("rep_starts", TensorProto.INT64, [1], [-1])))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["rep_ends"],
                                  value=helper.make_tensor("rep_ends", TensorProto.INT64, [1], [2147483647])))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["rep_axes"],
                                  value=helper.make_tensor("rep_axes", TensorProto.INT64, [1], [0])))
        nodes.append(helper.make_node("Slice", ["new_base", "rep_starts", "rep_ends", "rep_axes"],
                                  ["replaced_coord"], name="Slice_replaced_coord"))
    else:
        # ----- WEIGHT16: target axis is second-to-last axis -----
        # For a 3D tensor, target_axis = 3 - 2 = 1.
        # So, prefix = slice from index 0 to 1; suffix = slice from index 2 to end.
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["prefix_starts"],
                                  value=helper.make_tensor("prefix_starts", TensorProto.INT64, [1], [0])))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["prefix_ends"],
                                  value=helper.make_tensor("prefix_ends", TensorProto.INT64, [1], [1])))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["prefix_axes"],
                                  value=helper.make_tensor("prefix_axes", TensorProto.INT64, [1], [0])))
        nodes.append(helper.make_node("Slice", ["base_coord_squeezed", "prefix_starts", "prefix_ends", "prefix_axes"],
                                  ["prefix"], name="Slice_prefix"))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["suffix_starts"],
                                  value=helper.make_tensor("suffix_starts", TensorProto.INT64, [1], [2])))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["suffix_ends"],
                                  value=helper.make_tensor("suffix_ends", TensorProto.INT64, [1], [2147483647])))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["suffix_axes"],
                                  value=helper.make_tensor("suffix_axes", TensorProto.INT64, [1], [0])))
        nodes.append(helper.make_node("Slice", ["base_coord_squeezed", "suffix_starts", "suffix_ends", "suffix_axes"],
                                  ["suffix"], name="Slice_suffix"))
        nodes.append(helper.make_node("Concat", ["prefix", "start_index_block", "suffix"],
                                  ["new_base"], name="Concat_new_base", axis=0))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["rep_starts"],
                                  value=helper.make_tensor("rep_starts", TensorProto.INT64, [1], [1])))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["rep_ends"],
                                  value=helper.make_tensor("rep_ends", TensorProto.INT64, [1], [2])))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["rep_axes"],
                                  value=helper.make_tensor("rep_axes", TensorProto.INT64, [1], [0])))
        nodes.append(helper.make_node("Slice", ["new_base", "rep_starts", "rep_ends", "rep_axes"],
                                  ["replaced_coord"], name="Slice_replaced_coord"))
    
    # Compute block_indices = replaced_coord + offsets.
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["offsets"],
                                  value=helper.make_tensor("offsets", TensorProto.INT64, [block_size], list(range(block_size)))))
    nodes.append(helper.make_node("Add", ["replaced_coord", "offsets"], ["block_indices"], name="Add_offsets"))
    
    # --- Assemble final indices ---
    if target_axis_offset == 1:
        # INPUT16 mode: only tile the base_prefix.
        nodes.append(helper.make_node("Unsqueeze", ["base_prefix", "unsqueeze_axes_0"],
                                  ["base_prefix_unsqueezed"], name="Unsqueeze_prefix"))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["tile_multipliers"],
                                  value=helper.make_tensor("tile_multipliers", TensorProto.INT64, [2], [block_size, 1])))
        nodes.append(helper.make_node("Tile", ["base_prefix_unsqueezed", "tile_multipliers"],
                                  ["base_prefix_tiled"], name="Tile_prefix"))
        # Unsqueeze block_indices along axis 1 (to turn shape [block_size] into [block_size, 1]).
        nodes.append(helper.make_node("Unsqueeze", ["block_indices", "unsqueeze_axes_1"],
                                  ["block_indices_unsqueezed"], name="Unsqueeze_block_indices"))
        nodes.append(helper.make_node("Concat", ["base_prefix_tiled", "block_indices_unsqueezed"],
                                  ["final_indices"], name="Concat_final_indices", axis=1))
    else:
        # WEIGHT16 mode: tile both prefix and suffix.
        nodes.append(helper.make_node("Unsqueeze", ["prefix", "unsqueeze_axes_0"],
                                  ["prefix_unsqueezed"], name="Unsqueeze_prefix"))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["tile_multipliers_prefix"],
                                  value=helper.make_tensor("tile_multipliers_prefix", TensorProto.INT64, [2], [block_size, 1])))
        nodes.append(helper.make_node("Tile", ["prefix_unsqueezed", "tile_multipliers_prefix"],
                                  ["prefix_tiled"], name="Tile_prefix"))
        nodes.append(helper.make_node("Unsqueeze", ["suffix", "unsqueeze_axes_0"],
                                  ["suffix_unsqueezed"], name="Unsqueeze_suffix"))
        nodes.append(helper.make_node("Constant", inputs=[], outputs=["tile_multipliers_suffix"],
                                  value=helper.make_tensor("tile_multipliers_suffix", TensorProto.INT64, [2], [block_size, 1])))
        nodes.append(helper.make_node("Tile", ["suffix_unsqueezed", "tile_multipliers_suffix"],
                                  ["suffix_tiled"], name="Tile_suffix"))
        # Unsqueeze block_indices along axis -1 so that its shape becomes [block_size,1].
        nodes.append(helper.make_node("Unsqueeze", ["block_indices", "unsqueeze_axes_neg1"],
                                  ["block_indices_unsqueezed"], name="Unsqueeze_block_indices"))
        nodes.append(helper.make_node("Concat", ["prefix_tiled", "block_indices_unsqueezed", "suffix_tiled"],
                                  ["final_indices"], name="Concat_final_indices", axis=1))
    
    # --- Final common steps ---
    nodes.append(helper.make_node("ConstantOfShape", ["delta_shape"], ["zero_tensor"], name="ConstantOfShape_zero",
                                  value=helper.make_tensor("zero_value_float", TensorProto.FLOAT, [1], [0.0])))
    nodes.append(helper.make_node("ReduceSum", ["delta_temp"], ["perturb_value"], keepdims=0, name="ReduceSum_delta"))
    nodes.append(helper.make_node("Unsqueeze", ["perturb_value", "unsqueeze_axes_0"],
                                  ["perturb_value_unsqueezed"], name="Unsqueeze_perturb_value"))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["tile_multiplier_for_value"],
                                  value=helper.make_tensor("tile_multiplier_for_value", TensorProto.INT64, [1], [block_size])))
    nodes.append(helper.make_node("Tile", ["perturb_value_unsqueezed", "tile_multiplier_for_value"],
                                  ["replicated_value"], name="Tile_perturb_value"))
    nodes.append(helper.make_node("ScatterND", ["zero_tensor", "final_indices", "replicated_value"],
                                  [output_name], name="ScatterND_final"))
    
    return nodes

def build_fault_model(input_name, output_name, bit_position=0, block_size=16, target_axis_offset=1):
    nodes = create_iw16_fault_model(input_name, output_name, bit_position, block_size, target_axis_offset)
    # Assume a 3D tensor for demonstration.
    input_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, None, None])
    output_vi = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, None, None])
    graph = helper.make_graph(nodes, "FaultModel_Graph", [input_vi], [output_vi])
    opset = [helper.make_opsetid("", 18)]
    model = helper.make_model(graph, producer_name="FaultModel", opset_imports=opset)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model

# Parameters for testing.
# We'll test for both WEIGHT16 and INPUT16 modes.
target_axis_offsets = [2, 1]  # 2 for WEIGHT16, 1 for INPUT16
bit_positions = range(0, 8)    # Testing bit positions 0 to 7
block_sizes = range(1, 16)     # Testing block sizes from 1 up to 15

# Dummy input: adjust these dimensions as needed.
# For example, assume a 3D tensor with shape [1, 32, 16]



# We'll save results to a CSV file.
csv_filename = "fault_injection_results.csv"

with open(csv_filename, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write header row.
    csvwriter.writerow(["target_axis_offset", "mode", "bit_position", "block_size", "nonzero_indices", "nonzero_values"])
    
    # Loop over both modes.
    for target_axis_offset in target_axis_offsets:
        mode = "INPUT16" if target_axis_offset == 1 else "WEIGHT16"
        # Loop over bit positions.
        for bit_position in bit_positions:
            # Loop over block sizes.
            for block_size in block_sizes:
                if mode == "INPUT16":
                    block_size = 16
                # Build model for these parameters.
                model = build_fault_model("tensor_in", "delta_out", bit_position, block_size, target_axis_offset)
                # Save model (if desired) to a temporary file.
                temp_model_path = "CombinedFaultModel.onnx"
                save_model(model, temp_model_path)
                
                # Create an ONNX Runtime session.
                session = ort.InferenceSession(temp_model_path)
                in_name = session.get_inputs()[0].name
                out_name = session.get_outputs()[0].name
                dummy_input = np.random.randn(1, 512, 4096).astype(np.float32)
                # Run inference.
                outputs = session.run([out_name], {in_name: dummy_input})
                delta_out = outputs[0]
                
                # Find nonzero indices and values.
                nonzero_indices = np.argwhere(delta_out != 0)
                nonzero_values = delta_out[delta_out != 0]
                
                # For CSV writing, convert arrays to string.
                indices_str = np.array2string(nonzero_indices, separator=',')
                values_str = np.array2string(nonzero_values, separator=',')
                
                # Write a row to CSV.
                csvwriter.writerow([target_axis_offset, mode, bit_position, block_size, indices_str, values_str])
                print(f"Mode: {mode}, bit: {bit_position}, block: {block_size}")
                print("Nonzero indices:", indices_str)
                print("Nonzero values:", values_str)

print("Results saved to", csv_filename)