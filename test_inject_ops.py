import onnx
from onnx import helper, TensorProto, save_model
import numpy as np

def create_quantized_fault_injection(input_name, output_name, bit_position):
    """
    Part A – Perturbation (Bit-flip) Subgraph.
    
    This subgraph:
      1. Obtains the runtime shape of the input.
      2. Generates random indices (assuming a 3D input).
      3. Casts the input to INT8.
      4. Creates a constant bitmask (1 << bit_position).
      5. Creates a zero tensor (via ConstantOfShape) of the same shape.
      6. Uses ScatterND to insert the bitmask at the computed indices.
      7. Uses BitwiseXor to flip the bit.
      8. Casts the flipped and original values to INT32, subtracts them, and then casts to FLOAT.
      
    The output (named by output_name) is the FLOAT perturbation tensor (nonzero at one location).
    """
    nodes = []
    initializers = []
    
    # 1. Get the runtime shape.
    nodes.append(helper.make_node('Shape', [input_name], ['runtime_shape']))
    
    # 2. Cast the shape to FLOAT.
    nodes.append(helper.make_node('Cast', ['runtime_shape'], ['runtime_shape_float'], to=TensorProto.FLOAT))
    
    # 3. Generate random uniform values (with shape [3]).
    nodes.append(helper.make_node('RandomUniform', inputs=[], outputs=['random_vals'],
                                  dtype=TensorProto.FLOAT, high=1.0, low=0.0, shape=[3]))
    
    # 4. Multiply random values with runtime_shape_float.
    nodes.append(helper.make_node('Mul', ['random_vals', 'runtime_shape_float'], ['scaled_indices']))
    
    # 5. Floor the results.
    nodes.append(helper.make_node('Floor', ['scaled_indices'], ['floored_indices']))
    
    # 6. Cast floored indices to INT64.
    nodes.append(helper.make_node('Cast', ['floored_indices'], ['indices_int64'], to=TensorProto.INT64))
    
    # 7. Cast the input to INT8.
    nodes.append(helper.make_node('Cast', [input_name], ['int8_val'], to=TensorProto.INT8))
    
    # 8. Create a constant bitmask (1 << bit_position).
    nodes.append(helper.make_node('Constant', inputs=[], outputs=['bitmask'],
                                  value=helper.make_tensor('bitmask_val', TensorProto.INT8, [], [1 << bit_position])))
    
    # 9. Create a zero tensor of the same shape via ConstantOfShape.
    nodes.append(helper.make_node('ConstantOfShape', ['runtime_shape'], ['zero_base'],
                                  value=helper.make_tensor('zero_value', TensorProto.INT8, [1], [0])))
    
    # 10. Scatter the bitmask into zero_base at indices_int64.
    nodes.append(helper.make_node('ScatterND', ['zero_base', 'indices_int64', 'bitmask'], ['bit_mask']))
    
    # 11. BitwiseXor to flip the bit.
    nodes.append(helper.make_node('BitwiseXor', ['int8_val', 'bit_mask'], ['flipped_int']))
    
    # 12. Cast flipped_int and int8_val to INT32.
    nodes.append(helper.make_node('Cast', ['flipped_int'], ['flipped_int32'], to=TensorProto.INT32))
    nodes.append(helper.make_node('Cast', ['int8_val'], ['int8_val32'], to=TensorProto.INT32))
    
    # 13. Subtract original from flipped.
    nodes.append(helper.make_node('Sub', ['flipped_int32', 'int8_val32'], ['perturbation_int32']))
    
    # 14. Cast result to FLOAT -> output.
    nodes.append(helper.make_node('Cast', ['perturbation_int32'], [output_name], to=TensorProto.FLOAT))
    
    return nodes, initializers

def create_INPUT16_block_restriction(delta_input, output_name, block_size=16):
    """
    Part B – Block Restriction Subgraph.
    
    This subgraph takes the perturbation tensor (delta_input, e.g. "delta_temp")
    and restricts it so that only a contiguous block of size block_size (default 16)
    along the last axis is nonzero. It replicates the single perturbation value into
    that block.
    
    Process:
      1. Compute the dynamic shape and rank of delta_input.
      2. Gather the size of the last dimension.
      3. Compute the number of blocks = last_dim // block_size.
      4. Generate a random block index and compute start_index_block.
      5. Get nonzero indices from delta_input using NonZero.
      6. Slice to extract the base coordinate (first column) and Squeeze it.
      7. Slice the base coordinate to remove the last element, yielding base_prefix.
      8. Concatenate base_prefix with start_index_block (which is assumed to have shape [1]) to form new_base.
      9. Create a constant offsets tensor [0, 1, …, block_size-1].
      10. Slice new_base to extract its last element (base_last) and add offsets to form block_indices.
      11. Unsqueeze and tile base_prefix and unsqueeze block_indices so that they can be concatenated to form final_indices.
      12. Create a zero tensor (FLOAT) of the same shape as delta_input.
      13. Create a ones tensor (FLOAT) of shape [block_size].
      14. Use ScatterND to place ones into the zero tensor at final_indices (this forms a mask).
      15. Use ReduceMax to extract the nonzero perturbation value from delta_input.
      16. Unsqueeze and tile the perturbation value to shape [block_size] to create replicated_value.
      17. Use ScatterND to place replicated_value into the zero tensor at final_indices, producing the final output.
    """
    nodes = []
    initializers = []
    
    # 1. Get dynamic shape.
    nodes.append(helper.make_node("Shape", [delta_input], ["delta_shape"], name="Shape_delta_temp"))
    
    # 2. Compute rank.
    nodes.append(helper.make_node("Size", ["delta_shape"], ["delta_rank"], name="Size_delta_shape"))
    
    # 3. Constant one.
    one_int64 = helper.make_tensor("one_int64", TensorProto.INT64, [1], [1])
    initializers.append(one_int64)
    
    # 4. Compute index_last = delta_rank - 1.
    nodes.append(helper.make_node("Sub", ["delta_rank", "one_int64"], ["index_last"], name="Compute_index_last"))
    
    # 5. Gather last dimension.
    nodes.append(helper.make_node("Gather", ["delta_shape", "index_last"], ["last_dim"], name="Gather_last_dim"))
    
    # 6. Constant for block_size.
    block_size_tensor = helper.make_tensor("block_size_tensor", TensorProto.INT64, [1], [block_size])
    initializers.append(block_size_tensor)
    
    # 7. Compute num_blocks = last_dim // block_size.
    nodes.append(helper.make_node("Div", ["last_dim", "block_size_tensor"], ["num_blocks"], name="Compute_num_blocks"))
    
    # 8. Cast num_blocks to FLOAT.
    nodes.append(helper.make_node("Cast", ["num_blocks"], ["num_blocks_float"], name="Cast_num_blocks", to=TensorProto.FLOAT))
    
    # 9. Generate a random block index.
    nodes.append(helper.make_node("RandomUniform", [], ["ru_block"],
                                  name="RandomUniform_block", low=0.0, high=1.0, shape=[1]))
    nodes.append(helper.make_node("Mul", ["ru_block", "num_blocks_float"], ["scaled_block"], name="Mul_block"))
    nodes.append(helper.make_node("Floor", ["scaled_block"], ["floor_block"], name="Floor_block"))
    nodes.append(helper.make_node("Cast", ["floor_block"], ["rand_block_index"], name="Cast_rand_block", to=TensorProto.INT64))
    
    # 10. Compute start_index_block = rand_block_index * block_size.
    nodes.append(helper.make_node("Mul", ["rand_block_index", "block_size_tensor"], ["start_index_block"], name="Compute_start_index_block"))
    
    # 11. Obtain nonzero indices from delta_input.
    nodes.append(helper.make_node("NonZero", [delta_input], ["nonzero_indices"], name="NonZero_delta_temp"))
    
    # 12. Slice to get the first column (base coordinate).
    slice_starts = helper.make_tensor("slice_starts", TensorProto.INT64, [1], [0])
    slice_ends  = helper.make_tensor("slice_ends",  TensorProto.INT64, [1], [1])
    slice_axes  = helper.make_tensor("slice_axes",  TensorProto.INT64, [1], [1])
    initializers.extend([slice_starts, slice_ends, slice_axes])
    nodes.append(helper.make_node("Slice", ["nonzero_indices", "slice_starts", "slice_ends", "slice_axes"],
                                  ["base_coord_unsqueezed"], name="Slice_nonzero"))
    
    # 13. Squeeze the unsqueezed coordinate (using constant unsqueeze_axes_1 from the main graph).
    nodes.append(helper.make_node("Squeeze", ["base_coord_unsqueezed", "unsqueeze_axes_1"],
                                  ["base_coord"], name="Squeeze_base_coord"))
    
    # 14. Remove the last element of base_coord to get base_prefix.
    base_prefix_starts = helper.make_tensor("base_prefix_starts", TensorProto.INT64, [1], [0])
    base_prefix_ends = helper.make_tensor("base_prefix_ends", TensorProto.INT64, [1], [-1])
    base_prefix_axes = helper.make_tensor("base_prefix_axes", TensorProto.INT64, [1], [0])
    initializers.extend([base_prefix_starts, base_prefix_ends, base_prefix_axes])
    nodes.append(helper.make_node("Slice", ["base_coord", "base_prefix_starts", "base_prefix_ends", "base_prefix_axes"],
                                  ["base_prefix"], name="Slice_base_prefix"))
    
    # 15. Concatenate base_prefix with start_index_block.
    # (start_index_block is assumed to be of shape [1], rank 1.)
    nodes.append(helper.make_node("Concat", ["base_prefix", "start_index_block"],
                                  ["new_base"], name="Concat_new_base", axis=0))
    
    # 16. Create constant offsets tensor: [0, 1, ..., block_size-1].
    offsets = list(range(block_size))
    offsets_tensor = helper.make_tensor("offsets", TensorProto.INT64, [block_size], offsets)
    initializers.append(offsets_tensor)
    
    # 17. Extract the last element from new_base using Slice.
    new_base_last_starts = helper.make_tensor("new_base_last_starts", TensorProto.INT64, [1], [-1])
    new_base_last_ends = helper.make_tensor("new_base_last_ends", TensorProto.INT64, [1], [2147483647])
    new_base_last_axes = helper.make_tensor("new_base_last_axes", TensorProto.INT64, [1], [0])
    initializers.extend([new_base_last_starts, new_base_last_ends, new_base_last_axes])
    nodes.append(helper.make_node("Slice", ["new_base", "new_base_last_starts", "new_base_last_ends", "new_base_last_axes"],
                                  ["base_last"], name="Slice_new_base_last"))
    
    # 18. Compute block_indices = base_last + offsets.
    nodes.append(helper.make_node("Add", ["base_last", "offsets"], ["block_indices"], name="Add_offsets"))
    
    # 19. Reassemble full indices for the contiguous block.
    nodes.append(helper.make_node("Unsqueeze", ["base_prefix", "unsqueeze_axes_0"],
                                  ["base_prefix_unsqueezed"], name="Unsqueeze_base_prefix"))
    tile_multipliers = helper.make_tensor("tile_multipliers", TensorProto.INT64, [2], [block_size, 1])
    initializers.append(tile_multipliers)
    nodes.append(helper.make_node("Tile", ["base_prefix_unsqueezed", "tile_multipliers"],
                                  ["base_prefix_tiled"], name="Tile_base_prefix"))
    nodes.append(helper.make_node("Unsqueeze", ["block_indices", "unsqueeze_axes_1"],
                                  ["block_indices_unsqueezed"], name="Unsqueeze_block_indices"))
    nodes.append(helper.make_node("Concat", ["base_prefix_tiled", "block_indices_unsqueezed"],
                                  ["final_indices"], name="Concat_final_indices", axis=1))
    
    # 20. Create a zero tensor (FLOAT) with the same shape as delta_input.
    nodes.append(helper.make_node("ConstantOfShape", ["delta_shape"], ["zero_tensor"], name="ConstantOfShape_zero",
                                  value=helper.make_tensor("zero_value", TensorProto.FLOAT, [1], [0.0])))
    
    # 21. Create a ones tensor of shape [block_size] (FLOAT).
    ones_tensor = helper.make_tensor("ones_tensor", TensorProto.FLOAT, [block_size], [1.0]*block_size)
    initializers.append(ones_tensor)

    nodes.append(helper.make_node("ReduceSum", [delta_input], ["perturb_value"], keepdims=0, name="ReduceSum_delta"))
    # 24. Unsqueeze perturb_value to shape [1] using unsqueeze_axes_0.
    nodes.append(helper.make_node("Unsqueeze", ["perturb_value", "unsqueeze_axes_0"], ["perturb_value_unsqueezed"], name="Unsqueeze_perturb_value"))
    # 25. Tile perturb_value to shape [block_size] using a tile multiplier.
    tile_multiplier_for_value = helper.make_tensor("tile_multiplier_for_value", TensorProto.INT64, [1], [block_size])
    initializers.append(tile_multiplier_for_value)
    nodes.append(helper.make_node("Tile", ["perturb_value_unsqueezed", "tile_multiplier_for_value"], ["replicated_value"], name="Tile_perturb_value"))
    
    # 26. Use ScatterND to place replicated_value into zero_tensor at final_indices.
    nodes.append(helper.make_node("ScatterND", ["zero_tensor", "final_indices", "replicated_value"],
                                  [output_name], name="ScatterND_final"))
    
    return nodes, initializers

def create_INPUT16_perturbation_model(input_name, final_output_name, bit_position=0, block_size=16):
    """
    Creates a complete ONNX model for the INPUT16 fault model.
    
    Part A computes a perturbation tensor via bit-flip (using create_quantized_fault_injection).
    Part B restricts that perturbation to a contiguous block of size block_size along the last axis,
    replicating the nonzero perturbation value into that block.
    """
    all_nodes = []
    all_initializers = []
    
    # Create constant nodes for unsqueeze axes at the very beginning.
    const_unsqueeze_axes_0 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_axes_0"],
        value=helper.make_tensor("unsqueeze_axes_0_value", TensorProto.INT64, [1], [0])
    )
    const_unsqueeze_axes_1 = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_axes_1"],
        value=helper.make_tensor("unsqueeze_axes_1_value", TensorProto.INT64, [1], [1])
    )
    all_nodes.extend([const_unsqueeze_axes_0, const_unsqueeze_axes_1])
    
    # Part A: Compute perturbation via bit-flip.
    partA_output = "delta_temp"
    nodesA, initA = create_quantized_fault_injection(input_name, partA_output, bit_position)
    all_nodes.extend(nodesA)
    all_initializers.extend(initA)
    
    # Part B: Restrict perturbation to a block and replicate the value.
    nodesB, initB = create_INPUT16_block_restriction(partA_output, final_output_name, block_size)
    all_nodes.extend(nodesB)
    all_initializers.extend(initB)
    
    # Define model input and output (assuming 3D: [batch, sequence, hidden]).
    input_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [None, None, None])
    output_vi = helper.make_tensor_value_info(final_output_name, TensorProto.FLOAT, [None, None, None])
    
    graph = helper.make_graph(
        all_nodes,
        "INPUT16_Perturbation_Graph",
        [input_vi],
        [output_vi],
        initializer=all_initializers
    )
    
    # Set the opset to 18 (or later) for new Unsqueeze/Squeeze signatures.
    opset = [helper.make_opsetid("", 18)]
    model = helper.make_model(graph, producer_name="INPUT16_Perturbation_Model", opset_imports=opset)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model

if __name__ == "__main__":
    model = create_INPUT16_perturbation_model("input_tensor", "delta_out", bit_position=0, block_size=16)
    save_model(model, "INPUT16_perturbation.onnx")
    print("Saved INPUT16 perturbation ONNX model as INPUT16_perturbation.onnx")
    
    # Test with ONNX Runtime.
    import onnxruntime as ort
    session = ort.InferenceSession("INPUT16_perturbation.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("Input name:", input_name)
    print("Output name:", output_name)
    
    # Create a dummy input; adjust shape as needed.
    dummy_input = np.random.randn(1, 10, 512).astype(np.float32)
    outputs = session.run([output_name], {input_name: dummy_input})
    perturbation = outputs[0]
    print("Perturbation output:")
    print(perturbation)
    nonzero_indices = np.argwhere(perturbation != 0)
    print("Nonzero indices in the perturbation tensor:")
    print(nonzero_indices)
    nonzero_values = perturbation[perturbation != 0]
    print("Nonzero values:")
    print(nonzero_values)
    print(perturbation.shape)
