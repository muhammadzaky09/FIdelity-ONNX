import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

def create_input16_mask(matmul_output="y", masked_output="y_masked", block_length=16):
    """
    Create a mask that zeros out all values except for a contiguous block in the last dimension.
    Works for both 3D and 4D tensors automatically.
    
    Args:
        matmul_output: Name of the tensor to mask
        masked_output: Name of the output tensor
        block_length: Length of the non-zero block
    
    Returns:
        List of ONNX nodes for the masking operation
    """
    nodes = []
    suffix = "_mask"
    
    # 1. Get the shape of the input tensor
    nodes.append(helper.make_node("Shape", inputs=[matmul_output], outputs=["shape" + suffix]))
    
    # 2. Calculate the rank of the tensor (number of dimensions)
    nodes.append(helper.make_node("Shape", inputs=["shape" + suffix], outputs=["shape_of_shape" + suffix]))
    const_zero_idx = helper.make_tensor("zero_idx" + suffix, TensorProto.INT64, [1], [0])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["zero_idx" + suffix], value=const_zero_idx))
    nodes.append(helper.make_node("Gather", inputs=["shape_of_shape" + suffix, "zero_idx" + suffix], 
                                 outputs=["rank_tensor" + suffix], axis=0))
    nodes.append(helper.make_node("Squeeze", inputs=["rank_tensor" + suffix], outputs=["rank" + suffix]))
    
    # 3. Check if rank is 3 or 4
    const_rank3 = helper.make_tensor("rank3" + suffix, TensorProto.INT64, [], [3])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["rank3" + suffix], value=const_rank3))
    nodes.append(helper.make_node("Equal", inputs=["rank" + suffix, "rank3" + suffix], outputs=["is_rank3" + suffix]))
    
    # 4. Prepare constants
    const_zero = helper.make_tensor("zero" + suffix, TensorProto.INT64, [], [0])
    const_one = helper.make_tensor("one" + suffix, TensorProto.INT64, [], [1])
    const_two = helper.make_tensor("two" + suffix, TensorProto.INT64, [], [2])
    const_three = helper.make_tensor("three" + suffix, TensorProto.INT64, [], [3])
    
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["zero" + suffix], value=const_zero))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["one" + suffix], value=const_one))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["two" + suffix], value=const_two))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["three" + suffix], value=const_three))
    
    # 5. Determine the index for the last dimension based on rank
    nodes.append(helper.make_node("If", inputs=["is_rank3" + suffix], 
                                 outputs=["last_dim_idx" + suffix],
                                 then_branch=helper.make_graph(
                                     [helper.make_node("Identity", inputs=["two" + suffix], outputs=["last_dim_idx" + suffix])],
                                     "then_graph",
                                     [],
                                     [helper.make_tensor_value_info("last_dim_idx" + suffix, TensorProto.INT64, [])]
                                 ),
                                 else_branch=helper.make_graph(
                                     [helper.make_node("Identity", inputs=["three" + suffix], outputs=["last_dim_idx" + suffix])],
                                     "else_graph",
                                     [],
                                     [helper.make_tensor_value_info("last_dim_idx" + suffix, TensorProto.INT64, [])]
                                 )))
    
    # 6. Get the size of the last dimension
    nodes.append(helper.make_node("Gather", inputs=["shape" + suffix, "last_dim_idx" + suffix], 
                                 outputs=["last_dim_size_tensor" + suffix], axis=0))
    nodes.append(helper.make_node("Squeeze", inputs=["last_dim_size_tensor" + suffix], 
                                 outputs=["last_dim_size" + suffix]))
    
    # 7. Calculate random start index for the non-zero block
    const_block = helper.make_tensor("block_size" + suffix, TensorProto.INT64, [], [block_length])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["block_size" + suffix], value=const_block))
    
    # Calculate max start index (last_dim_size - block_length)
    nodes.append(helper.make_node("Sub", inputs=["last_dim_size" + suffix, "block_size" + suffix], 
                                 outputs=["max_start" + suffix]))
    
    # Ensure max_start is not negative
    nodes.append(helper.make_node("Max", inputs=["max_start" + suffix, "zero" + suffix], 
                                 outputs=["safe_max_start" + suffix]))
    
    # Add 1 to include max_start itself in the range
    nodes.append(helper.make_node("Add", inputs=["safe_max_start" + suffix, "one" + suffix], 
                                 outputs=["rand_range" + suffix]))
    
    # Generate random start index between 0 and rand_range
    nodes.append(helper.make_node("Cast", inputs=["rand_range" + suffix], 
                                 outputs=["rand_range_float" + suffix], to=TensorProto.FLOAT))
    
    nodes.append(helper.make_node("RandomUniform", inputs=[], outputs=["rand_val" + suffix], 
                                 dtype=TensorProto.FLOAT, high=1.0, low=0.0, shape=[1]))
    
    nodes.append(helper.make_node("Mul", inputs=["rand_val" + suffix, "rand_range_float" + suffix], 
                                 outputs=["scaled_rand" + suffix]))
    
    nodes.append(helper.make_node("Floor", inputs=["scaled_rand" + suffix], 
                                 outputs=["start_idx_float" + suffix]))
    
    nodes.append(helper.make_node("Cast", inputs=["start_idx_float" + suffix], 
                                 outputs=["start_idx_tensor" + suffix], to=TensorProto.INT64))
    
    nodes.append(helper.make_node("Squeeze", inputs=["start_idx_tensor" + suffix], 
                                 outputs=["start_idx" + suffix]))
    
    # 8. Calculate end index
    nodes.append(helper.make_node("Add", inputs=["start_idx" + suffix, "block_size" + suffix], 
                                 outputs=["end_idx" + suffix]))
    
    # 9. Create a range of indices for the last dimension
    nodes.append(helper.make_node("Range", inputs=["zero" + suffix, "last_dim_size" + suffix, "one" + suffix], 
                                 outputs=["indices" + suffix]))
    
    # 10. Create boolean mask for indices between start_idx and end_idx
    nodes.append(helper.make_node("GreaterOrEqual", inputs=["indices" + suffix, "start_idx" + suffix], 
                                 outputs=["ge_mask" + suffix]))
    
    nodes.append(helper.make_node("Less", inputs=["indices" + suffix, "end_idx" + suffix], 
                                 outputs=["lt_mask" + suffix]))
    
    nodes.append(helper.make_node("And", inputs=["ge_mask" + suffix, "lt_mask" + suffix], 
                                 outputs=["bool_mask_1d" + suffix]))
    
    # 11. Convert boolean mask to float16
    nodes.append(helper.make_node("Cast", inputs=["bool_mask_1d" + suffix], 
                                 outputs=["float_mask_1d" + suffix], to=TensorProto.FLOAT16))
    
    # 12. Create unsqueeze axes based on rank
    # For rank 3: unsqueeze axes = [0, 1]
    # For rank 4: unsqueeze axes = [0, 1, 2]
    
    # Create axes for 3D tensor [0, 1]
    const_axes_3d = helper.make_tensor("axes_3d" + suffix, TensorProto.INT64, [2], [0, 1])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["axes_3d" + suffix], value=const_axes_3d))
    
    # Create axes for 4D tensor [0, 1, 2]
    const_axes_4d = helper.make_tensor("axes_4d" + suffix, TensorProto.INT64, [3], [0, 1, 2])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["axes_4d" + suffix], value=const_axes_4d))
    
    # Choose axes based on rank
    nodes.append(helper.make_node("If", inputs=["is_rank3" + suffix], 
                                 outputs=["unsqueeze_axes" + suffix],
                                 then_branch=helper.make_graph(
                                     [helper.make_node("Identity", inputs=["axes_3d" + suffix], outputs=["unsqueeze_axes" + suffix])],
                                     "then_graph_axes",
                                     [],
                                     [helper.make_tensor_value_info("unsqueeze_axes" + suffix, TensorProto.INT64, [2])]
                                 ),
                                 else_branch=helper.make_graph(
                                     [helper.make_node("Identity", inputs=["axes_4d" + suffix], outputs=["unsqueeze_axes" + suffix])],
                                     "else_graph_axes",
                                     [],
                                     [helper.make_tensor_value_info("unsqueeze_axes" + suffix, TensorProto.INT64, [3])]
                                 )))
    
    # 13. Reshape 1D mask using unsqueeze
    nodes.append(helper.make_node("Unsqueeze", inputs=["float_mask_1d" + suffix, "unsqueeze_axes" + suffix], 
                                 outputs=["unsqueezed_mask" + suffix]))
    
    # 14. Extract individual dimensions for tiling
    # Extract first dimension (B - dim 0)
    nodes.append(helper.make_node("Gather", inputs=["shape" + suffix, "zero" + suffix], 
                                 outputs=["dim0_tensor" + suffix], axis=0))
    nodes.append(helper.make_node("Squeeze", inputs=["dim0_tensor" + suffix], 
                                 outputs=["dim0" + suffix]))
    
    # Extract second dimension (S for 3D or C for 4D - dim 1)
    nodes.append(helper.make_node("Gather", inputs=["shape" + suffix, "one" + suffix], 
                                 outputs=["dim1_tensor" + suffix], axis=0))
    nodes.append(helper.make_node("Squeeze", inputs=["dim1_tensor" + suffix], 
                                 outputs=["dim1" + suffix]))
    
    # For 4D only - Extract third dimension (S - dim 2)
    nodes.append(helper.make_node("Gather", inputs=["shape" + suffix, "two" + suffix], 
                                 outputs=["dim2_tensor" + suffix], axis=0))
    nodes.append(helper.make_node("Squeeze", inputs=["dim2_tensor" + suffix], 
                                 outputs=["dim2" + suffix]))
    
    # 15. Create tile multiples tensors
    # For 3D: [dim0, dim1, 1]
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["tile_shape_3d" + suffix], 
                                 value=helper.make_tensor("tile_shape_3d" + suffix, TensorProto.INT64, [3], [0, 0, 1])))
    
    # Update tile_shape_3d with actual dimensions
    nodes.append(helper.make_node("ScatterElements", 
                                 inputs=["tile_shape_3d" + suffix, "zero" + suffix, "dim0" + suffix], 
                                 outputs=["tile_shape_3d_temp" + suffix], axis=0))
    nodes.append(helper.make_node("ScatterElements", 
                                 inputs=["tile_shape_3d_temp" + suffix, "one" + suffix, "dim1" + suffix], 
                                 outputs=["tile_multiples_3d" + suffix], axis=0))
    
    # For 4D: [dim0, dim1, dim2, 1]
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["tile_shape_4d" + suffix], 
                                 value=helper.make_tensor("tile_shape_4d" + suffix, TensorProto.INT64, [4], [0, 0, 0, 1])))
    
    # Update tile_shape_4d with actual dimensions
    nodes.append(helper.make_node("ScatterElements", 
                                 inputs=["tile_shape_4d" + suffix, "zero" + suffix, "dim0" + suffix], 
                                 outputs=["tile_shape_4d_temp1" + suffix], axis=0))
    nodes.append(helper.make_node("ScatterElements", 
                                 inputs=["tile_shape_4d_temp1" + suffix, "one" + suffix, "dim1" + suffix], 
                                 outputs=["tile_shape_4d_temp2" + suffix], axis=0))
    nodes.append(helper.make_node("ScatterElements", 
                                 inputs=["tile_shape_4d_temp2" + suffix, "two" + suffix, "dim2" + suffix], 
                                 outputs=["tile_multiples_4d" + suffix], axis=0))
    
    # Choose tile multiples based on rank
    nodes.append(helper.make_node("If", inputs=["is_rank3" + suffix], 
                                 outputs=["tile_multiples" + suffix],
                                 then_branch=helper.make_graph(
                                     [helper.make_node("Identity", 
                                                     inputs=["tile_multiples_3d" + suffix], 
                                                     outputs=["tile_multiples" + suffix])],
                                     "then_graph_tile",
                                     [],
                                     [helper.make_tensor_value_info("tile_multiples" + suffix, TensorProto.INT64, [3])]
                                 ),
                                 else_branch=helper.make_graph(
                                     [helper.make_node("Identity", 
                                                     inputs=["tile_multiples_4d" + suffix], 
                                                     outputs=["tile_multiples" + suffix])],
                                     "else_graph_tile",
                                     [],
                                     [helper.make_tensor_value_info("tile_multiples" + suffix, TensorProto.INT64, [4])]
                                 )))
    
    # 16. Tile the mask to full shape
    nodes.append(helper.make_node("Tile", inputs=["unsqueezed_mask" + suffix, "tile_multiples" + suffix], 
                                 outputs=["mask_full" + suffix]))
    
    # 17. Apply mask to input
    nodes.append(helper.make_node("Mul", inputs=[matmul_output, "mask_full" + suffix], 
                                 outputs=[masked_output]))
    
    return nodes

def build_masked_model(input_shape, block_length=16):
    """
    Build an ONNX model that applies the mask to an input tensor.
    
    Args:
        input_shape: Shape of the input tensor (3D or 4D)
        block_length: Number of non-zero elements to keep in the last dimension
        
    Returns:
        ONNX model
    """
    # Define model inputs and outputs
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT16, input_shape)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT16, input_shape)
    
    # Create mask nodes
    mask_nodes = create_input16_mask("input", "output", block_length)
    
    # Create ONNX graph and model
    graph = helper.make_graph(
        mask_nodes, 
        "test_mask",
        [input_tensor],
        [output_tensor]
    )
    
    model = helper.make_model(graph, producer_name="test_mask")
    model.opset_import[0].version = 18
    
    # Save model to file for debugging (optional)
    # onnx.save(model, f"mask_model_{len(input_shape)}D.onnx")
    
    return model

def count_nonzero_by_position(output):
    """
    Count non-zero elements and analyze their distribution.
    """
    # Find all non-zero elements
    non_zero_indices = np.nonzero(output)
    non_zero_count = len(non_zero_indices[0])
    
    # Group by all dimensions except the last
    position_counts = {}
    for i in range(non_zero_count):
        # Create a tuple of all indices except the last one
        position = tuple(dim[i] for dim in non_zero_indices[:-1])
        last_dim_idx = non_zero_indices[-1][i]
        
        if position not in position_counts:
            position_counts[position] = []
        position_counts[position].append(last_dim_idx)
    
    return position_counts

def test_mask():
    """
    Test the masking implementation with both 3D and 4D tensors.
    """
    # Test parameters
    block_length = 16
    
    # Test shapes - both 3D and 4D
    test_shapes = [
        (1, 25, 32),       # 3D tensor
        (1, 40, 25, 32),   # 4D tensor
        (2, 32, 128),      # 3D tensor with different dimensions
        (2, 32, 20, 128)   # 4D tensor like your example
    ]
    
    all_passed = True
    
    for i, shape in enumerate(test_shapes):
        print(f"\n\nTest {i+1}: Testing with shape {shape} ({len(shape)}D tensor)")
        print("-" * 60)
        
        try:
            # Build model for this shape
            model = build_masked_model(shape, block_length)
            
            # Create input tensor with all ones
            input_tensor = np.ones(shape, dtype=np.float16)
            
            # Run inference
            session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            output = session.run(None, {"input": input_tensor})[0]
            
            # Check results
            non_zero_count = np.count_nonzero(output)
            print(f"Input shape: {input_tensor.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Non-zero elements: {non_zero_count} (should be {block_length})")
            
            # Analyze distribution of non-zero elements
            position_counts = count_nonzero_by_position(output)
            
            print(f"\nNon-zero elements distribution:")
            total_nonzero = 0
            for position, indices in position_counts.items():
                indices.sort()
                is_contiguous = (indices == list(range(min(indices), max(indices)+1)))
                print(f"  Position {position}: {len(indices)} elements at indices {indices}")
                print(f"  Contiguous: {'Yes' if is_contiguous else 'No'}")
                total_nonzero += len(indices)
            
            # Check if we have exactly block_length non-zero elements
            if non_zero_count == block_length:
                print(f"\n✅ SUCCESS: Found exactly {block_length} non-zero elements")
            else:
                print(f"\n❌ FAILURE: Found {non_zero_count} non-zero elements (expected {block_length})")
                all_passed = False
            
            # Check if there's only one position with non-zero elements
            if len(position_counts) == 1:
                print(f"✅ SUCCESS: All non-zero elements are at a single position")
            else:
                print(f"❌ FAILURE: Non-zero elements are distributed across {len(position_counts)} positions")
                all_passed = False
            
            # Check if the elements are contiguous
            position = list(position_counts.keys())[0]
            indices = position_counts[position]
            is_contiguous = (len(indices) == indices[-1] - indices[0] + 1)
            if is_contiguous:
                print(f"✅ SUCCESS: Non-zero elements are contiguous in the last dimension")
            else:
                print(f"❌ FAILURE: Non-zero elements are NOT contiguous in the last dimension")
                all_passed = False
            
        except Exception as e:
            print(f"❌ ERROR with shape {shape}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    if all_passed:
        print("\n\n🎉 ALL TESTS PASSED! The mask works correctly for both 3D and 4D tensors.")
    else:
        print("\n\n❌ SOME TESTS FAILED. Please check the error messages above.")

if __name__ == "__main__":
    test_mask()