import numpy as np
import onnx
from onnx import helper, TensorProto, shape_inference
import onnxruntime as ort
import os

def create_input16_mask(matmul_output="y", masked_output="y_masked", block_length=16):
    """
    Modified to work on 4D tensors with shape [B,S,H,W].
    The mask will PRESERVE a specific section of length block_length in the last dimension (W)
    and zero out everything else.
    """
    nodes = []
    # 1. Get the shape of the MatMul output.
    nodes.append(helper.make_node("Shape", inputs=[matmul_output], outputs=["y_shape"]))
    
    # 2. Extract last dimension (W) from y_shape (assume y_shape = [B,S,H,W]).
    const_W_start = helper.make_tensor("W_start", TensorProto.INT64, [1], [3])
    const_W_end   = helper.make_tensor("W_end",   TensorProto.INT64, [1], [4])
    const_W_axes  = helper.make_tensor("W_axes",  TensorProto.INT64, [1], [0])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["W_starts"], value=const_W_start))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["W_ends"],   value=const_W_end))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["W_axes"],   value=const_W_axes))
    nodes.append(helper.make_node("Slice", inputs=["y_shape", "W_starts", "W_ends", "W_axes"], outputs=["W_value"]))
    # Squeeze to get scalar W. (Omit axes so all 1-dim are removed.)
    nodes.append(helper.make_node("Squeeze", inputs=["W_value"], outputs=["W_scalar"]))
    
    # 3. Compute dynamic start index along W.
    const_block = helper.make_tensor("block_length", TensorProto.INT64, [], [block_length])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["block_length_const"], value=const_block))
    nodes.append(helper.make_node("Sub", inputs=["W_scalar", "block_length_const"], outputs=["W_minus_block"]))
    const_one = helper.make_tensor("one_const", TensorProto.INT64, [], [1])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_const"], value=const_one))
    nodes.append(helper.make_node("Add", inputs=["W_minus_block", "one_const"], outputs=["range_size"]))
    nodes.append(helper.make_node("Cast", inputs=["range_size"], outputs=["range_size_float"], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("RandomUniform", inputs=[], outputs=["rand_val_temp"], dtype=TensorProto.FLOAT, high=1.0, low=0.0, shape=[1]))
    nodes.append(helper.make_node("Squeeze", inputs=["rand_val_temp"], outputs=["rand_val"]))
    nodes.append(helper.make_node("Mul", inputs=["rand_val", "range_size_float"], outputs=["rand_scaled"]))
    nodes.append(helper.make_node("Floor", inputs=["rand_scaled"], outputs=["rand_index_float"]))
    nodes.append(helper.make_node("Cast", inputs=["rand_index_float"], outputs=["start_index_dynamic"], to=TensorProto.INT64))
    nodes.append(helper.make_node("Add", inputs=["start_index_dynamic", "block_length_const"], outputs=["end_index_dynamic"]))
    
    # 4. Build 1D mask over W - preserve only the block, zero everything else
    const_zero = helper.make_tensor("zero_const", TensorProto.INT64, [], [0])
    const_one_step = helper.make_tensor("one_step", TensorProto.INT64, [], [1])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["zero_const_W"], value=const_zero))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_const_W_step"], value=const_one_step))
    nodes.append(helper.make_node("Range", inputs=["zero_const_W", "W_scalar", "one_const_W_step"], outputs=["indices_W"]))
    
    # Create mask where 1 = positions to preserve (within the block), 0 = positions to zero out
    nodes.append(helper.make_node("GreaterOrEqual", inputs=["indices_W", "start_index_dynamic"], outputs=["ge_mask_W"]))
    nodes.append(helper.make_node("Less", inputs=["indices_W", "end_index_dynamic"], outputs=["lt_mask_W"]))
    nodes.append(helper.make_node("And", inputs=["ge_mask_W", "lt_mask_W"], outputs=["mask_bool_W"]))
    
    nodes.append(helper.make_node("Cast", inputs=["mask_bool_W"], outputs=["mask_1d"], to=TensorProto.FLOAT16))
    
    # 5. Unsqueeze mask_1d to shape [1,1,1,W]. Use a constant axes tensor [0,1,2].
    const_unsqueeze_axes = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [3], [0,1,2])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["unsqueeze_axes"], value=const_unsqueeze_axes))
    nodes.append(helper.make_node("Unsqueeze", inputs=["mask_1d", "unsqueeze_axes"], outputs=["mask_unsqueezed"]))
    
    # 6. Tile mask_unsqueezed to shape [B,S,H,W]. Extract B, S, and H from y_shape.
    # Extract B dimension
    const_B_start = helper.make_tensor("B_start", TensorProto.INT64, [1], [0])
    const_B_end   = helper.make_tensor("B_end",   TensorProto.INT64, [1], [1])
    const_B_axes  = helper.make_tensor("B_axes",  TensorProto.INT64, [1], [0])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["B_starts_out"], value=const_B_start))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["B_ends_out"], value=const_B_end))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["B_axes_out"], value=const_B_axes))
    nodes.append(helper.make_node("Slice", inputs=["y_shape", "B_starts_out", "B_ends_out", "B_axes_out"], outputs=["B_value_out"]))
    nodes.append(helper.make_node("Squeeze", inputs=["B_value_out"], outputs=["B_scalar_out"]))
    
    # Extract S dimension
    const_S_start = helper.make_tensor("S_start", TensorProto.INT64, [1], [1])
    const_S_end   = helper.make_tensor("S_end",   TensorProto.INT64, [1], [2])
    const_S_axes  = helper.make_tensor("S_axes",  TensorProto.INT64, [1], [0])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["S_starts_out"], value=const_S_start))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["S_ends_out"], value=const_S_end))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["S_axes_out"], value=const_S_axes))
    nodes.append(helper.make_node("Slice", inputs=["y_shape", "S_starts_out", "S_ends_out", "S_axes_out"], outputs=["S_value_out"]))
    nodes.append(helper.make_node("Squeeze", inputs=["S_value_out"], outputs=["S_scalar_out"]))
    
    # Extract H dimension
    const_H_start = helper.make_tensor("H_start", TensorProto.INT64, [1], [2])
    const_H_end   = helper.make_tensor("H_end",   TensorProto.INT64, [1], [3])
    const_H_axes  = helper.make_tensor("H_axes",  TensorProto.INT64, [1], [0])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_starts_out"], value=const_H_start))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_ends_out"], value=const_H_end))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_axes_out"], value=const_H_axes))
    nodes.append(helper.make_node("Slice", inputs=["y_shape", "H_starts_out", "H_ends_out", "H_axes_out"], outputs=["H_value_out"]))
    nodes.append(helper.make_node("Squeeze", inputs=["H_value_out"], outputs=["H_scalar_out"]))
    
    # Convert scalars to 1D tensors.
    const_unsqueeze_axis0 = helper.make_tensor("unsqueeze_axis0", TensorProto.INT64, [1], [0])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["unsqueeze_axis0"], value=const_unsqueeze_axis0))
    nodes.append(helper.make_node("Unsqueeze", inputs=["B_scalar_out", "unsqueeze_axis0"], outputs=["B_1d"]))
    nodes.append(helper.make_node("Unsqueeze", inputs=["S_scalar_out", "unsqueeze_axis0"], outputs=["S_1d"]))
    nodes.append(helper.make_node("Unsqueeze", inputs=["H_scalar_out", "unsqueeze_axis0"], outputs=["H_1d"]))
    
    const_one_tile = helper.make_tensor("one_tile", TensorProto.INT64, [1], [1])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_for_tile"], value=const_one_tile))
    nodes.append(helper.make_node("Concat", inputs=["B_1d", "S_1d", "H_1d", "one_for_tile"], outputs=["tile_multiples"], axis=0))
    nodes.append(helper.make_node("Tile", inputs=["mask_unsqueezed", "tile_multiples"], outputs=["mask_full"]))
    
    # 7. Multiply the MatMul output with the mask.
    nodes.append(helper.make_node("Mul", inputs=[matmul_output, "mask_full"], outputs=[masked_output]))
    
    return nodes


def test_4d_masking():
    """
    Test the 4D masking operation with a carefully crafted example.
    
    This function:
    1. Creates a sample 4D tensor with all ones
    2. Builds an ONNX model with the masking operation using a block length of 8
    3. Runs the model to mask a portion of the tensor
    4. Analyzes the output to verify the masking was applied correctly
    """
    print("Testing 4D masking operation...")
    
    # Configuration for the test
    batch_size = 2
    seq_length = 3
    hidden_dim = 4
    width = 32   # Must be larger than block_length
    block_length = 8
    model_path = "test_4d_mask.onnx"
    
    # Create a 4D input tensor with shape [B,S,H,W] filled with ones
    input_shape = [batch_size, seq_length, hidden_dim, width]
    input_data = np.ones(input_shape, dtype=np.float16)
    
    # Create ONNX graph inputs and outputs
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT16, input_shape)
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT16, input_shape)
    masked_output = helper.make_tensor_value_info("masked_output", TensorProto.FLOAT16, input_shape)
    
    # Create a simple identity op to pass through the input
    identity_node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["output"]
    )
    
    # Create masking nodes
    masking_nodes = create_input16_mask(
        matmul_output="output", 
        masked_output="masked_output",
        block_length=block_length
    )
    
    # Create the graph
    graph = helper.make_graph(
        [identity_node] + masking_nodes,
        "test_4d_masking",
        [input_tensor],
        [output_tensor, masked_output]
    )
    
    # Create the model with opset 14 (for Unsqueeze with axes as input)
    model = helper.make_model(graph)
    model.opset_import[0].version = 14
    
    # Run shape inference to validate the model
    try:
        model = shape_inference.infer_shapes(model)
        print("Shape inference passed.")
    except Exception as e:
        print(f"Shape inference failed: {e}")
        return
    
    # Check and save the model
    try:
        onnx.checker.check_model(model)
        print("Model validation passed.")
        onnx.save(model, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Model validation failed: {e}")
        return
    
    # Run inference on the model
    try:
        print("Running inference...")
        session = ort.InferenceSession(model_path)
        outputs = session.run(["output", "masked_output"], {"input": input_data})
        original = outputs[0]
        masked = outputs[1]
        
        # Analyze the masked output
        print(f"Input shape: {input_data.shape}")
        print(f"Original output shape: {original.shape}")
        print(f"Masked output shape: {masked.shape}")
        
        # Find the preserved region by finding non-zero values
        preserved_indices = np.where(masked > 0)
        
        if len(preserved_indices[0]) > 0:
            # Get the preserved boundary
            min_w = np.min(preserved_indices[3])
            max_w = np.max(preserved_indices[3])
            
            print(f"\nPreserved block at positions {min_w} to {max_w} in the W dimension")
            print(f"Preserved block width: {max_w - min_w + 1} (expected {block_length})")
            
            # Check if preservation happened in the right region
            if max_w - min_w + 1 == block_length:
                print("PASS: Block preservation applied correctly with expected block length.")
            else:
                print(f"FAIL: Unexpected preserved width. Got {max_w - min_w + 1}, expected {block_length}.")
            
            # Verify the pattern is consistent across all dimensions
            unique_b = np.unique(preserved_indices[0])
            unique_s = np.unique(preserved_indices[1])
            unique_h = np.unique(preserved_indices[2])
            
            print(f"Batch indices with preserved values: {unique_b}")
            print(f"Sequence indices with preserved values: {unique_s}")
            print(f"Hidden dimension indices with preserved values: {unique_h}")
            
            # Check if preservation is applied to all elements
            if len(unique_b) == batch_size and len(unique_s) == seq_length and len(unique_h) == hidden_dim:
                print("PASS: Preservation applied consistently across all B, S, H dimensions.")
            else:
                print("FAIL: Preservation not applied consistently across all dimensions.")
                
            # Print full tensor values for each sample in the first batch
            print("\nDetailed tensor values for each position in the first batch, first sequence:")
            for h in range(hidden_dim):
                print(f"\nH={h}, values across W dimension:")
                values = masked[0, 0, h, :]
                print("W indices:   ", list(range(width)))
                print("Values:      ", values)
                print("Mask (1=preserved):", (values > 0).astype(np.int8))
        else:
            print("FAIL: No preserved values found in the output tensor.")
            
    except Exception as e:
        print(f"Inference failed: {e}")
        return

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    test_4d_masking()