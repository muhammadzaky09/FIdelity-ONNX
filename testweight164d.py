import numpy as np
import onnx
from onnx import helper, TensorProto, shape_inference
import onnxruntime as ort
import os

def create_weight16_mask(matmul_output="y", masked_output="y_masked", block_length=4):
    """
    Create a mask that keeps only 'block_length' consecutive elements in the hidden dimension (-2).
    Modified to work with 4D tensors of shape [batch, sequence, hidden, width].
    """
    nodes = []
    suffix = "_mask"
    
    # 1. Get the shape of the input tensor
    nodes.append(helper.make_node(
        "Shape",
        inputs=[matmul_output],
        outputs=["y_shape" + suffix]
    ))
    
    # 2. Get the hidden dimension size (dimension 2)
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["dim2_idx" + suffix],
        value=helper.make_tensor(
            name="dim2_idx_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[2]  # Third dimension (index 2) - this is the hidden dimension
        )
    ))
    
    nodes.append(helper.make_node(
        "Gather",
        inputs=["y_shape" + suffix, "dim2_idx" + suffix],
        outputs=["hidden_dim_tensor" + suffix],
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
        inputs=["hidden_dim_tensor" + suffix, "squeeze_axes" + suffix],
        outputs=["hidden_dim" + suffix]
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
    
    # 5. Create a range of indices (0 to hidden_dim-1)
    nodes.append(helper.make_node(
        "Range",
        inputs=["zero_scalar" + suffix, "hidden_dim" + suffix, "one_scalar" + suffix],
        outputs=["hidden_indices" + suffix]
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
    
    # 7. Calculate valid block length (min of block_length and hidden_dim)
    nodes.append(helper.make_node(
        "Min",
        inputs=["block_len" + suffix, "hidden_dim" + suffix],
        outputs=["valid_block" + suffix]
    ))
    
    # 8. Calculate max start index
    nodes.append(helper.make_node(
        "Sub",
        inputs=["hidden_dim" + suffix, "valid_block" + suffix],
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
        inputs=["hidden_indices" + suffix, "start_idx" + suffix],
        outputs=["ge_mask" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Less",
        inputs=["hidden_indices" + suffix, "end_idx" + suffix],
        outputs=["lt_mask" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "And",
        inputs=["ge_mask" + suffix, "lt_mask" + suffix],
        outputs=["bool_mask_1d" + suffix]
    ))
    
    # 12. Create shape for reshaping the mask to 4D, with masking on the hidden dimension
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["reshape_shape" + suffix],
        value=helper.make_tensor(
            name="reshape_shape_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[4],
            vals=[1, 1, -1, 1]  # [1, 1, hidden, 1] with hidden dim inferred
        )
    ))
    
    # 13. Reshape the boolean mask to 4D for proper broadcasting
    nodes.append(helper.make_node(
        "Reshape",
        inputs=["bool_mask_1d" + suffix, "reshape_shape" + suffix],
        outputs=["bool_mask_4d" + suffix]
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
        inputs=["bool_mask_4d" + suffix, matmul_output, "zeros" + suffix],
        outputs=[masked_output]
    ))
    
    return nodes

def test_4d_hidden_masking():
    """
    Test the 4D masking operation on the hidden dimension (-2).
    
    This function:
    1. Creates a sample 4D tensor with all ones
    2. Builds an ONNX model with the masking operation
    3. Runs the model to mask a portion of the hidden dimension
    4. Analyzes the output to verify the masking was applied correctly
    """
    print("Testing 4D hidden dimension masking...")
    
    # Configuration for the test
    batch_size = 2
    seq_length = 3
    hidden_dim = 8   # Must be larger than block_length
    width = 4
    block_length = 4
    model_path = "test_4d_hidden_mask.onnx"
    
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
    masking_nodes = create_weight16_mask(
        matmul_output="output", 
        masked_output="masked_output",
        block_length=block_length
    )
    
    # Create the graph
    graph = helper.make_graph(
        [identity_node] + masking_nodes,
        "test_4d_hidden_masking",
        [input_tensor],
        [output_tensor, masked_output]
    )
    
    # Create the model
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
            # Get the preserved boundary in the hidden dimension (dim 2)
            min_h = np.min(preserved_indices[2])
            max_h = np.max(preserved_indices[2])
            
            print(f"\nPreserved block at positions {min_h} to {max_h} in the hidden dimension")
            print(f"Preserved block width: {max_h - min_h + 1} (expected {block_length})")
            
            # Check if preservation happened in the right region
            if max_h - min_h + 1 == block_length:
                print("PASS: Block preservation applied correctly with expected block length.")
            else:
                print(f"FAIL: Unexpected preserved width. Got {max_h - min_h + 1}, expected {block_length}.")
            
            # Verify the pattern is consistent across all dimensions
            unique_b = np.unique(preserved_indices[0])
            unique_s = np.unique(preserved_indices[1])
            unique_w = np.unique(preserved_indices[3])
            
            print(f"Batch indices with preserved values: {unique_b}")
            print(f"Sequence indices with preserved values: {unique_s}")
            print(f"Width indices with preserved values: {unique_w}")
            
            # Check if preservation is applied to all elements
            if len(unique_b) == batch_size and len(unique_s) == seq_length and len(unique_w) == width:
                print("PASS: Preservation applied consistently across all B, S, W dimensions.")
            else:
                print("FAIL: Preservation not applied consistently across all dimensions.")
                
            # Print full tensor values for a single example
            print("\nDetailed tensor values for position B=0, S=0, across H dimension:")
            print("H indices:   ", list(range(hidden_dim)))
            print("Values at W=0:", masked[0, 0, :, 0])
            print("Mask (1=preserved):", (masked[0, 0, :, 0] > 0).astype(np.int8))
            
            print("\nThe entire tensor for B=0, S=0:")
            for w in range(width):
                print(f"W={w}, values across H dimension:")
                values = masked[0, 0, :, w]
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
    test_4d_hidden_masking()