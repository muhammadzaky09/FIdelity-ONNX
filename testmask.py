import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

def create_input16_mask(matmul_output="y", masked_output="y_masked", block_length=2):
    nodes = []
    # 1. Get the shape of the MatMul output.
    nodes.append(helper.make_node("Shape", inputs=[matmul_output], outputs=["y_shape"]))
    
    # 2. Extract the head dimension from y_shape.
    # Assuming y_shape = [B, S, num_heads, head_dim], we slice indices [3,4] to get head_dim.
    const_H_start = helper.make_tensor("H_start", TensorProto.INT64, [1], [3])
    const_H_end   = helper.make_tensor("H_end",   TensorProto.INT64, [1], [4])
    const_H_axes  = helper.make_tensor("H_axes",  TensorProto.INT64, [1], [0])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_starts"], value=const_H_start))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_ends"],   value=const_H_end))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_axes"],   value=const_H_axes))
    nodes.append(helper.make_node("Slice", inputs=["y_shape", "H_starts", "H_ends", "H_axes"], outputs=["H_value"]))
    # Squeeze to get scalar head_dim.
    nodes.append(helper.make_node("Squeeze", inputs=["H_value"], outputs=["H_scalar"]))
    
    # 3. Compute dynamic start index along head_dim.
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
    
    # 4. Build 1D mask over the head dimension.
    # Create a range over the head dimension.
    const_zero = helper.make_tensor("zero_const", TensorProto.INT64, [], [0])
    const_one_step = helper.make_tensor("one_step", TensorProto.INT64, [], [1])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["zero_const_H"], value=const_zero))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_const_H_step"], value=const_one_step))
    nodes.append(helper.make_node("Range", inputs=["zero_const_H", "H_scalar", "one_const_H_step"], outputs=["indices_H"]))
    
    # Create boolean masks for indices within the dynamic range [start_index_dynamic, end_index_dynamic).
    nodes.append(helper.make_node("GreaterOrEqual", inputs=["indices_H", "start_index_dynamic"], outputs=["ge_mask_H"]))
    nodes.append(helper.make_node("Less", inputs=["indices_H", "end_index_dynamic"], outputs=["lt_mask_H"]))
    nodes.append(helper.make_node("And", inputs=["ge_mask_H", "lt_mask_H"], outputs=["mask_bool_H"]))
    # Cast boolean mask to FLOAT.
    nodes.append(helper.make_node("Cast", inputs=["mask_bool_H"], outputs=["mask_1d"], to=TensorProto.FLOAT))
    
    # 5. Create a Constant node for reshape_shape.
    # We want the mask to have shape [1, 1, 1, head_dim] for broadcasting.
    reshape_shape_tensor = helper.make_tensor("reshape_shape", TensorProto.INT64, [4], [1, 1, 1, -1])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["reshape_shape"], value=reshape_shape_tensor))
    
    # 6. Reshape the 1D mask.
    nodes.append(helper.make_node("Reshape", inputs=["mask_1d", "reshape_shape"], outputs=["mask_reshaped"]))
    
    # 7. Multiply the MatMul output with the reshaped mask.
    nodes.append(helper.make_node("Mul", inputs=[matmul_output, "mask_reshaped"], outputs=[masked_output]))
    
    return nodes
def create_weight16_mask(matmul_output="y", masked_output="y_masked", block_length=4):
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
            dims=[],  # scalar
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
            dims=[],  # scalar
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
    nodes.append(helper.make_node(
        "Add",
        inputs=["start_idx" + suffix, "valid_block" + suffix],
        outputs=["end_idx" + suffix]
    ))
    
    # 10. Create boolean mask over the sequence dimension.
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
    
    # 11. Create a constant for reshape_shape with 4 dimensions.
    # Changing from [1, -1, 1] to [1, -1, 1, 1] so the mask becomes 4D.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["reshape_shape" + suffix],
        value=helper.make_tensor(
            name="reshape_shape_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[4],
            vals=[1, -1, 1, 1]
        )
    ))
    
    # 12. Reshape the 1D mask to a 4D tensor.
    nodes.append(helper.make_node(
        "Reshape",
        inputs=["bool_mask_1d" + suffix, "reshape_shape" + suffix],
        outputs=["bool_mask_4d" + suffix]
    ))
    
    # 13. Create a zeros tensor for the "false" branch of Where.
    # Change type to FLOAT to match matmul_output.
    nodes.append(helper.make_node(
        "ConstantOfShape",
        inputs=["y_shape" + suffix],
        outputs=["zeros" + suffix],
        value=helper.make_tensor(
            name="zeros_value" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[1],
            vals=[0.0]
        )
    ))
    
    # 14. Use Where to select between matmul_output and zeros based on the mask.
    nodes.append(helper.make_node(
        "Where",
        inputs=["bool_mask_4d" + suffix, matmul_output, "zeros" + suffix],
        outputs=[masked_output]
    ))
    
    return nodes
# Build a test graph for the mask.
# We'll use a dummy input "y" with shape [1, 8, 3, 3] (B=1, S=8, num_heads=3, head_dim=3)
input_tensor = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 3, 3])
output_tensor = helper.make_tensor_value_info("y_masked", TensorProto.FLOAT, [1, 8, 3, 3])

nodes = create_weight16_mask(matmul_output="y", masked_output="y_masked", block_length=2)

graph = helper.make_graph(
    nodes,
    "test_mask_graph",
    [input_tensor],
    [output_tensor]
)

model = helper.make_model(graph)

# Set the opset version to a supported one (e.g. 21)
model.opset_import[0].version = 21

onnx.save(model, "test_mask_model.onnx")
print("Model saved to 'test_mask_model.onnx'.")

# ---------------- Run the model using ONNX Runtime ----------------
# Create a dummy input array for "y"
y_input = np.random.rand(1, 8, 3, 3).astype(np.float32)
print("Input y:\n", y_input)

# Create an inference session
sess = ort.InferenceSession("test_mask_model.onnx")
outputs = sess.run(["y_masked"], {"y": y_input})
y_masked = outputs[0]
print("Output y_masked:\n", y_masked)
