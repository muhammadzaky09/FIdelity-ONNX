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
    """
    Create a mask that keeps only 'block_length' consecutive elements in the last dimension.
    Works with tensors of any rank, including 4D tensors with shape [batch, seq_len, heads, hidden].
    """
    nodes = []
    suffix = "_mask"
    
    # 1. Get the shape of the input tensor
    nodes.append(helper.make_node("Shape", inputs=[matmul_output], outputs=["y_shape" + suffix]))
    
    # 2. Get the rank (number of dimensions) of the tensor
    nodes.append(helper.make_node("Size", inputs=["y_shape" + suffix], outputs=["rank" + suffix]))
    
    # 3. Create constants for dimension indices
    # Last dimension will be the hidden dimension (in 4D case, index 3)
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_const" + suffix],
                                 value=helper.make_tensor("one_const" + suffix, TensorProto.INT64, [], [1])))
    
    # Calculate last dimension index (rank - 1)
    nodes.append(helper.make_node("Sub", inputs=["rank" + suffix, "one_const" + suffix], 
                                 outputs=["last_dim_idx" + suffix]))
    
    # 4. Extract the size of the last dimension (hidden dimension)
    nodes.append(helper.make_node("Gather", inputs=["y_shape" + suffix, "last_dim_idx" + suffix], 
                                 outputs=["H_value" + suffix], axis=0))
    
    # Squeeze to get scalar
    nodes.append(helper.make_node("Squeeze", inputs=["H_value" + suffix], outputs=["H_scalar" + suffix]))
    
    # 5. Calculate the block range for masking
    const_block = helper.make_tensor("block_length" + suffix, TensorProto.INT64, [], [block_length])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["block_length_const" + suffix], 
                                 value=const_block))
    
    # Compute max start index
    nodes.append(helper.make_node("Sub", inputs=["H_scalar" + suffix, "block_length_const" + suffix], 
                                 outputs=["H_minus_block" + suffix]))
    nodes.append(helper.make_node("Add", inputs=["H_minus_block" + suffix, "one_const" + suffix], 
                                 outputs=["range_size" + suffix]))
    
    # Generate random start index
    nodes.append(helper.make_node("Cast", inputs=["range_size" + suffix], 
                                 outputs=["range_size_float" + suffix], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("RandomUniform", inputs=[], outputs=["rand_val_temp" + suffix], 
                                 dtype=TensorProto.FLOAT, high=1.0, low=0.0, shape=[1]))
    nodes.append(helper.make_node("Squeeze", inputs=["rand_val_temp" + suffix], 
                                 outputs=["rand_val" + suffix]))
    nodes.append(helper.make_node("Mul", inputs=["rand_val" + suffix, "range_size_float" + suffix], 
                                 outputs=["rand_scaled" + suffix]))
    nodes.append(helper.make_node("Floor", inputs=["rand_scaled" + suffix], 
                                 outputs=["rand_index_float" + suffix]))
    nodes.append(helper.make_node("Cast", inputs=["rand_index_float" + suffix], 
                                 outputs=["start_index_dynamic" + suffix], to=TensorProto.INT64))
    nodes.append(helper.make_node("Add", inputs=["start_index_dynamic" + suffix, "block_length_const" + suffix], 
                                 outputs=["end_index_dynamic" + suffix]))
    
    # 6. Build 1D mask over the last dimension
    const_zero = helper.make_tensor("zero_const" + suffix, TensorProto.INT64, [], [0])
    const_one_step = helper.make_tensor("one_step" + suffix, TensorProto.INT64, [], [1])
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["zero_const_H" + suffix], value=const_zero))
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_const_H_step" + suffix], value=const_one_step))
    nodes.append(helper.make_node("Range", inputs=["zero_const_H" + suffix, "H_scalar" + suffix, "one_const_H_step" + suffix], 
                                 outputs=["indices_H" + suffix]))
    nodes.append(helper.make_node("GreaterOrEqual", inputs=["indices_H" + suffix, "start_index_dynamic" + suffix], 
                                 outputs=["ge_mask_H" + suffix]))
    nodes.append(helper.make_node("Less", inputs=["indices_H" + suffix, "end_index_dynamic" + suffix], 
                                 outputs=["lt_mask_H" + suffix]))
    nodes.append(helper.make_node("And", inputs=["ge_mask_H" + suffix, "lt_mask_H" + suffix], 
                                 outputs=["mask_bool_H" + suffix]))
    nodes.append(helper.make_node("Cast", inputs=["mask_bool_H" + suffix], outputs=["mask_1d" + suffix], 
                                 to=TensorProto.FLOAT16))
    
    # 7. Create a dynamic shape for reshaping the mask
    # For a 4D tensor [B,S,H,W], we need to unsqueeze to shape [1,1,1,W]
    # Generate an array of all dimensions except the last one
    nodes.append(helper.make_node("Range", 
                                 inputs=["zero_const_H" + suffix, "last_dim_idx" + suffix, "one_const_H_step" + suffix], 
                                 outputs=["unsqueeze_dims" + suffix]))
    
    # 8. Unsqueeze the 1D mask to add dimensions for all but the last dimension
    nodes.append(helper.make_node("Unsqueeze", 
                                 inputs=["mask_1d" + suffix, "unsqueeze_dims" + suffix], 
                                 outputs=["mask_expanded" + suffix]))
    
    # 9. Get the shape of the input tensor for expanding
    nodes.append(helper.make_node("Shape", inputs=[matmul_output], outputs=["full_shape" + suffix]))
    
    # 10. Expand the mask to match the shape of the input tensor
    nodes.append(helper.make_node("Expand", 
                                 inputs=["mask_expanded" + suffix, "full_shape" + suffix], 
                                 outputs=["mask_full" + suffix]))
    
    # 11. Apply the mask via element-wise multiplication
    nodes.append(helper.make_node("Mul", 
                                 inputs=[matmul_output, "mask_full" + suffix], 
                                 outputs=[masked_output]))
    
    return nodes

###############################################################################
# Build Complete Model: Injection -> MatMul -> Masking (Post-MatMul)
###############################################################################
def build_input_matmul_model_with_mask(input_shape, weight_shape=None,
                                      input_dtype=np.float16, bit_position=3, block_length=16):
    input_name = "x"
    faulty_input_name = "x_faulty"
    matmul_output = "y"
    masked_output = "y_masked"
    weight_name = "W"
    
    # Get the last dimension of input for matmul weight shape
    last_dim = input_shape[-1]
    
    # If weight_shape is not specified, create a square matrix for the last dimension
    if weight_shape is None:
        weight_shape = (last_dim, last_dim)
    
    # Create input tensor with dynamic shape (only specify the data type)
    input_tensor_info = helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, list(input_shape))
    
    # Create fault injection nodes
    injection_nodes = create_quantized_fault_injection(input_name, faulty_input_name, bit_position)
    
    # Create weight tensor
    weight_value = np.random.rand(*weight_shape).astype(input_dtype)
    weight_node = helper.make_node("Constant", inputs=[], outputs=[weight_name],
                                   value=helper.make_tensor("weight_tensor", TensorProto.FLOAT16, 
                                                           dims=weight_shape,
                                                           vals=weight_value.flatten().tolist()))
    
    # Create matmul node
    matmul_node = helper.make_node("MatMul", inputs=[faulty_input_name, weight_name], 
                                  outputs=[matmul_output], name="MatMul_fault_injected")
    
    # Create masking nodes
    mask_nodes = create_input16_mask(matmul_output, masked_output, block_length)
    
    # Combine all nodes
    all_nodes = injection_nodes + [weight_node, matmul_node] + mask_nodes
    
    # Calculate output shape - For MatMul, the output shape replaces the last dimension
    # of the input with the last dimension of the weight
    output_shape = list(input_shape)
    output_shape[-1] = weight_shape[-1]
    
    # Define output tensor
    output_tensor_info = helper.make_tensor_value_info(masked_output, TensorProto.FLOAT16, output_shape)
    
    # Create graph and model
    graph = helper.make_graph(nodes=all_nodes,
                              name="InputFaultInjectionMatMulMaskGraph",
                              inputs=[input_tensor_info],
                              outputs=[output_tensor_info])
    
    model = helper.make_model(graph, producer_name="FaultInjectionInputMaskTest")
    onnx.checker.check_model(model)
    model.opset_import[0].version = 18
    return model

def run_model(model):
    # Create 4D input with shape taken from the model
    input_info = model.graph.input[0]
    input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
    
    # Generate random input
    x = np.random.randn(*input_shape).astype(np.float16)
    
    # Run inference
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"x": x})[0]
    
    return x, y

if __name__ == "__main__":
    # Test with 4D input
    size = 16
    model = build_input_matmul_model_with_mask(
        input_shape=(1, 40, 25, 32),  # 4D input
        weight_shape=(1,1,32, 32),        # Last dim to last dim
        input_dtype=np.float16, 
        bit_position=3, 
        block_length=size
    )
    
    # The shape should now be preserved through the entire operation
    x, y = run_model(model)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Non-zero elements: {np.count_nonzero(y)} out of {y.size}")