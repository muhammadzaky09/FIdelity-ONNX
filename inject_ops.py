from onnx import helper, TensorProto
import numpy as np
from typing import List





def create_quantized_fault_injection_weight(input_name, output_name, bit_position):
    nodes = []
    suffix = "_w"

    # 1. Promote weight from FLOAT16 to FLOAT32.
    nodes.append(helper.make_node(
        'Cast',
        inputs=[input_name],
        outputs=['weight_fp32' + suffix],
        to=TensorProto.FLOAT
    ))

    # 2. Get dynamic shape of the FP32 weight.
    nodes.append(helper.make_node(
        'Shape',
        inputs=['weight_fp32' + suffix],
        outputs=['runtime_shape' + suffix]
    ))

    # 3. Cast the shape to FLOAT.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['runtime_shape' + suffix],
        outputs=['runtime_shape_float' + suffix],
        to=TensorProto.FLOAT
    ))

    # 4. Generate random uniform values with shape [2] (for a 2D weight).
    nodes.append(helper.make_node(
        'RandomUniform',
        inputs=[],
        outputs=['random_vals' + suffix],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0,
        shape=[2]
    ))

    # 5. Multiply the random values by the shape.
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals' + suffix, 'runtime_shape_float' + suffix],
        outputs=['scaled_indices' + suffix]
    ))

    # 6. Floor the scaled indices.
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_indices' + suffix],
        outputs=['floored_indices' + suffix]
    ))

    # 7. Cast the floored indices to INT64.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_indices' + suffix],
        outputs=['indices_int64' + suffix],
        to=TensorProto.INT64
    ))

    # 8. Create a Constant for Unsqueeze axes (for opsets >= 13).
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['unsqueeze_axes' + suffix],
        value=helper.make_tensor(
            name='unsqueeze_axes_tensor' + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))

    # 9. Unsqueeze the indices to shape [1,2] for ScatterND.
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['indices_int64' + suffix, 'unsqueeze_axes' + suffix],
        outputs=['indices_int64_unsqueezed' + suffix]
    ))

    # 10. Cast the FP32 weight to INT8.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['weight_fp32' + suffix],
        outputs=['int8_val' + suffix],
        to=TensorProto.INT8
    ))

    # 11. Create a constant bitmask (scalar) for the desired bit, but with shape [1].
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['bitmask' + suffix],
        value=helper.make_tensor(
            name='bitmask_val' + suffix,
            data_type=TensorProto.INT8,
            dims=[1],            # <-- Set dims to [1] so that updates shape is [1]
            vals=[1 << bit_position]
        )
    ))

    # 12. Create a zero tensor of the same shape using ConstantOfShape.
    nodes.append(helper.make_node(
        'ConstantOfShape',
        inputs=['runtime_shape' + suffix],
        outputs=['zero_base' + suffix],
        value=helper.make_tensor(
            name='zero_value' + suffix,
            data_type=TensorProto.INT8,
            dims=[1],
            vals=[0]
        )
    ))

    # 13. Scatter the bitmask into the zero tensor at the computed indices.
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=['zero_base' + suffix, 'indices_int64_unsqueezed' + suffix, 'bitmask' + suffix],
        outputs=['bit_mask' + suffix]
    ))

    # 14. Apply BitwiseXor between the int8 weight and the bit mask.
    nodes.append(helper.make_node(
        'BitwiseXor',
        inputs=['int8_val' + suffix, 'bit_mask' + suffix],
        outputs=['flipped_int' + suffix]
    ))

    # 15. Cast both the flipped tensor and the original int8 tensor to INT32.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['flipped_int' + suffix],
        outputs=['flipped_int32' + suffix],
        to=TensorProto.INT32
    ))
    nodes.append(helper.make_node(
        'Cast',
        inputs=['int8_val' + suffix],
        outputs=['int8_val32' + suffix],
        to=TensorProto.INT32
    ))

    # 16. Subtract: flipped_int32 - int8_val32 = perturbation (INT32)
    nodes.append(helper.make_node(
        'Sub',
        inputs=['flipped_int32' + suffix, 'int8_val32' + suffix],
        outputs=['perturbation_int32' + suffix]
    ))

    # 17. Cast the perturbation to FLOAT32.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['perturbation_int32' + suffix],
        outputs=['perturbation_fp32' + suffix],
        to=TensorProto.FLOAT
    ))

    # 18. Finally, cast the perturbation to FLOAT16 (to match original precision).
    nodes.append(helper.make_node(
        'Cast',
        inputs=['perturbation_fp32' + suffix],
        outputs=[output_name],
        to=TensorProto.FLOAT16
    ))

    return nodes
def create_weight16_mask(matmul_output="y", masked_output="y_masked", block_length=4):
    """
    Create a mask that keeps only 'block_length' consecutive rows in the sequence dimension.
    Fixed to properly handle broadcasting with 3D tensors of shape [batch, sequence, hidden].
    """
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
    
    # 10. Calculate end index
    nodes.append(helper.make_node(
        "Add",
        inputs=["start_idx" + suffix, "valid_block" + suffix],
        outputs=["end_idx" + suffix]
    ))
    
    # 11. Create boolean mask
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
    
    # 12. Create shape for reshaping the mask to 3D
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["reshape_shape" + suffix],
        value=helper.make_tensor(
            name="reshape_shape_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[3],
            vals=[1, -1, 1]  # [1, seq_len, 1] with second dim inferred
        )
    ))
    
    # 13. Reshape the boolean mask to 3D for proper broadcasting
    nodes.append(helper.make_node(
        "Reshape",
        inputs=["bool_mask_1d" + suffix, "reshape_shape" + suffix],
        outputs=["bool_mask_3d" + suffix]
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
        inputs=["bool_mask_3d" + suffix, matmul_output, "zeros" + suffix],
        outputs=[masked_output]
    ))
    
    return nodes


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
        outputs=['perturbation_fp32'],
        to=TensorProto.FLOAT
    ))
    # 18. Finally, cast the perturbation to FLOAT16 (to match original precision).
    nodes.append(helper.make_node(
        'Cast',
        inputs=['perturbation_fp32'],
        outputs=[output_name],
        to=TensorProto.FLOAT16
    ))
    return nodes

# def create_input16_mask(matmul_output="y", masked_output="y_masked", block_length=16):

#     nodes = []
#     # 1. Get the shape of the MatMul output.
#     nodes.append(helper.make_node("Shape", inputs=[matmul_output], outputs=["y_shape"]))
    
#     # 2. Extract hidden dimension H from y_shape (assume y_shape = [B,S,H]).
#     const_H_start = helper.make_tensor("H_start", TensorProto.INT64, [1], [2])
#     const_H_end   = helper.make_tensor("H_end",   TensorProto.INT64, [1], [3])
#     const_H_axes  = helper.make_tensor("H_axes",  TensorProto.INT64, [1], [0])
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_starts"], value=const_H_start))
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_ends"],   value=const_H_end))
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_axes"],   value=const_H_axes))
#     nodes.append(helper.make_node("Slice", inputs=["y_shape", "H_starts", "H_ends", "H_axes"], outputs=["H_value"]))
#     # Squeeze to get scalar H. (Omit axes so all 1-dim are removed.)
#     nodes.append(helper.make_node("Squeeze", inputs=["H_value"], outputs=["H_scalar"]))
    
#     # 3. Compute dynamic start index along H.
#     const_block = helper.make_tensor("block_length", TensorProto.INT64, [], [block_length])
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["block_length_const"], value=const_block))
#     nodes.append(helper.make_node("Sub", inputs=["H_scalar", "block_length_const"], outputs=["H_minus_block"]))
#     const_one = helper.make_tensor("one_const", TensorProto.INT64, [], [1])
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_const"], value=const_one))
#     nodes.append(helper.make_node("Add", inputs=["H_minus_block", "one_const"], outputs=["range_size"]))
#     nodes.append(helper.make_node("Cast", inputs=["range_size"], outputs=["range_size_float"], to=TensorProto.FLOAT))
#     nodes.append(helper.make_node("RandomUniform", inputs=[], outputs=["rand_val_temp"], dtype=TensorProto.FLOAT, high=1.0, low=0.0, shape=[1]))
#     nodes.append(helper.make_node("Squeeze", inputs=["rand_val_temp"], outputs=["rand_val"]))
#     nodes.append(helper.make_node("Mul", inputs=["rand_val", "range_size_float"], outputs=["rand_scaled"]))
#     nodes.append(helper.make_node("Floor", inputs=["rand_scaled"], outputs=["rand_index_float"]))
#     nodes.append(helper.make_node("Cast", inputs=["rand_index_float"], outputs=["start_index_dynamic"], to=TensorProto.INT64))
#     nodes.append(helper.make_node("Add", inputs=["start_index_dynamic", "block_length_const"], outputs=["end_index_dynamic"]))
    
#     # 4. Build 1D mask over H.
#     const_zero = helper.make_tensor("zero_const", TensorProto.INT64, [], [0])
#     const_one_step = helper.make_tensor("one_step", TensorProto.INT64, [], [1])
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["zero_const_H"], value=const_zero))
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_const_H_step"], value=const_one_step))
#     nodes.append(helper.make_node("Range", inputs=["zero_const_H", "H_scalar", "one_const_H_step"], outputs=["indices_H"]))
#     nodes.append(helper.make_node("GreaterOrEqual", inputs=["indices_H", "start_index_dynamic"], outputs=["ge_mask_H"]))
#     nodes.append(helper.make_node("Less", inputs=["indices_H", "end_index_dynamic"], outputs=["lt_mask_H"]))
#     nodes.append(helper.make_node("And", inputs=["ge_mask_H", "lt_mask_H"], outputs=["mask_bool_H"]))
#     nodes.append(helper.make_node("Cast", inputs=["mask_bool_H"], outputs=["mask_1d"], to=TensorProto.FLOAT16))
    
#     # 5. Unsqueeze mask_1d to shape [1,1,H]. Use a constant axes tensor [0,1].
#     const_unsqueeze_axes = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [2], [0,1])
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["unsqueeze_axes"], value=const_unsqueeze_axes))
#     nodes.append(helper.make_node("Unsqueeze", inputs=["mask_1d", "unsqueeze_axes"], outputs=["mask_unsqueezed"]))
    
#     # 6. Tile mask_unsqueezed to shape [B,S,H]. Extract B and S from y_shape.
#     const_B_start = helper.make_tensor("B_start", TensorProto.INT64, [1], [0])
#     const_B_end   = helper.make_tensor("B_end",   TensorProto.INT64, [1], [1])
#     const_B_axes  = helper.make_tensor("B_axes",  TensorProto.INT64, [1], [0])
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["B_starts_out"], value=const_B_start))
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["B_ends_out"], value=const_B_end))
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["B_axes_out"], value=const_B_axes))
#     nodes.append(helper.make_node("Slice", inputs=["y_shape", "B_starts_out", "B_ends_out", "B_axes_out"], outputs=["B_value_out"]))
#     nodes.append(helper.make_node("Squeeze", inputs=["B_value_out"], outputs=["B_scalar_out"]))
    
#     const_S_start = helper.make_tensor("S_start", TensorProto.INT64, [1], [1])
#     const_S_end   = helper.make_tensor("S_end",   TensorProto.INT64, [1], [2])
#     const_S_axes  = helper.make_tensor("S_axes",  TensorProto.INT64, [1], [0])
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["S_starts_out"], value=const_S_start))
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["S_ends_out"], value=const_S_end))
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["S_axes_out"], value=const_S_axes))
#     nodes.append(helper.make_node("Slice", inputs=["y_shape", "S_starts_out", "S_ends_out", "S_axes_out"], outputs=["S_value_out"]))
#     nodes.append(helper.make_node("Squeeze", inputs=["S_value_out"], outputs=["S_scalar_out"]))
    
#     # Convert B_scalar_out and S_scalar_out (scalars) to 1D tensors.
#     const_unsqueeze_axis0 = helper.make_tensor("unsqueeze_axis0", TensorProto.INT64, [1], [0])
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["unsqueeze_axis0"], value=const_unsqueeze_axis0))
#     nodes.append(helper.make_node("Unsqueeze", inputs=["B_scalar_out", "unsqueeze_axis0"], outputs=["B_1d"]))
#     nodes.append(helper.make_node("Unsqueeze", inputs=["S_scalar_out", "unsqueeze_axis0"], outputs=["S_1d"]))
    
#     const_one_tile = helper.make_tensor("one_tile", TensorProto.INT64, [1], [1])
#     nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_for_tile"], value=const_one_tile))
#     nodes.append(helper.make_node("Concat", inputs=["B_1d", "S_1d", "one_for_tile"], outputs=["tile_multiples"], axis=0))
#     nodes.append(helper.make_node("Tile", inputs=["mask_unsqueezed", "tile_multiples"], outputs=["mask_full"]))
    
#     # 7. Multiply the MatMul output with the mask.
#     nodes.append(helper.make_node("Mul", inputs=[matmul_output, "mask_full"], outputs=[masked_output]))
    
#     return nodes

def create_input16_mask(matmul_output="y", masked_output="y_masked", block_length=16):
    nodes = []
    suffix = "_mask"
    
    # 1. Get the shape of the MatMul output
    nodes.append(helper.make_node("Shape", inputs=[matmul_output], outputs=["y_shape" + suffix]))
    
    # 2. Get the rank (number of dimensions) of the tensor
    nodes.append(helper.make_node("Size", inputs=["y_shape" + suffix], outputs=["rank" + suffix]))
    
    # 3. Create constants for dimension indices
    # Last dimension will be the H dimension (in 3D case) or equivalent in higher dimensions
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["one_const" + suffix],
                                 value=helper.make_tensor("one_const" + suffix, TensorProto.INT64, [], [1])))
    
    # Calculate last dimension index (rank - 1)
    nodes.append(helper.make_node("Sub", inputs=["rank" + suffix, "one_const" + suffix], 
                                 outputs=["last_dim_idx" + suffix]))
    
    # For getting the second-last dimension (S in 3D case)
    nodes.append(helper.make_node("Sub", inputs=["last_dim_idx" + suffix, "one_const" + suffix], 
                                 outputs=["second_last_dim_idx" + suffix]))
    
    # 4. Extract the size of the hidden dimension (last dimension)
    # Create constants for Slice operation
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["H_starts" + suffix],
                                 value=helper.make_tensor("H_starts" + suffix, TensorProto.INT64, [1], 
                                                        [0])))  # Will use Gather instead
    
    # Get H dimension size using Gather
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
    
    # 6. Build 1D mask over H dimension
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
    # Generate unsqueeze axes for all dimensions except the last
    nodes.append(helper.make_node("Constant", inputs=[], outputs=["unsqueeze_dims" + suffix],
                                 value=helper.make_tensor("unsqueeze_dims" + suffix, TensorProto.INT64, 
                                                        [1], [0])))

    # Get shape as a vector with 1s for all dimensions except the last (which will be H)
    # First create shape of 1s with same rank as input
    nodes.append(helper.make_node("ConstantOfShape", inputs=["rank" + suffix], 
                                 outputs=["ones_shape" + suffix],
                                 value=helper.make_tensor("ones" + suffix, TensorProto.INT64, [1], [1])))
    
    # Replace last dimension with -1 (to be inferred)
    nodes.append(helper.make_node("Gather", inputs=["y_shape" + suffix, "last_dim_idx" + suffix], 
                                 outputs=["last_dim_size" + suffix], axis=0))
    
    # Use Unsqueeze to expand dimensions
    nodes.append(helper.make_node("Unsqueeze", inputs=["mask_1d" + suffix, "unsqueeze_dims" + suffix], 
                                 outputs=["mask_expanded" + suffix]))
    
    # 8. Get the actual shape of the input tensor to build a shape tensor for broadcasting
    nodes.append(helper.make_node("Shape", inputs=[matmul_output], outputs=["full_shape" + suffix]))
    
    # 9. Create broadcast mask using Expand rather than Tile
    nodes.append(helper.make_node("Expand", inputs=["mask_expanded" + suffix, "full_shape" + suffix], 
                                 outputs=["mask_full" + suffix]))
    
    # 10. Multiply the MatMul output with the mask
    nodes.append(helper.make_node("Mul", inputs=[matmul_output, "mask_full" + suffix], 
                                 outputs=[masked_output]))
    
    return nodes

def create_random_fault_injection(output_name: str, random_value: float):
    nodes = []
    suffix = "_random"
    
    # 1. Get the runtime shape of the tensor
    nodes.append(helper.make_node(
        'Shape',
        inputs=[output_name],
        outputs=['runtime_shape' + suffix]
    ))
    
    # 2. Cast runtime shape to FLOAT
    nodes.append(helper.make_node(
        'Cast',
        inputs=['runtime_shape' + suffix],
        outputs=['runtime_shape_float' + suffix],
        to=TensorProto.FLOAT
    ))
    
    # 3. Generate random values with the same shape as runtime_shape
    nodes.append(helper.make_node(
        'RandomUniformLike',
        inputs=['runtime_shape' + suffix],
        outputs=['random_vals' + suffix],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0
    ))
    
    # 4. Multiply random values by shape dimensions
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals' + suffix, 'runtime_shape_float' + suffix],
        outputs=['scaled_indices' + suffix]
    ))
    
    # 5. Floor the scaled indices
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_indices' + suffix],
        outputs=['floored_indices' + suffix]
    ))
    
    # 6. Cast to INT64
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_indices' + suffix],
        outputs=['indices_int64' + suffix],
        to=TensorProto.INT64
    ))
    
    # 7. Unsqueeze the indices for ScatterND
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['unsqueeze_axes' + suffix],
        value=helper.make_tensor(
            name='unsqueeze_axes_tensor' + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['indices_int64' + suffix, 'unsqueeze_axes' + suffix],
        outputs=['indices_unsqueezed' + suffix]
    ))
    
    # 8. Create a constant for the fault value (FP16)
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['fault_value' + suffix],
        value=helper.make_tensor(
            name='fault_value_tensor' + suffix,
            data_type=TensorProto.FLOAT16,
            dims=[1],
            vals=[random_value]
        )
    ))
    
    # 9. Use ScatterND to inject the fault
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=[output_name, 'indices_unsqueezed' + suffix, 'fault_value' + suffix],
        outputs=[f'{output_name}_faulty']
    ))
    
    return nodes


def create_random_bitflip_injection(output_name: str, bit_position: int):
    nodes = []
    suffix = "_fp16"
    faulty_output = f"{output_name}_faulty"
    
    # 1. Get the runtime shape of the input tensor
    nodes.append(helper.make_node(
        'Shape',
        inputs=[output_name],
        outputs=['runtime_shape' + suffix]
    ))
    
    # 2. Cast runtime shape to FLOAT
    nodes.append(helper.make_node(
        'Cast',
        inputs=['runtime_shape' + suffix],
        outputs=['runtime_shape_float' + suffix],
        to=TensorProto.FLOAT
    ))
    
    # 3. Generate random values with the same shape as runtime_shape
    nodes.append(helper.make_node(
        'RandomUniformLike',
        inputs=['runtime_shape' + suffix],
        outputs=['random_vals' + suffix],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0
    ))
    
    # 4. Multiply random values by shape dimensions
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals' + suffix, 'runtime_shape_float' + suffix],
        outputs=['scaled_indices' + suffix]
    ))
    
    # 5. Floor the scaled indices
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_indices' + suffix],
        outputs=['floored_indices' + suffix]
    ))
    
    # 6. Cast to INT64
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_indices' + suffix],
        outputs=['indices_int64' + suffix],
        to=TensorProto.INT64
    ))
    
    # 7. Unsqueeze the indices for ScatterND
    unsqueeze_axes = helper.make_tensor(
        name="unsqueeze_axes_tensor" + suffix,
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0]
    )
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['unsqueeze_axes' + suffix],
        value=unsqueeze_axes
    ))
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['indices_int64' + suffix, 'unsqueeze_axes' + suffix],
        outputs=['indices_unsqueezed' + suffix]
    ))
    
    # 8. Create a zero tensor with the same shape as input
    nodes.append(helper.make_node(
        'ConstantOfShape',
        inputs=['runtime_shape' + suffix],
        outputs=['zero_tensor' + suffix],
        value=helper.make_tensor(
            name='zero_tensor_val' + suffix,
            data_type=TensorProto.FLOAT16,
            dims=[1],
            vals=[0]
        )
    ))
    
    # 9. Create a constant one (FP16) to scatter
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['one_scalar' + suffix],
        value=helper.make_tensor(
            name='one_tensor' + suffix,
            data_type=TensorProto.FLOAT16,
            dims=[1],
            vals=[1]
        )
    ))
    
    # 10. Use ScatterND to create a one-hot mask
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=['zero_tensor' + suffix, 'indices_unsqueezed' + suffix, 'one_scalar' + suffix],
        outputs=['one_hot_mask' + suffix]
    ))
    
    # 11. Create a constant for the bit position
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['bit_pos_const' + suffix],
        value=helper.make_tensor(
            name='bit_pos_tensor' + suffix,
            data_type=TensorProto.INT32,
            dims=[1],
            vals=[bit_position]
        )
    ))
    
    # 12. Call the custom FP16 BitFlip operator
    nodes.append(helper.make_node(
        'BitFlip',
        inputs=[output_name, 'bit_pos_const' + suffix],
        outputs=['flipped_tensor' + suffix],
        domain='custom.bitflip'
    ))
    
    # 13. Compute the difference
    nodes.append(helper.make_node(
        'Sub',
        inputs=['flipped_tensor' + suffix, output_name],
        outputs=['difference' + suffix]
    ))
    
    # 14. Apply the mask
    nodes.append(helper.make_node(
        'Mul',
        inputs=['difference' + suffix, 'one_hot_mask' + suffix],
        outputs=['perturbation' + suffix]
    ))
    
    # 15. Add the perturbation back
    nodes.append(helper.make_node(
        'Add',
        inputs=[output_name, 'perturbation' + suffix],
        outputs=[faulty_output]
    ))
    
    return nodes

# def create_fp16_fault_injection_weight(input_name, output_name, bit_position):
#     nodes = []
#     suffix = "_fp16"
    
#     # 1. Get the shape of the input weight
#     nodes.append(helper.make_node(
#         'Shape',
#         inputs=[input_name],
#         outputs=['runtime_shape' + suffix]
#     ))
    
#     # 2. Cast the shape to FLOAT
#     nodes.append(helper.make_node(
#         'Cast',
#         inputs=['runtime_shape' + suffix],
#         outputs=['runtime_shape_float' + suffix],
#         to=TensorProto.FLOAT
#     ))
    
#     # 3. Generate random values with the same shape as runtime_shape
#     nodes.append(helper.make_node(
#         'RandomUniformLike',
#         inputs=['runtime_shape' + suffix],
#         outputs=['random_vals' + suffix],
#         dtype=TensorProto.FLOAT,
#         high=1.0,
#         low=0.0
#     ))
    
#     # 4. Multiply random values by shape dimensions to get indices
#     nodes.append(helper.make_node(
#         'Mul',
#         inputs=['random_vals' + suffix, 'runtime_shape_float' + suffix],
#         outputs=['scaled_indices' + suffix]
#     ))
    
#     # 5. Floor the scaled indices
#     nodes.append(helper.make_node(
#         'Floor',
#         inputs=['scaled_indices' + suffix],
#         outputs=['floored_indices' + suffix]
#     ))
    
#     # 6. Cast to INT64
#     nodes.append(helper.make_node(
#         'Cast',
#         inputs=['floored_indices' + suffix],
#         outputs=['indices_int64' + suffix],
#         to=TensorProto.INT64
#     ))
    
#     # 7. Unsqueeze the indices for ScatterND
#     unsqueeze_axes = helper.make_tensor(
#         name="unsqueeze_axes_tensor" + suffix,
#         data_type=TensorProto.INT64,
#         dims=[1],
#         vals=[0]
#     )
#     nodes.append(helper.make_node(
#         'Constant',
#         inputs=[],
#         outputs=['unsqueeze_axes' + suffix],
#         value=unsqueeze_axes
#     ))
#     nodes.append(helper.make_node(
#         'Unsqueeze',
#         inputs=['indices_int64' + suffix, 'unsqueeze_axes' + suffix],
#         outputs=['indices_unsqueezed' + suffix]
#     ))
    
#     # 8. Create a zero tensor with the same shape as the input
#     nodes.append(helper.make_node(
#         'ConstantOfShape',
#         inputs=['runtime_shape' + suffix],
#         outputs=['zero_tensor' + suffix],
#         value=helper.make_tensor(
#             name='zero_value_tensor' + suffix,
#             data_type=TensorProto.FLOAT16,
#             dims=[1],
#             vals=[0]
#         )
#     ))
    
#     # 9. Create a constant one (FP16) for scattering
#     nodes.append(helper.make_node(
#         'Constant',
#         inputs=[],
#         outputs=['one_scalar' + suffix],
#         value=helper.make_tensor(
#             name='one_tensor' + suffix,
#             data_type=TensorProto.FLOAT16,
#             dims=[1],
#             vals=[1]
#         )
#     ))
    
#     # 10. ScatterND: Create one-hot mask
#     nodes.append(helper.make_node(
#         'ScatterND',
#         inputs=['zero_tensor' + suffix, 'indices_unsqueezed' + suffix, 'one_scalar' + suffix],
#         outputs=['one_hot_mask' + suffix]
#     ))
    
#     # 11. Create bit position constant
#     nodes.append(helper.make_node(
#         'Constant',
#         inputs=[],
#         outputs=['bit_pos_const' + suffix],
#         value=helper.make_tensor(
#             name='bit_pos_tensor' + suffix,
#             data_type=TensorProto.INT32,
#             dims=[1],
#             vals=[bit_position]
#         )
#     ))
    
#     # 12. Apply BitFlip to the weight 
#     nodes.append(helper.make_node(
#         'BitFlip',
#         inputs=[input_name, 'bit_pos_const' + suffix],
#         outputs=['flipped_weight' + suffix],
#         domain='custom.bitflip'
#     ))
    
#     # 13. Calculate difference between flipped and original
#     nodes.append(helper.make_node(
#         'Sub',
#         inputs=['flipped_weight' + suffix, input_name],
#         outputs=['difference' + suffix]
#     ))
    
#     # 14. Apply mask to isolate perturbation to one element
#     nodes.append(helper.make_node(
#         'Mul',
#         inputs=['difference' + suffix, 'one_hot_mask' + suffix],
#         outputs=['perturbation' + suffix]
#     ))
    
#     # 15. Return the perturbation
#     nodes.append(helper.make_node(
#         'Identity',
#         inputs=['perturbation' + suffix],
#         outputs=[output_name]
#     ))
    
#     return nodes



def create_fp16_fault_injection(input_name, output_name, bit_position):
    nodes = []
    suffix = ""  # No suffix needed for the integrated operator

    # 1. Create a constant node for the bit position.
    bit_pos_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['bit_pos_const' + suffix],
        value=helper.make_tensor(
            name='bit_pos_tensor' + suffix,
            data_type=TensorProto.INT32,
            dims=[1],
            vals=[bit_position]
        )
    )
    nodes.append(bit_pos_node)

    # 2. Create the perturb node that does everything internally.
    perturb_node = helper.make_node(
        'perturb',  # This is your custom operator's name (GetName() returns "perturb")
        inputs=[input_name, 'bit_pos_const' + suffix],
        outputs=[output_name],
        domain='custom.perturb'
    )
    nodes.append(perturb_node)

    return nodes

def create_fp16_fault_injection_weight(input_name, output_name, bit_position):
    nodes = []
    suffix = ""  # No suffix needed for the integrated operator

    # 1. Create a constant node for the bit position.
    bit_pos_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['bit_pos_const' + suffix],
        value=helper.make_tensor(
            name='bit_pos_tensor' + suffix,
            data_type=TensorProto.INT32,
            dims=[1],
            vals=[bit_position]
        )
    )
    nodes.append(bit_pos_node)

    # 2. Create the perturb node that does everything internally.
    perturb_node = helper.make_node(
        'Perturb',  # This is your custom operator's name (GetName() returns "perturb")
        inputs=[input_name, 'bit_pos_const' + suffix],
        outputs=[output_name],
        domain='custom.perturb'
    )
    nodes.append(perturb_node)

    return nodes
