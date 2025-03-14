from onnx import helper, TensorProto
import numpy as np
from typing import List
import onnx
from onnx import shape_inference
from onnx import save_model
from inject_utils.utils import delta_init, delta_init_int8 

from onnx import helper, TensorProto

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

def create_random_fault_injection(output_name: str, random_value: float):
    nodes = []
    
    # 1. Get the runtime shape of the tensor.
    nodes.append(helper.make_node(
        'Shape',
        inputs=[output_name],
        outputs=['runtime_shape']
    ))
    
    # 2. Generate a random index vector with shape [expected_rank] using RandomUniform.
    # For a 3D tensor the expected rank is 3.
    nodes.append(helper.make_node(
        'RandomUniform',
        inputs=[],  # shape is provided as attribute.
        outputs=['random_vals'],
        dtype=TensorProto.FLOAT,
        low=0.0,
        high=1.0,
        shape=[3]
    ))
    # Force the output to be a 1D tensor of shape [3] using Reshape.
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['const_shape'],
        value=helper.make_tensor("const_shape_tensor", TensorProto.INT64, [1], [3])
    ))
    nodes.append(helper.make_node(
        'Reshape',
        inputs=['random_vals', 'const_shape'],
        outputs=['random_vals_reshaped'],
        name="Reshape_random_vals"
    ))
    
    # 3. Cast the runtime shape (INT64) to FLOAT.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['runtime_shape'],
        outputs=['runtime_shape_float'],
        to=TensorProto.FLOAT
    ))
    # Reshape runtime_shape_float to a 1D tensor of length 3.
    nodes.append(helper.make_node(
        'Reshape',
        inputs=['runtime_shape_float', 'const_shape'],
        outputs=['runtime_shape_float_reshaped'],
        name="Reshape_runtime_shape_float"
    ))
    
    # 4. Multiply the reshaped random values by the reshaped runtime shape.
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals_reshaped', 'runtime_shape_float_reshaped'],
        outputs=['scaled_random']
    ))
    
    # 5. Floor the scaled random values.
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_random'],
        outputs=['floored_random']
    ))
    
    # 6. Cast the floored values to INT64 to get valid indices.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_random'],
        outputs=['random_indices_raw'],
        to=TensorProto.INT64
    ))
    
    # 7. Unsqueeze the random indices so that their shape becomes [1, 3] as required by ScatterND.
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['unsqueeze_axes'],
        value=helper.make_tensor("unsqueeze_axes_value", TensorProto.INT64, [1], [0])
    ))
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['random_indices_raw', 'unsqueeze_axes'],
        outputs=['random_indices']
    ))
    
    # 8. Create a constant node for the fault value.
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['fault_value'],
        value=helper.make_tensor(
            name='const_fault',
            data_type=TensorProto.FLOAT16,
            dims=[1],
            vals=[random_value]
        )
    ))
    
    # 9. Use ScatterND to inject the fault value at the generated index.
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=[output_name, 'random_indices', 'fault_value'],
        outputs=[f'{output_name}_faulty']
    ))
    
    return nodes

def create_random_bitflip_injection(output_name: str, bit_position: int):
    """Create random bitflip injection using Shape operator to handle dynamic dimensions"""
    nodes = []
    
    # 1. Get shape of tensor at runtime
    nodes.append(helper.make_node(
        'Shape',
        inputs=[output_name],
        outputs=['runtime_shape']
    ))
    
    # 2. Create random indices based on shape
    nodes.append(helper.make_node(
        'RandomUniformLike',
        inputs=['runtime_shape'],
        outputs=['indices'],
        dtype=TensorProto.INT64
    ))
    
    # 3. Gather the target value
    nodes.append(helper.make_node(
        "GatherND",
        inputs=[output_name, "indices"],
        outputs=["gathered_val"]
    ))
    
    # 4. Create bitmask for the bit to flip
    bitmask = 1 << bit_position
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["bitmask"],
        value=helper.make_tensor(
            name="const_bitmask",
            data_type=TensorProto.INT32,
            dims=[1],
            vals=[bitmask]
        )
    ))
    
    # 5. Cast float to int bits for manipulation
    nodes.append(helper.make_node(
        "Cast",
        inputs=["gathered_val"],
        outputs=["gathered_float"],
        to=TensorProto.FLOAT
    ))
    
    # 6. Reinterpret float as int32 for bit manipulation
    nodes.append(helper.make_node(
        "BitShift",
        inputs=["gathered_float"],
        outputs=["float_as_int"],
        direction="RIGHT",
        shift_amount=0
    ))
    
    # 7. Perform XOR to flip the bit
    nodes.append(helper.make_node(
        "BitwiseXor",
        inputs=["float_as_int", "bitmask"],
        outputs=["flipped_int"]
    ))
    
    # 8. Reinterpret int back as float
    nodes.append(helper.make_node(
        "BitShift",
        inputs=["flipped_int"],
        outputs=["int_as_float"],
        direction="LEFT",
        shift_amount=0
    ))
    
    # 9. Cast back to float32
    nodes.append(helper.make_node(
        "Cast",
        inputs=["int_as_float"],
        outputs=["flipped_float"],
        to=TensorProto.FLOAT
    ))
    
    # 10. Replace original value with flipped value using ScatterND
    nodes.append(helper.make_node(
        "ScatterND",
        inputs=[output_name, "indices", "flipped_float"],
        outputs=[f"{output_name}_faulty"],
        reduction="none"
    ))
    
    return nodes

def create_fp16_bit_flip(input_name, output_name, bit_position):
    nodes = []
    suffix = "_bf"
    
    sign_bit = 15
    exponent_start = 10
    exponent_bits = 5
    mantissa_bits = 10
    exponent_bias = 15
    
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero" + suffix],
        value=helper.make_tensor(
            name="zero_tensor" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[0.0]
        )
    ))
    
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["one" + suffix],
        value=helper.make_tensor(
            name="one_tensor" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[1.0]
        )
    ))
    
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["two" + suffix],
        value=helper.make_tensor(
            name="two_tensor" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[2.0]
        )
    ))
    
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["bit_pos" + suffix],
        value=helper.make_tensor(
            name="bit_pos_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],
            vals=[bit_position]
        )
    ))
    
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["sign_bit_pos" + suffix],
        value=helper.make_tensor(
            name="sign_bit_pos_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],
            vals=[sign_bit]
        )
    ))
    
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["exp_start" + suffix],
        value=helper.make_tensor(
            name="exp_start_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],
            vals=[exponent_start]
        )
    ))
    
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["exp_bias" + suffix],
        value=helper.make_tensor(
            name="exp_bias_tensor" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[float(exponent_bias)]
        )
    ))
    
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["mantissa_scale" + suffix],
        value=helper.make_tensor(
            name="mantissa_scale_tensor" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[1024.0]  # 2^10
        )
    ))
    
    # 3. Determine bit type
    # Is it sign bit?
    nodes.append(helper.make_node(
        "Equal",
        inputs=["bit_pos" + suffix, "sign_bit_pos" + suffix],
        outputs=["is_sign_bit" + suffix]
    ))
    
    # Is it exponent bit?
    nodes.append(helper.make_node(
        "Less",
        inputs=["bit_pos" + suffix, "sign_bit_pos" + suffix],
        outputs=["lt_sign" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "GreaterOrEqual",
        inputs=["bit_pos" + suffix, "exp_start" + suffix],
        outputs=["ge_exp_start" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "And",
        inputs=["lt_sign" + suffix, "ge_exp_start" + suffix],
        outputs=["is_exp_bit" + suffix]
    ))
    
    # Is it mantissa bit?
    nodes.append(helper.make_node(
        "Less",
        inputs=["bit_pos" + suffix, "exp_start" + suffix],
        outputs=["is_mantissa_bit" + suffix]
    ))
    
    # 4. SIGN BIT HANDLING
    # Simply negate the input
    nodes.append(helper.make_node(
        "Neg",
        inputs=[input_name],
        outputs=["sign_flipped" + suffix]
    ))
    
    # 5. EXPONENT BIT HANDLING
    # Get absolute value for log calculation
    nodes.append(helper.make_node(
        "Abs",
        inputs=[input_name],
        outputs=["abs_value" + suffix]
    ))
    
    # Handle zero values (log is undefined)
    nodes.append(helper.make_node(
        "Equal",
        inputs=["abs_value" + suffix, "zero" + suffix],
        outputs=["is_zero" + suffix]
    ))
    
    # For non-zero values, calculate log2
    nodes.append(helper.make_node(
        "Log",
        inputs=["abs_value" + suffix],
        outputs=["log_value" + suffix]
    ))
    
    # Convert to log2 (divide by ln(2))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["ln2" + suffix],
        value=helper.make_tensor(
            name="ln2_tensor" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[0.693147]  # ln(2)
        )
    ))
    
    nodes.append(helper.make_node(
        "Div",
        inputs=["log_value" + suffix, "ln2" + suffix],
        outputs=["log2_value" + suffix]
    ))
    
    # For zero input, use a dummy value that won't matter
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_zero" + suffix, "zero" + suffix, "log2_value" + suffix],
        outputs=["safe_log2" + suffix]
    ))
    
    # Calculate biased exponent
    nodes.append(helper.make_node(
        "Add",
        inputs=["safe_log2" + suffix, "exp_bias" + suffix],
        outputs=["biased_exp_float" + suffix]
    ))
    
    # Floor to get integer value
    nodes.append(helper.make_node(
        "Floor",
        inputs=["biased_exp_float" + suffix],
        outputs=["biased_exp" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Sub",
        inputs=["bit_pos" + suffix, "exp_start" + suffix],
        outputs=["exp_bit_pos" + suffix]
    ))
    
    # Convert to float for power operation
    nodes.append(helper.make_node(
        "Cast",
        inputs=["exp_bit_pos" + suffix],
        outputs=["exp_bit_pos_float" + suffix],
        to=TensorProto.FLOAT
    ))
    
    nodes.append(helper.make_node(
        "Pow",
        inputs=["two" + suffix, "exp_bit_pos_float" + suffix],
        outputs=["exp_bit_weight" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Cast",
        inputs=["exp_bit_weight" + suffix],
        outputs=["exp_divisor" + suffix],
        to=TensorProto.FLOAT
    ))
    
    # Divide exponent by weight
    nodes.append(helper.make_node(
        "Div",
        inputs=["biased_exp" + suffix, "exp_divisor" + suffix],
        outputs=["exp_div_result" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Floor",
        inputs=["exp_div_result" + suffix],
        outputs=["exp_div_floor" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Div",
        inputs=["exp_div_floor" + suffix, "two" + suffix],
        outputs=["exp_div_half" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Floor",
        inputs=["exp_div_half" + suffix],
        outputs=["exp_div_half_floor" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Mul",
        inputs=["exp_div_half_floor" + suffix, "two" + suffix],
        outputs=["exp_double_floor_half" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Sub",
        inputs=["exp_div_floor" + suffix, "exp_double_floor_half" + suffix],
        outputs=["exp_bit" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Equal",
        inputs=["exp_bit" + suffix, "one" + suffix],
        outputs=["exp_bit_is_one" + suffix]
    ))

    nodes.append(helper.make_node(
        "Pow",
        inputs=["two" + suffix, "exp_bit_weight" + suffix],
        outputs=["exp_scale_up" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Div",
        inputs=["one" + suffix, "exp_scale_up" + suffix],
        outputs=["exp_scale_down" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Where",
        inputs=["exp_bit_is_one" + suffix, "exp_scale_down" + suffix, "exp_scale_up" + suffix],
        outputs=["exp_scale" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Mul",
        inputs=[input_name, "exp_scale" + suffix],
        outputs=["exp_flipped" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Floor",
        inputs=["safe_log2" + suffix],
        outputs=["floor_log2" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Pow",
        inputs=["two" + suffix, "floor_log2" + suffix],
        outputs=["pow2_floor" + suffix]
    ))
    
    # Handle zero (avoid division by zero)
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_zero" + suffix, "one" + suffix, "pow2_floor" + suffix],
        outputs=["safe_pow2_floor" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Div",
        inputs=["abs_value" + suffix, "safe_pow2_floor" + suffix],
        outputs=["normalized_mantissa" + suffix]
    ))
    
    # For zero, set normalized mantissa to 0
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_zero" + suffix, "zero" + suffix, "normalized_mantissa" + suffix],
        outputs=["safe_normalized_mantissa" + suffix]
    ))
    
    # Extract fractional part (remove implicit leading 1)
    nodes.append(helper.make_node(
        "Sub",
        inputs=["safe_normalized_mantissa" + suffix, "one" + suffix],
        outputs=["mantissa_frac" + suffix]
    ))
    
    # Scale to integer range [0, 1024)
    nodes.append(helper.make_node(
        "Mul",
        inputs=["mantissa_frac" + suffix, "mantissa_scale" + suffix],
        outputs=["scaled_mantissa" + suffix]
    ))
    
    # Calculate mantissa bit position weight
    nodes.append(helper.make_node(
        "Cast",
        inputs=["bit_pos" + suffix],
        outputs=["bit_pos_float" + suffix],
        to=TensorProto.FLOAT
    ))
    
    nodes.append(helper.make_node(
        "Pow",
        inputs=["two" + suffix, "bit_pos_float" + suffix],
        outputs=["mantissa_bit_pow2" + suffix]
    ))
    
    # Divide mantissa by bit weight
    nodes.append(helper.make_node(
        "Div",
        inputs=["scaled_mantissa" + suffix, "mantissa_bit_pow2" + suffix],
        outputs=["mantissa_div_result" + suffix]
    ))
    
    # Floor the result
    nodes.append(helper.make_node(
        "Floor",
        inputs=["mantissa_div_result" + suffix],
        outputs=["mantissa_div_floor" + suffix]
    ))
    
    # Modulo 2 using same trick as before
    nodes.append(helper.make_node(
        "Div",
        inputs=["mantissa_div_floor" + suffix, "two" + suffix],
        outputs=["mantissa_div_half" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Floor",
        inputs=["mantissa_div_half" + suffix],
        outputs=["mantissa_div_half_floor" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Mul",
        inputs=["mantissa_div_half_floor" + suffix, "two" + suffix],
        outputs=["mantissa_double_floor_half" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Sub",
        inputs=["mantissa_div_floor" + suffix, "mantissa_double_floor_half" + suffix],
        outputs=["mantissa_bit" + suffix]
    ))
    
    # Check if bit is 1
    nodes.append(helper.make_node(
        "Equal",
        inputs=["mantissa_bit" + suffix, "one" + suffix],
        outputs=["mantissa_bit_is_one" + suffix]
    ))
    
    # Calculate delta for mantissa toggle
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["neg_ten" + suffix],
        value=helper.make_tensor(
            name="neg_ten_tensor" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[-10.0]
        )
    ))
    
    nodes.append(helper.make_node(
        "Add",
        inputs=["bit_pos_float" + suffix, "neg_ten" + suffix],
        outputs=["mantissa_power" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Pow",
        inputs=["two" + suffix, "mantissa_power" + suffix],
        outputs=["mantissa_delta_pos" + suffix]
    ))
    
    # Negate delta for 1→0 flip
    nodes.append(helper.make_node(
        "Neg",
        inputs=["mantissa_delta_pos" + suffix],
        outputs=["mantissa_delta_neg" + suffix]
    ))
    
    # Choose delta based on bit value
    nodes.append(helper.make_node(
        "Where",
        inputs=["mantissa_bit_is_one" + suffix, "mantissa_delta_neg" + suffix, "mantissa_delta_pos" + suffix],
        outputs=["mantissa_delta" + suffix]
    ))
    
    # Apply mantissa change
    nodes.append(helper.make_node(
        "Add",
        inputs=[input_name, "mantissa_delta" + suffix],
        outputs=["mantissa_flipped" + suffix]
    ))
    
    # 7. SELECT APPROPRIATE OUTPUT based on bit type
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_exp_bit" + suffix, "exp_flipped" + suffix, "mantissa_flipped" + suffix],
        outputs=["exp_or_mantissa" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_sign_bit" + suffix, "sign_flipped" + suffix, "exp_or_mantissa" + suffix],
        outputs=[output_name]
    ))
    
    return nodes
