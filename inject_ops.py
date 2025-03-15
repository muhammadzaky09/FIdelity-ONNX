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
    """
    Create ONNX nodes that implement IEEE-754 FP16 bit flipping at a specified position.
    Improved mathematical approach that correctly handles all bit positions.
    
    Args:
        input_name: Name of the input tensor (FLOAT)
        output_name: Name of the output tensor (FLOAT)
        bit_position: Position of bit to flip (0-15, where 0 is LSB of mantissa)
        
    Returns:
        List of ONNX nodes implementing the bit flip operation
    """
    nodes = []
    suffix = "_bf"
    
    # IEEE-754 FP16 parameters
    sign_bit = 15
    exponent_start = 10
    exponent_bits = 5
    mantissa_bits = 10
    exponent_bias = 15
    
    # Basic constants
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
        outputs=["neg_one" + suffix],
        value=helper.make_tensor(
            name="neg_one_tensor" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[-1.0]
        )
    ))
    
    # Bit position constants
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
    
    # Cast to FP16 for consistent behavior
    nodes.append(helper.make_node(
        "Cast",
        inputs=[input_name],
        outputs=["input_fp16" + suffix],
        to=TensorProto.FLOAT16
    ))
    
    # Cast back to float32 for calculations
    nodes.append(helper.make_node(
        "Cast",
        inputs=["input_fp16" + suffix],
        outputs=["input_fp32" + suffix],
        to=TensorProto.FLOAT
    ))
    
    # Check if input is negative or zero
    nodes.append(helper.make_node(
        "Less",
        inputs=["input_fp32" + suffix, "zero" + suffix],
        outputs=["is_negative" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Equal",
        inputs=["input_fp32" + suffix, "zero" + suffix],
        outputs=["is_zero" + suffix]
    ))
    
    # --------------------------------
    # 1. ADD SPECIAL ZERO HANDLING
    # --------------------------------
    
    # For zero value, create special constants for each bit position
    # For bit 0-9 (mantissa bits):
    for mantissa_bit in range(10):
        # Calculate the exact value that results from setting this bit in zero
        value = 2.0 ** (mantissa_bit - 24)  # 2^(bit_pos - 24) for denormal values
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"zero_mantissa_bit_{mantissa_bit}" + suffix],
            value=helper.make_tensor(
                name=f"zero_mantissa_bit_{mantissa_bit}_tensor" + suffix,
                data_type=TensorProto.FLOAT16,
                dims=[],
                vals=[float(value)]
            )
        ))
    
    # For bit 10-14 (exponent bits):
    for exp_bit_idx, exp_bit in enumerate(range(10, 15)):
        # Each exponent bit creates a specific power of 2
        value = 2.0 ** (exp_bit_idx)  # 2^0 for bit 10, 2^1 for bit 11, etc.
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"zero_exp_bit_{exp_bit}" + suffix],
            value=helper.make_tensor(
                name=f"zero_exp_bit_{exp_bit}_tensor" + suffix,
                data_type=TensorProto.FLOAT16,
                dims=[],
                vals=[float(value)]
            )
        ))
    
    # For bit 15 (sign bit): just -0.0
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero_sign_bit_15" + suffix],
        value=helper.make_tensor(
            name="zero_sign_bit_15_tensor" + suffix,
            data_type=TensorProto.FLOAT16,
            dims=[],
            vals=[-0.0]
        )
    ))
    
    # --------------------------------
    # 2. DETERMINE BIT TYPE
    # --------------------------------
    
    # Check if sign bit
    nodes.append(helper.make_node(
        "Equal",
        inputs=["bit_pos" + suffix, "sign_bit_pos" + suffix],
        outputs=["is_sign_bit" + suffix]
    ))
    
    # Check if exponent bit
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
    
    # Check if mantissa bit
    nodes.append(helper.make_node(
        "Less",
        inputs=["bit_pos" + suffix, "exp_start" + suffix],
        outputs=["is_mantissa_bit" + suffix]
    ))
    
    # --------------------------------
    # 3. GET ABSOLUTE VALUE AND CALCULATE EXPONENT
    # --------------------------------
    
    nodes.append(helper.make_node(
        "Abs",
        inputs=["input_fp32" + suffix],
        outputs=["abs_value" + suffix]
    ))
    
    # Add a small epsilon to avoid log(0)
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["epsilon" + suffix],
        value=helper.make_tensor(
            name="epsilon_tensor" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[1e-10]
        )
    ))
    
    nodes.append(helper.make_node(
        "Add",
        inputs=["abs_value" + suffix, "epsilon" + suffix],
        outputs=["safe_abs" + suffix]
    ))
    
    # Calculate log2 with higher precision
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["ln2" + suffix],
        value=helper.make_tensor(
            name="ln2_tensor" + suffix,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[0.693147180559945]  # ln(2)
        )
    ))
    
    nodes.append(helper.make_node(
        "Log",
        inputs=["safe_abs" + suffix],
        outputs=["log_value" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Div",
        inputs=["log_value" + suffix, "ln2" + suffix],
        outputs=["log2_value" + suffix]
    ))
    
    # Handle zero input
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_zero" + suffix, "zero" + suffix, "log2_value" + suffix],
        outputs=["safe_log2" + suffix]
    ))
    
    # Calculate exponent
    nodes.append(helper.make_node(
        "Floor",
        inputs=["safe_log2" + suffix],
        outputs=["unbiased_exp" + suffix]
    ))
    
    # Add exponent bias for FP16
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
        "Add",
        inputs=["unbiased_exp" + suffix, "exp_bias" + suffix],
        outputs=["biased_exp" + suffix]
    ))
    
    # --------------------------------
    # 4. SIGN BIT HANDLING
    # --------------------------------
    
    # Simply negate the input
    nodes.append(helper.make_node(
        "Neg",
        inputs=["input_fp32" + suffix],
        outputs=["sign_flipped_raw" + suffix]
    ))
    
    # Re-quantize to FP16 precision
    nodes.append(helper.make_node(
        "Cast",
        inputs=["sign_flipped_raw" + suffix],
        outputs=["sign_flipped_fp16" + suffix],
        to=TensorProto.FLOAT16
    ))
    
    nodes.append(helper.make_node(
        "Cast",
        inputs=["sign_flipped_fp16" + suffix],
        outputs=["sign_flipped" + suffix],
        to=TensorProto.FLOAT
    ))
    
    # --------------------------------
    # 5. EXPONENT BIT HANDLING - IMPROVED APPROACH
    # --------------------------------
    
    # Improved approach for exponent bits - use pre-calculated constants for each bit
    # We'll create constants for each exponent bit position
    
    # Create a value to represent each exponent bit (based on biased exponent)
    for exp_bit_idx, exp_bit in enumerate(range(10, 15)):
        bit_value = 1 << exp_bit_idx  # 1, 2, 4, 8, 16
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"exp_bit_value_{exp_bit}" + suffix],
            value=helper.make_tensor(
                name=f"exp_bit_value_{exp_bit}_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[float(bit_value)]
            )
        ))
    
    # Extract the exponent bits (convert to INT32 for bit operations)
    nodes.append(helper.make_node(
        "Cast",
        inputs=["biased_exp" + suffix],
        outputs=["exp_int" + suffix],
        to=TensorProto.INT32
    ))
    
    # Create a mask for each exponent bit
    for exp_bit_idx, exp_bit in enumerate(range(10, 15)):
        bit_mask = 1 << exp_bit_idx
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"exp_bit_mask_{exp_bit}" + suffix],
            value=helper.make_tensor(
                name=f"exp_bit_mask_{exp_bit}_tensor" + suffix,
                data_type=TensorProto.INT32,
                dims=[],
                vals=[bit_mask]
            )
        ))
    
    # Check if each exponent bit is set
    for exp_bit in range(10, 15):
        nodes.append(helper.make_node(
            "BitwiseAnd",
            inputs=["exp_int" + suffix, f"exp_bit_mask_{exp_bit}" + suffix],
            outputs=[f"exp_bit_check_{exp_bit}" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Greater",
            inputs=[f"exp_bit_check_{exp_bit}" + suffix, "zero" + suffix],
            outputs=[f"exp_bit_is_set_{exp_bit}" + suffix]
        ))
    
    # Calculate scaling factor for each exponent bit
    # When bit is set: divide by 2^bit_weight
    # When bit is clear: multiply by 2^bit_weight
    for exp_bit_idx, exp_bit in enumerate(range(10, 15)):
        # Get the power of 2 this bit represents
        scaling = 2.0 ** (2 ** exp_bit_idx)  # 2^(2^idx)
        
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"exp_scaling_{exp_bit}" + suffix],
            value=helper.make_tensor(
                name=f"exp_scaling_{exp_bit}_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[scaling]
            )
        ))
        
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"exp_inv_scaling_{exp_bit}" + suffix],
            value=helper.make_tensor(
                name=f"exp_inv_scaling_{exp_bit}_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[1.0/scaling]
            )
        ))
    
    # For each exponent bit, if this is the target bit:
    # - If bit is set, use inv_scaling
    # - If bit is clear, use scaling
    for exp_bit in range(10, 15):
        # Check if this is the target bit
        nodes.append(helper.make_node(
            "Equal",
            inputs=["bit_pos" + suffix, f"exp_bit_value_{exp_bit}" + suffix],
            outputs=[f"is_target_exp_bit_{exp_bit}" + suffix]
        ))
        
        # Choose scale direction based on bit status
        nodes.append(helper.make_node(
            "Where",
            inputs=[f"exp_bit_is_set_{exp_bit}" + suffix, f"exp_inv_scaling_{exp_bit}" + suffix, f"exp_scaling_{exp_bit}" + suffix],
            outputs=[f"exp_scale_factor_{exp_bit}" + suffix]
        ))
        
        # Apply scaling if this is the target bit
        nodes.append(helper.make_node(
            "Mul",
            inputs=["input_fp32" + suffix, f"exp_scale_factor_{exp_bit}" + suffix],
            outputs=[f"exp_flipped_{exp_bit}_raw" + suffix]
        ))
        
        # Re-quantize to FP16 precision
        nodes.append(helper.make_node(
            "Cast",
            inputs=[f"exp_flipped_{exp_bit}_raw" + suffix],
            outputs=[f"exp_flipped_{exp_bit}_fp16" + suffix],
            to=TensorProto.FLOAT16
        ))
        
        nodes.append(helper.make_node(
            "Cast",
            inputs=[f"exp_flipped_{exp_bit}_fp16" + suffix],
            outputs=[f"exp_flipped_{exp_bit}" + suffix],
            to=TensorProto.FLOAT
        ))
    
    # --------------------------------
    # 6. MANTISSA BIT HANDLING - EXACT CALCULATION
    # --------------------------------
    
    # Get power of 2 for the exponent - for scaling mantissa bit weights
    nodes.append(helper.make_node(
        "Pow",
        inputs=["two" + suffix, "unbiased_exp" + suffix],
        outputs=["pow2_exp" + suffix]
    ))
    
    # Safe version that handles zero input
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_zero" + suffix, "one" + suffix, "pow2_exp" + suffix],
        outputs=["safe_pow2_exp" + suffix]
    ))
    
    # Create constants for each mantissa bit weight
    for mantissa_bit in range(10):
        # The weight for each mantissa bit depends on its position:
        # bit 0 (LSB) = 2^-10, bit 1 = 2^-9, etc.
        weight = 2.0 ** (mantissa_bit - 10)
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"mantissa_weight_{mantissa_bit}" + suffix],
            value=helper.make_tensor(
                name=f"mantissa_weight_{mantissa_bit}_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[weight]
            )
        ))
    
    # Extract the normalized mantissa
    nodes.append(helper.make_node(
        "Div",
        inputs=["abs_value" + suffix, "safe_pow2_exp" + suffix],
        outputs=["norm_mantissa_raw" + suffix]
    ))
    
    # Safe normalized mantissa (handle zero)
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_zero" + suffix, "zero" + suffix, "norm_mantissa_raw" + suffix],
        outputs=["norm_mantissa" + suffix]
    ))
    
    # Remove the implicit leading 1 to get just the fractional part
    nodes.append(helper.make_node(
        "Sub",
        inputs=["norm_mantissa" + suffix, "one" + suffix],
        outputs=["mantissa_frac" + suffix]
    ))
    
    # Calculate the mantissa as an integer (0-1023)
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
    
    nodes.append(helper.make_node(
        "Mul",
        inputs=["mantissa_frac" + suffix, "mantissa_scale" + suffix],
        outputs=["scaled_mantissa_raw" + suffix]
    ))
    
    # Round to nearest integer to avoid floating-point errors
    nodes.append(helper.make_node(
        "Round",
        inputs=["scaled_mantissa_raw" + suffix],
        outputs=["scaled_mantissa" + suffix]
    ))
    
    # For each mantissa bit 0-9, determine if it's set
    for mantissa_bit in range(10):
        # Create bit mask for this position
        bit_mask = 1 << mantissa_bit
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"mantissa_bit_mask_{mantissa_bit}" + suffix],
            value=helper.make_tensor(
                name=f"mantissa_bit_mask_{mantissa_bit}_tensor" + suffix,
                data_type=TensorProto.FLOAT,
                dims=[],
                vals=[float(bit_mask)]
            )
        ))
        
        # Check if this bit is set in the mantissa
        nodes.append(helper.make_node(
            "Mod",
            inputs=["scaled_mantissa" + suffix, f"mantissa_bit_mask_{mantissa_bit}" + suffix],
            outputs=[f"bit_check_{mantissa_bit}" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Greater",
            inputs=[f"bit_check_{mantissa_bit}" + suffix, "zero" + suffix],
            outputs=[f"mantissa_bit_is_set_{mantissa_bit}" + suffix]
        ))
        
        # Scale the bit weight by the exponent
        nodes.append(helper.make_node(
            "Mul",
            inputs=[f"mantissa_weight_{mantissa_bit}" + suffix, "safe_pow2_exp" + suffix],
            outputs=[f"scaled_mantissa_weight_{mantissa_bit}" + suffix]
        ))
        
        # Account for sign
        nodes.append(helper.make_node(
            "Where",
            inputs=["is_negative" + suffix, "neg_one" + suffix, "one" + suffix],
            outputs=["sign_factor" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Mul",
            inputs=[f"scaled_mantissa_weight_{mantissa_bit}" + suffix, "sign_factor" + suffix],
            outputs=[f"signed_mantissa_weight_{mantissa_bit}" + suffix]
        ))
        
        # For 0->1, add weight; for 1->0, subtract weight
        nodes.append(helper.make_node(
            "Neg",
            inputs=[f"signed_mantissa_weight_{mantissa_bit}" + suffix],
            outputs=[f"neg_signed_mantissa_weight_{mantissa_bit}" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Where",
            inputs=[f"mantissa_bit_is_set_{mantissa_bit}" + suffix, f"neg_signed_mantissa_weight_{mantissa_bit}" + suffix, f"signed_mantissa_weight_{mantissa_bit}" + suffix],
            outputs=[f"mantissa_delta_{mantissa_bit}" + suffix]
        ))
        
        # Apply delta if this is the target bit
        nodes.append(helper.make_node(
            "Equal",
            inputs=["bit_pos" + suffix, "mantissa_bit" + suffix],
            outputs=[f"is_target_mantissa_bit_{mantissa_bit}" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Add",
            inputs=["input_fp32" + suffix, f"mantissa_delta_{mantissa_bit}" + suffix],
            outputs=[f"mantissa_flipped_{mantissa_bit}_raw" + suffix]
        ))
        
        # Re-quantize to FP16 precision
        nodes.append(helper.make_node(
            "Cast",
            inputs=[f"mantissa_flipped_{mantissa_bit}_raw" + suffix],
            outputs=[f"mantissa_flipped_{mantissa_bit}_fp16" + suffix],
            to=TensorProto.FLOAT16
        ))
        
        nodes.append(helper.make_node(
            "Cast",
            inputs=[f"mantissa_flipped_{mantissa_bit}_fp16" + suffix],
            outputs=[f"mantissa_flipped_{mantissa_bit}" + suffix],
            to=TensorProto.FLOAT
        ))
    
    # --------------------------------
    # 7. COMBINE OUTPUTS WITH SPECIAL ZERO HANDLING
    # --------------------------------
    
    # Combine the sign, exponent, and mantissa bit flipping results
    
    # First handle zero value with precomputed constants
    nodes.append(helper.make_node(
        "And",
        inputs=["is_zero" + suffix, "is_sign_bit" + suffix],
        outputs=["is_zero_sign_bit" + suffix]
    ))
    
    # For each bit, create a condition to check if it's zero and the target bit
    for bit in range(16):
        nodes.append(helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"bit_const_{bit}" + suffix],
            value=helper.make_tensor(
                name=f"bit_const_{bit}_tensor" + suffix,
                data_type=TensorProto.INT64,
                dims=[],
                vals=[bit]
            )
        ))
        
        nodes.append(helper.make_node(
            "Equal",
            inputs=["bit_pos" + suffix, f"bit_const_{bit}" + suffix],
            outputs=[f"is_bit_{bit}" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "And",
            inputs=["is_zero" + suffix, f"is_bit_{bit}" + suffix],
            outputs=[f"is_zero_bit_{bit}" + suffix]
        ))
    
    # Combine all zero bit cases
    # For mantissa bits 0-9
    for mantissa_bit in range(10):
        nodes.append(helper.make_node(
            "Cast",
            inputs=[f"zero_mantissa_bit_{mantissa_bit}" + suffix],
            outputs=[f"zero_mantissa_bit_{mantissa_bit}_fp32" + suffix],
            to=TensorProto.FLOAT
        ))
    
    # For exponent bits 10-14
    for exp_bit in range(10, 15):
        nodes.append(helper.make_node(
            "Cast",
            inputs=[f"zero_exp_bit_{exp_bit}" + suffix],
            outputs=[f"zero_exp_bit_{exp_bit}_fp32" + suffix],
            to=TensorProto.FLOAT
        ))
    
    # For sign bit
    nodes.append(helper.make_node(
        "Cast",
        inputs=["zero_sign_bit_15" + suffix],
        outputs=["zero_sign_bit_15_fp32" + suffix],
        to=TensorProto.FLOAT
    ))
    
    # Create a result tensor for non-zero handling
    # Start with result of mantissa bit 0, then update with conditions
    nodes.append(helper.make_node(
        "Identity",
        inputs=["mantissa_flipped_0" + suffix],
        outputs=["normal_result" + suffix]
    ))
    
    # For each mantissa bit 1-9, update if it's the target bit
    for mantissa_bit in range(1, 10):
        nodes.append(helper.make_node(
            "Where",
            inputs=[f"is_target_mantissa_bit_{mantissa_bit}" + suffix, f"mantissa_flipped_{mantissa_bit}" + suffix, "normal_result" + suffix],
            outputs=["temp_result" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Identity",
            inputs=["temp_result" + suffix],
            outputs=["normal_result" + suffix]
        ))
    
    # For each exponent bit 10-14, update if it's the target bit
    for exp_bit in range(10, 15):
        nodes.append(helper.make_node(
            "Where",
            inputs=[f"is_target_exp_bit_{exp_bit}" + suffix, f"exp_flipped_{exp_bit}" + suffix, "normal_result" + suffix],
            outputs=["temp_result" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Identity",
            inputs=["temp_result" + suffix],
            outputs=["normal_result" + suffix]
        ))
    
    # Update if it's the sign bit
    nodes.append(helper.make_node(
        "Where",
        inputs=["is_sign_bit" + suffix, "sign_flipped" + suffix, "normal_result" + suffix],
        outputs=["non_zero_result" + suffix]
    ))
    
    # Combine zero and non-zero results
    # Start with zero case for mantissa bit 0
    nodes.append(helper.make_node(
        "Where",
        inputs=[f"is_zero_bit_0" + suffix, f"zero_mantissa_bit_0_fp32" + suffix, "non_zero_result" + suffix],
        outputs=["final_result" + suffix]
    ))
    
    # Handle other zero bit cases
    for bit in range(1, 16):
        if bit < 10:
            zero_val = f"zero_mantissa_bit_{bit}_fp32" + suffix
        elif bit < 15:
            zero_val = f"zero_exp_bit_{bit}_fp32" + suffix
        else:
            zero_val = "zero_sign_bit_15_fp32" + suffix
        
        nodes.append(helper.make_node(
            "Where",
            inputs=[f"is_zero_bit_{bit}" + suffix, zero_val, "final_result" + suffix],
            outputs=["temp_final" + suffix]
        ))
        
        nodes.append(helper.make_node(
            "Identity",
            inputs=["temp_final" + suffix],
            outputs=["final_result" + suffix]
        ))
    
    # Return the final result
    nodes.append(helper.make_node(
        "Identity",
        inputs=["final_result" + suffix],
        outputs=[output_name]
    ))
    
    return nodes

# def create_fp16_bit_flip(input_name, output_name, bit_position):
#     """
#     Create ONNX nodes that implement IEEE-754 FP16 bit flipping at a specified position.
#     Correctly handles all bit positions in the IEEE-754 format, including negative numbers.
    
#     Args:
#         input_name: Name of the input tensor (FLOAT)
#         output_name: Name of the output tensor (FLOAT)
#         bit_position: Position of bit to flip (0-15, where 0 is LSB of mantissa)
        
#     Returns:
#         List of ONNX nodes implementing the bit flip operation
#     """
#     nodes = []
#     suffix = "_bf"
    
#     # IEEE-754 FP16 parameters
#     sign_bit = 15
#     exponent_start = 10
#     exponent_bits = 5
#     mantissa_bits = 10
#     exponent_bias = 15
    
#     # Basic constants
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["zero" + suffix],
#         value=helper.make_tensor(
#             name="zero_tensor" + suffix,
#             data_type=TensorProto.FLOAT,
#             dims=[],
#             vals=[0.0]
#         )
#     ))
    
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["one" + suffix],
#         value=helper.make_tensor(
#             name="one_tensor" + suffix,
#             data_type=TensorProto.FLOAT,
#             dims=[],
#             vals=[1.0]
#         )
#     ))
    
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["two" + suffix],
#         value=helper.make_tensor(
#             name="two_tensor" + suffix,
#             data_type=TensorProto.FLOAT,
#             dims=[],
#             vals=[2.0]
#         )
#     ))
    
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["neg_one" + suffix],
#         value=helper.make_tensor(
#             name="neg_one_tensor" + suffix,
#             data_type=TensorProto.FLOAT,
#             dims=[],
#             vals=[-1.0]
#         )
#     ))
    
#     # Bit position constants
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["bit_pos" + suffix],
#         value=helper.make_tensor(
#             name="bit_pos_tensor" + suffix,
#             data_type=TensorProto.INT64,
#             dims=[],
#             vals=[bit_position]
#         )
#     ))
    
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["sign_bit_pos" + suffix],
#         value=helper.make_tensor(
#             name="sign_bit_pos_tensor" + suffix,
#             data_type=TensorProto.INT64,
#             dims=[],
#             vals=[sign_bit]
#         )
#     ))
    
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["exp_start" + suffix],
#         value=helper.make_tensor(
#             name="exp_start_tensor" + suffix,
#             data_type=TensorProto.INT64,
#             dims=[],
#             vals=[exponent_start]
#         )
#     ))
    
#     # Ensure FP16 precision at the start
#     nodes.append(helper.make_node(
#         "Cast",
#         inputs=[input_name],
#         outputs=["input_fp16" + suffix],
#         to=TensorProto.FLOAT16
#     ))
    
#     nodes.append(helper.make_node(
#         "Cast",
#         inputs=["input_fp16" + suffix],
#         outputs=["input_fp32" + suffix],
#         to=TensorProto.FLOAT
#     ))
    
#     # Check if input is negative
#     nodes.append(helper.make_node(
#         "Less",
#         inputs=["input_fp32" + suffix, "zero" + suffix],
#         outputs=["is_negative" + suffix]
#     ))
    
#     # --------------------------------
#     # 1. DETERMINE BIT TYPE
#     # --------------------------------
    
#     # Check if sign bit
#     nodes.append(helper.make_node(
#         "Equal",
#         inputs=["bit_pos" + suffix, "sign_bit_pos" + suffix],
#         outputs=["is_sign_bit" + suffix]
#     ))
    
#     # Check if exponent bit
#     nodes.append(helper.make_node(
#         "Less",
#         inputs=["bit_pos" + suffix, "sign_bit_pos" + suffix],
#         outputs=["lt_sign" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "GreaterOrEqual",
#         inputs=["bit_pos" + suffix, "exp_start" + suffix],
#         outputs=["ge_exp_start" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "And",
#         inputs=["lt_sign" + suffix, "ge_exp_start" + suffix],
#         outputs=["is_exp_bit" + suffix]
#     ))
    
#     # Check if mantissa bit
#     nodes.append(helper.make_node(
#         "Less",
#         inputs=["bit_pos" + suffix, "exp_start" + suffix],
#         outputs=["is_mantissa_bit" + suffix]
#     ))
    
#     # --------------------------------
#     # 2. SIGN BIT HANDLING
#     # --------------------------------
    
#     # Simply negate the input
#     nodes.append(helper.make_node(
#         "Neg",
#         inputs=["input_fp32" + suffix],
#         outputs=["sign_flipped" + suffix]
#     ))
    
#     # --------------------------------
#     # 3. EXPONENT BIT HANDLING
#     # --------------------------------
    
#     # Get absolute value
#     nodes.append(helper.make_node(
#         "Abs",
#         inputs=["input_fp32" + suffix],
#         outputs=["abs_value" + suffix]
#     ))
    
#     # Check if input is zero
#     nodes.append(helper.make_node(
#         "Equal",
#         inputs=["abs_value" + suffix, "zero" + suffix],
#         outputs=["is_zero" + suffix]
#     ))
    
#     # Calculate log2
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["ln2" + suffix],
#         value=helper.make_tensor(
#             name="ln2_tensor" + suffix,
#             data_type=TensorProto.FLOAT,
#             dims=[],
#             vals=[0.693147]  # ln(2)
#         )
#     ))
    
#     nodes.append(helper.make_node(
#         "Log",
#         inputs=["abs_value" + suffix],
#         outputs=["log_value" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Div",
#         inputs=["log_value" + suffix, "ln2" + suffix],
#         outputs=["log2_value" + suffix]
#     ))
    
#     # Handle zero input
#     nodes.append(helper.make_node(
#         "Where",
#         inputs=["is_zero" + suffix, "zero" + suffix, "log2_value" + suffix],
#         outputs=["safe_log2" + suffix]
#     ))
    
#     # Calculate exponent
#     nodes.append(helper.make_node(
#         "Floor",
#         inputs=["safe_log2" + suffix],
#         outputs=["unbiased_exp" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["exp_bias" + suffix],
#         value=helper.make_tensor(
#             name="exp_bias_tensor" + suffix,
#             data_type=TensorProto.FLOAT,
#             dims=[],
#             vals=[float(exponent_bias)]
#         )
#     ))
    
#     nodes.append(helper.make_node(
#         "Add",
#         inputs=["unbiased_exp" + suffix, "exp_bias" + suffix],
#         outputs=["biased_exp" + suffix]
#     ))
    
#     # Calculate bit position relative to exponent field
#     nodes.append(helper.make_node(
#         "Sub",
#         inputs=["bit_pos" + suffix, "exp_start" + suffix],
#         outputs=["exp_rel_pos" + suffix]
#     ))
    
#     # Calculate bit weight (2^position)
#     nodes.append(helper.make_node(
#         "Cast",
#         inputs=["exp_rel_pos" + suffix],
#         outputs=["exp_rel_pos_float" + suffix],
#         to=TensorProto.FLOAT
#     ))
    
#     nodes.append(helper.make_node(
#         "Pow",
#         inputs=["two" + suffix, "exp_rel_pos_float" + suffix],
#         outputs=["exp_bit_weight" + suffix]
#     ))
    
#     # Check if the bit is set
#     nodes.append(helper.make_node(
#         "Div",
#         inputs=["biased_exp" + suffix, "exp_bit_weight" + suffix],
#         outputs=["exp_div_result" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Floor",
#         inputs=["exp_div_result" + suffix],
#         outputs=["exp_div_floor" + suffix]
#     ))
    
#     # Calculate modulo 2
#     nodes.append(helper.make_node(
#         "Div",
#         inputs=["exp_div_floor" + suffix, "two" + suffix],
#         outputs=["exp_div_half" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Floor",
#         inputs=["exp_div_half" + suffix],
#         outputs=["exp_half_floor" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Mul",
#         inputs=["exp_half_floor" + suffix, "two" + suffix],
#         outputs=["exp_double_half_floor" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Sub",
#         inputs=["exp_div_floor" + suffix, "exp_double_half_floor" + suffix],
#         outputs=["exp_bit" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Equal",
#         inputs=["exp_bit" + suffix, "one" + suffix],
#         outputs=["exp_bit_is_one" + suffix]
#     ))
    
#     # Calculate the scale factor for flipping the exponent bit
#     nodes.append(helper.make_node(
#         "Pow",
#         inputs=["two" + suffix, "exp_bit_weight" + suffix],
#         outputs=["exp_scale_factor" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Div",
#         inputs=["one" + suffix, "exp_scale_factor" + suffix],
#         outputs=["exp_scale_down" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Where",
#         inputs=["exp_bit_is_one" + suffix, "exp_scale_down" + suffix, "exp_scale_factor" + suffix],
#         outputs=["exp_scale" + suffix]
#     ))
    
#     # Apply the exponent bit flip
#     nodes.append(helper.make_node(
#         "Mul",
#         inputs=["input_fp32" + suffix, "exp_scale" + suffix],
#         outputs=["exp_flipped_raw" + suffix]
#     ))
    
#     # Re-quantize to FP16 precision
#     nodes.append(helper.make_node(
#         "Cast",
#         inputs=["exp_flipped_raw" + suffix],
#         outputs=["exp_flipped_fp16" + suffix],
#         to=TensorProto.FLOAT16
#     ))
    
#     nodes.append(helper.make_node(
#         "Cast",
#         inputs=["exp_flipped_fp16" + suffix],
#         outputs=["exp_flipped" + suffix],
#         to=TensorProto.FLOAT
#     ))
    
#     # --------------------------------
#     # 4. MANTISSA BIT HANDLING WITH SIGN CONSIDERATION
#     # --------------------------------
    
#     # Get the power of 2 exponent
#     nodes.append(helper.make_node(
#         "Pow",
#         inputs=["two" + suffix, "unbiased_exp" + suffix],
#         outputs=["pow2_exp" + suffix]
#     ))
    
#     # Handle zero input for division
#     nodes.append(helper.make_node(
#         "Where",
#         inputs=["is_zero" + suffix, "one" + suffix, "pow2_exp" + suffix],
#         outputs=["safe_pow2_exp" + suffix]
#     ))
    
#     # Calculate the exact bit value in the IEEE-754 representation
#     # For mantissa bits, we need 2^(bit_position-10) * 2^exponent
#     nodes.append(helper.make_node(
#         "Cast",
#         inputs=["bit_pos" + suffix],
#         outputs=["bit_pos_float" + suffix],
#         to=TensorProto.FLOAT
#     ))
    
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["ten_float" + suffix],
#         value=helper.make_tensor(
#             name="ten_float_tensor" + suffix,
#             data_type=TensorProto.FLOAT,
#             dims=[],
#             vals=[10.0]
#         )
#     ))
    
#     # Calculate bit_position - 10
#     nodes.append(helper.make_node(
#         "Sub",
#         inputs=["bit_pos_float" + suffix, "ten_float" + suffix],
#         outputs=["mantissa_bit_adj" + suffix]
#     ))
    
#     # Calculate 2^(bit_position - 10)
#     nodes.append(helper.make_node(
#         "Pow",
#         inputs=["two" + suffix, "mantissa_bit_adj" + suffix],
#         outputs=["mantissa_bit_val" + suffix]
#     ))
    
#     # Multiply by 2^exponent to get the actual bit weight
#     nodes.append(helper.make_node(
#         "Mul",
#         inputs=["mantissa_bit_val" + suffix, "safe_pow2_exp" + suffix],
#         outputs=["mantissa_bit_weight" + suffix]
#     ))
    
#     # Calculate normalized mantissa to extract the bit value
#     nodes.append(helper.make_node(
#         "Div",
#         inputs=["abs_value" + suffix, "safe_pow2_exp" + suffix],
#         outputs=["norm_mantissa" + suffix]
#     ))
    
#     # Handle zero
#     nodes.append(helper.make_node(
#         "Where",
#         inputs=["is_zero" + suffix, "zero" + suffix, "norm_mantissa" + suffix],
#         outputs=["safe_norm_mantissa" + suffix]
#     ))
    
#     # Remove leading 1
#     nodes.append(helper.make_node(
#         "Sub",
#         inputs=["safe_norm_mantissa" + suffix, "one" + suffix],
#         outputs=["mantissa_frac" + suffix]
#     ))
    
#     # Scale to integer range [0, 1024)
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["mantissa_scale" + suffix],
#         value=helper.make_tensor(
#             name="mantissa_scale_tensor" + suffix,
#             data_type=TensorProto.FLOAT,
#             dims=[],
#             vals=[1024.0]  # 2^10
#         )
#     ))
    
#     nodes.append(helper.make_node(
#         "Mul",
#         inputs=["mantissa_frac" + suffix, "mantissa_scale" + suffix],
#         outputs=["scaled_mantissa" + suffix]
#     ))
    
#     # Get bit value at specified position
#     nodes.append(helper.make_node(
#         "Pow",
#         inputs=["two" + suffix, "bit_pos_float" + suffix],
#         outputs=["bit_pos_power" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Div",
#         inputs=["scaled_mantissa" + suffix, "bit_pos_power" + suffix],
#         outputs=["mantissa_shifted" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Floor",
#         inputs=["mantissa_shifted" + suffix],
#         outputs=["mantissa_shifted_floor" + suffix]
#     ))
    
#     # Calculate modulo 2
#     nodes.append(helper.make_node(
#         "Div",
#         inputs=["mantissa_shifted_floor" + suffix, "two" + suffix],
#         outputs=["mantissa_div_half" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Floor",
#         inputs=["mantissa_div_half" + suffix],
#         outputs=["mantissa_half_floor" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Mul",
#         inputs=["mantissa_half_floor" + suffix, "two" + suffix],
#         outputs=["mantissa_double_half" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Sub",
#         inputs=["mantissa_shifted_floor" + suffix, "mantissa_double_half" + suffix],
#         outputs=["mantissa_bit" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Equal",
#         inputs=["mantissa_bit" + suffix, "one" + suffix],
#         outputs=["mantissa_bit_is_one" + suffix]
#     ))
    
#     # Calculate delta to flip the bit
#     # For positive numbers:
#     #   - If bit is 0: add bit_weight
#     #   - If bit is 1: subtract bit_weight
#     # For negative numbers:
#     #   - If bit is 0: subtract bit_weight (makes more negative)
#     #   - If bit is 1: add bit_weight (makes less negative)
    
#     # First, calculate sign multiplier: 1 for positive, -1 for negative
#     nodes.append(helper.make_node(
#         "Where",
#         inputs=["is_negative" + suffix, "neg_one" + suffix, "one" + suffix],
#         outputs=["sign_multiplier" + suffix]
#     ))
    
#     # For positive, bit 0->1: add weight
#     # For negative, bit 0->1: subtract weight (multiply by -1)
#     nodes.append(helper.make_node(
#         "Mul",
#         inputs=["mantissa_bit_weight" + suffix, "sign_multiplier" + suffix],
#         outputs=["mantissa_delta_pos" + suffix]
#     ))
    
#     # For bit 1->0, use the opposite
#     nodes.append(helper.make_node(
#         "Neg",
#         inputs=["mantissa_delta_pos" + suffix],
#         outputs=["mantissa_delta_neg" + suffix]
#     ))
    
#     # Choose delta based on the current bit value
#     nodes.append(helper.make_node(
#         "Where",
#         inputs=["mantissa_bit_is_one" + suffix, "mantissa_delta_neg" + suffix, "mantissa_delta_pos" + suffix],
#         outputs=["mantissa_delta" + suffix]
#     ))
    
#     # Apply delta to input
#     nodes.append(helper.make_node(
#         "Add",
#         inputs=["input_fp32" + suffix, "mantissa_delta" + suffix],
#         outputs=["mantissa_flipped_raw" + suffix]
#     ))
    
#     # Re-quantize to FP16 precision
#     nodes.append(helper.make_node(
#         "Cast",
#         inputs=["mantissa_flipped_raw" + suffix],
#         outputs=["mantissa_flipped_fp16" + suffix],
#         to=TensorProto.FLOAT16
#     ))
    
#     nodes.append(helper.make_node(
#         "Cast",
#         inputs=["mantissa_flipped_fp16" + suffix],
#         outputs=["mantissa_flipped" + suffix],
#         to=TensorProto.FLOAT
#     ))
    
#     # --------------------------------
#     # 5. SPECIAL HANDLING FOR ZERO
#     # --------------------------------
    
#     # For bit 10 (lowest exponent bit), flipping gives smallest normal value
#     nodes.append(helper.make_node(
#         "Equal",
#         inputs=["bit_pos" + suffix, "exp_start" + suffix],
#         outputs=["is_lowest_exp_bit" + suffix]
#     ))
    
#     # Special case for zero with exponent bit flipping
#     nodes.append(helper.make_node(
#         "And", 
#         inputs=["is_zero" + suffix, "is_exp_bit" + suffix],
#         outputs=["zero_exp_flip" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "And",
#         inputs=["zero_exp_flip" + suffix, "is_lowest_exp_bit" + suffix],
#         outputs=["zero_to_smallest" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["smallest_normal" + suffix],
#         value=helper.make_tensor(
#             name="smallest_normal_tensor" + suffix,
#             data_type=TensorProto.FLOAT,
#             dims=[],
#             vals=[6.103515625e-5]  # Smallest normal FP16
#         )
#     ))
    
#     # For zero with mantissa bits, calculate proper denormal
#     nodes.append(helper.make_node(
#         "And", 
#         inputs=["is_zero" + suffix, "is_mantissa_bit" + suffix],
#         outputs=["zero_mantissa_flip" + suffix]
#     ))
    
#     # Calculate 2^(-14 + bit_position - 10) = 2^(bit_position - 24)
#     nodes.append(helper.make_node(
#         "Constant",
#         inputs=[],
#         outputs=["neg_14" + suffix],
#         value=helper.make_tensor(
#             name="neg_14_tensor" + suffix,
#             data_type=TensorProto.FLOAT,
#             dims=[],
#             vals=[-14.0]
#         )
#     ))
    
#     nodes.append(helper.make_node(
#         "Add",
#         inputs=["neg_14" + suffix, "mantissa_bit_adj" + suffix],
#         outputs=["zero_mantissa_power" + suffix]
#     ))
    
#     nodes.append(helper.make_node(
#         "Pow",
#         inputs=["two" + suffix, "zero_mantissa_power" + suffix],
#         outputs=["zero_mantissa_value" + suffix]
#     ))
    
#     # --------------------------------
#     # 6. COMBINE OUTPUTS
#     # --------------------------------
    
#     # First choose between exponent and mantissa flipped values
#     nodes.append(helper.make_node(
#         "Where",
#         inputs=["is_exp_bit" + suffix, "exp_flipped" + suffix, "mantissa_flipped" + suffix],
#         outputs=["exp_or_mantissa" + suffix]
#     ))
    
#     # Then choose sign-flipped if appropriate
#     nodes.append(helper.make_node(
#         "Where",
#         inputs=["is_sign_bit" + suffix, "sign_flipped" + suffix, "exp_or_mantissa" + suffix],
#         outputs=["normal_result" + suffix]
#     ))
    
#     # Handle special case for zero to smallest normal
#     nodes.append(helper.make_node(
#         "Where",
#         inputs=["zero_to_smallest" + suffix, "smallest_normal" + suffix, "normal_result" + suffix],
#         outputs=["zero_exp_handled" + suffix]
#     ))
    
#     # Handle special case for zero with mantissa bit
#     nodes.append(helper.make_node(
#         "Where",
#         inputs=["zero_mantissa_flip" + suffix, "zero_mantissa_value" + suffix, "zero_exp_handled" + suffix],
#         outputs=[output_name]
#     ))
    
#     return nodes

def create_fp16_input_bit_flip(input_name, output_name, bit_position):
    nodes = []
    suffix = "_rf"  # random flip suffix
    
    # -------------------------------
    # 1. Random Index Generation
    # -------------------------------
    
    # Get input shape (e.g., [batch, sequence, hidden])
    nodes.append(helper.make_node(
        'Shape',
        inputs=[input_name],
        outputs=['runtime_shape' + suffix]
    ))
    
    # Cast shape to float for multiplication
    nodes.append(helper.make_node(
        'Cast',
        inputs=['runtime_shape' + suffix],
        outputs=['runtime_shape_float' + suffix],
        to=TensorProto.FLOAT
    ))
    
    # Generate random uniform values with the same length as the shape
    nodes.append(helper.make_node(
        'Shape',
        inputs=['runtime_shape' + suffix],
        outputs=['shape_rank' + suffix]
    ))
    
    # Create dynamic shape for RandomUniform
    nodes.append(helper.make_node(
        'ConstantOfShape',
        inputs=['shape_rank' + suffix],
        outputs=['rand_shape' + suffix],
        value=helper.make_tensor(
            name='rand_shape_val' + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]
        )
    ))
    
    # Generate random values between 0 and 1
    nodes.append(helper.make_node(
        'RandomUniform',
        inputs=['rand_shape' + suffix],
        outputs=['random_vals' + suffix],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0
    ))
    
    # Scale random values by the tensor dimensions
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals' + suffix, 'runtime_shape_float' + suffix],
        outputs=['scaled_indices' + suffix]
    ))
    
    # Floor to get integer indices
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_indices' + suffix],
        outputs=['floored_indices' + suffix]
    ))
    
    # Cast to INT64 for indexing
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_indices' + suffix],
        outputs=['indices_int64' + suffix],
        to=TensorProto.INT64
    ))
    
    # Unsqueeze to create index for GatherND
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['indices_int64' + suffix],
        outputs=['indices_unsqueezed' + suffix],
        axes=[0]  # Add a new axis at position 0
    ))
    
    # -------------------------------
    # 2. Extract value at random position & apply bit flip
    # -------------------------------
    
    # Use GatherND to get the value at the random index
    nodes.append(helper.make_node(
        'GatherND',
        inputs=[input_name, 'indices_unsqueezed' + suffix],
        outputs=['selected_value' + suffix]
    ))
    
    # Cast to float32 for bit flip operations
    nodes.append(helper.make_node(
        'Cast',
        inputs=['selected_value' + suffix],
        outputs=['selected_value_float' + suffix],
        to=TensorProto.FLOAT
    ))
    
    # Create an intermediate output name
    flipped_value = 'flipped_value' + suffix
    
    # Generate the bit flip nodes for the selected value
    bit_flip_nodes = create_fp16_bit_flip('selected_value_float' + suffix, flipped_value, bit_position)
    nodes.extend(bit_flip_nodes)
    
    # -------------------------------
    # 3. Calculate perturbation (difference between original and flipped value)
    # -------------------------------
    
    # Calculate the perturbation (flipped - original)
    nodes.append(helper.make_node(
        'Sub',
        inputs=[flipped_value, 'selected_value_float' + suffix],
        outputs=['value_perturbation' + suffix]
    ))
    
    # Cast perturbation to FP16
    nodes.append(helper.make_node(
        'Cast',
        inputs=['value_perturbation' + suffix],
        outputs=['value_perturbation_fp16' + suffix],
        to=TensorProto.FLOAT16
    ))
    
    # -------------------------------
    # 4. Create perturbation tensor (zeros with perturbation at target position)
    # -------------------------------
    
    # Create zero tensor with same shape as input
    nodes.append(helper.make_node(
        'ConstantOfShape',
        inputs=['runtime_shape' + suffix],
        outputs=['zeros' + suffix],
        value=helper.make_tensor(
            name='zeros_val' + suffix,
            data_type=TensorProto.FLOAT16,
            dims=[1],
            vals=[0.0]
        )
    ))
    
    # Use ScatterND to insert the perturbation value at the random position
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=['zeros' + suffix, 'indices_unsqueezed' + suffix, 'value_perturbation_fp16' + suffix],
        outputs=[output_name]  # This is the perturbation tensor only
    ))
    
    return nodes