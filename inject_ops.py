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

def create_weight16_mask(matmul_output="y", masked_output="y_masked",
                                  block_length=4):
    nodes = []
    suffix = "_mask"

    # 1. Get the shape of the MatMul output 'y'
    nodes.append(helper.make_node(
        "Shape",
        inputs=[matmul_output],
        outputs=["y_shape_mask"]
    ))

    # --- Create constants for slicing the shape tensor ---
    # For M_value (number of rows): slice indices [0:1] along axis 0.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["M_starts"],
        value=helper.make_tensor(
            name="M_starts_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["M_ends"],
        value=helper.make_tensor(
            name="M_ends_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]
        )
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["M_axes"],
        value=helper.make_tensor(
            name="M_axes_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))

    # For N_value (number of columns): slice indices [1:2] along axis 0.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["N_starts"],
        value=helper.make_tensor(
            name="N_starts_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]
        )
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["N_ends"],
        value=helper.make_tensor(
            name="N_ends_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[2]
        )
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["N_axes"],
        value=helper.make_tensor(
            name="N_axes_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))

    # 2. Slice out the first dimension (M) from y_shape_mask.
    nodes.append(helper.make_node(
        "Slice",
        inputs=["y_shape_mask", "M_starts", "M_ends", "M_axes"],
        outputs=["M_value"]
    ))

    # 2.5. Squeeze M_value from shape [1] to a scalar.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["squeeze_axes"],
        value=helper.make_tensor(
            name="squeeze_axes_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))
    nodes.append(helper.make_node(
        "Squeeze",
        inputs=["M_value", "squeeze_axes"],
        outputs=["M_scalar"]
    ))

    # 3. Slice out the second dimension (N) from y_shape_mask.
    nodes.append(helper.make_node(
        "Slice",
        inputs=["y_shape_mask", "N_starts", "N_ends", "N_axes"],
        outputs=["N_value"]
    ))

    # 4. Create constants 0 and 1 for use with Range.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero_const_mask"],
        value=helper.make_tensor(
            name="zero_tensor_mask",
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[0]
        )
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["one_const_mask"],
        value=helper.make_tensor(
            name="one_tensor_mask",
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[1]
        )
    ))

    # 5. Generate a vector of row indices: Range(0, M_scalar, 1)
    nodes.append(helper.make_node(
        "Range",
        inputs=["zero_const_mask", "M_scalar", "one_const_mask"],
        outputs=["row_indices"]
    ))

    # 6. --- Compute dynamic (random) start index ---
    # Create a constant for block_length.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["block_length_const"],
        value=helper.make_tensor(
            name="block_length_tensor",
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[block_length]
        )
    ))
    # Compute M_minus_block = M_scalar - block_length_const.
    nodes.append(helper.make_node(
        "Sub",
        inputs=["M_scalar", "block_length_const"],
        outputs=["M_minus_block"]
    ))
    # Compute range_size = M_minus_block + 1.
    nodes.append(helper.make_node(
        "Add",
        inputs=["M_minus_block", "one_const_mask"],
        outputs=["range_size"]
    ))
    # Cast range_size to FLOAT.
    nodes.append(helper.make_node(
        "Cast",
        inputs=["range_size"],
        outputs=["range_size_float"],
        to=TensorProto.FLOAT
    ))
    # Generate a random FLOAT value in [0, 1) with shape [1].
    nodes.append(helper.make_node(
        "RandomUniform",
        inputs=[],
        outputs=["rand_val_temp"],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0,
        shape=[1]
    ))
    # Squeeze the random value to obtain a scalar.
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["squeeze_axes_rand"],
        value=helper.make_tensor(
            name="squeeze_axes_rand_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))
    nodes.append(helper.make_node(
        "Squeeze",
        inputs=["rand_val_temp", "squeeze_axes_rand"],
        outputs=["rand_val"]
    ))
    # Scale the random value: rand_scaled = rand_val * range_size_float.
    nodes.append(helper.make_node(
        "Mul",
        inputs=["rand_val", "range_size_float"],
        outputs=["rand_scaled"]
    ))
    # Floor the scaled random value.
    nodes.append(helper.make_node(
        "Floor",
        inputs=["rand_scaled"],
        outputs=["rand_index_float"]
    ))
    # Cast the floored value to INT64 to get dynamic start index.
    nodes.append(helper.make_node(
        "Cast",
        inputs=["rand_index_float"],
        outputs=["start_index_dynamic"],
        to=TensorProto.INT64
    ))
    # Compute end_index_dynamic = start_index_dynamic + block_length_const.
    nodes.append(helper.make_node(
        "Add",
        inputs=["start_index_dynamic", "block_length_const"],
        outputs=["end_index_dynamic"]
    ))

    # 7. --- Build the mask using the dynamic start index ---
    # Compare: row_indices >= start_index_dynamic.
    nodes.append(helper.make_node(
        "GreaterOrEqual",
        inputs=["row_indices", "start_index_dynamic"],
        outputs=["ge_mask"]
    ))
    # Compare: row_indices < end_index_dynamic.
    nodes.append(helper.make_node(
        "Less",
        inputs=["row_indices", "end_index_dynamic"],
        outputs=["lt_mask"]
    ))
    # Combine the two Boolean masks using And.
    nodes.append(helper.make_node(
        "And",
        inputs=["ge_mask", "lt_mask"],
        outputs=["mask_bool"]
    ))
    # Cast the Boolean mask to FLOAT16.
    nodes.append(helper.make_node(
        "Cast",
        inputs=["mask_bool"],
        outputs=["mask_float"],
        to=TensorProto.FLOAT16
    ))
    # Unsqueeze mask_float to shape [M, 1] (provide axes as input).
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_axes_mask"],
        value=helper.make_tensor(
            name="unsqueeze_axes_mask_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]
        )
    ))
    nodes.append(helper.make_node(
        "Unsqueeze",
        inputs=["mask_float", "unsqueeze_axes_mask"],
        outputs=["mask_unsqueezed"]
    ))
    # Create a constant '1' for tiling as a 1D tensor (shape [1]).
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["one_for_tile"],
        value=helper.make_tensor(
            name="one_for_tile_tensor",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]
        )
    ))
    # Concatenate [1] with N_value to form tile multiples: [1, N].
    nodes.append(helper.make_node(
        "Concat",
        inputs=["one_for_tile", "N_value"],
        outputs=["tile_multiples"],
        axis=0
    ))
    # Tile the mask_unsqueezed to shape [M, N].
    nodes.append(helper.make_node(
        "Tile",
        inputs=["mask_unsqueezed", "tile_multiples"],
        outputs=["mask_full"]
    ))
    # Multiply the MatMul output 'y' with the mask.
    nodes.append(helper.make_node(
        "Mul",
        inputs=[matmul_output, "mask_full"],
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
