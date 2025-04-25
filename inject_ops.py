from onnx import helper, TensorProto
import numpy as np
from typing import List
from onnxruntime_extensions import onnx_op, PyCustomOpDef, get_library_path as _get_library_path
import struct

def create_quantized_fault_injection(input_name, output_name, bit_position, fp32=False, is_signed=True ):
    nodes = []
    suffix = "_inject"
    int_type = TensorProto.INT8 if is_signed else TensorProto.UINT8
    prec = TensorProto.FLOAT if fp32 else TensorProto.FLOAT16
    
    # Cast directly to the integer type for bit manipulation
    nodes.append(helper.make_node(
        'Cast',
        inputs=[input_name],
        outputs=['int_val' + suffix],
        to=int_type
    ))
    
    # Get shape (we still need the shape for generating random indices)
    nodes.append(helper.make_node(
        'Shape',
        inputs=[input_name],
        outputs=['runtime_shape' + suffix]
    ))

    # Cast shape to float for random calculations
    nodes.append(helper.make_node(
        'Cast',
        inputs=['runtime_shape' + suffix],
        outputs=['runtime_shape_float' + suffix],
        to=TensorProto.FLOAT
    ))
    
    # Random index generation (same as before)
    nodes.append(helper.make_node(
        'RandomUniformLike',
        inputs=['runtime_shape_float' + suffix],
        outputs=['random_vals' + suffix],
        high=1.0,
        low=0.0
    ))
    
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals' + suffix, 'runtime_shape_float' + suffix],
        outputs=['scaled_indices' + suffix]
    ))
    
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_indices' + suffix],
        outputs=['floored_indices' + suffix]
    ))
    
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_indices' + suffix],
        outputs=['indices_int64' + suffix],
        to=TensorProto.INT64
    ))
    
    # Reshape indices for ScatterND
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
        outputs=['indices_int64_unsqueezed' + suffix]
    ))
    
    # Create bitmask
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['bitmask' + suffix],
        value=helper.make_tensor(
            name='bitmask_val' + suffix,
            data_type=int_type,
            dims=[1],            
            vals=[1 << bit_position]
        )
    ))
    
    # Create zero tensor
    nodes.append(helper.make_node(
        'ConstantOfShape',
        inputs=['runtime_shape' + suffix],
        outputs=['zero_base' + suffix],
        value=helper.make_tensor(
            name='zero_value' + suffix,
            data_type=int_type,
            dims=[1],
            vals=[0]
        )
    ))
    
    # Place bitmask at random position
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=['zero_base' + suffix, 'indices_int64_unsqueezed' + suffix, 'bitmask' + suffix],
        outputs=['bit_mask' + suffix]
    ))
    
    # Apply bit flip with XOR
    nodes.append(helper.make_node(
        'BitwiseXor',
        inputs=['int_val' + suffix, 'bit_mask' + suffix],
        outputs=['flipped_int' + suffix]
    ))
    
    # Calculate perturbation
    nodes.append(helper.make_node(
        'Cast',
        inputs=['flipped_int' + suffix],
        outputs=['flipped_int32' + suffix],
        to=TensorProto.INT32
    ))
    
    nodes.append(helper.make_node(
        'Cast',
        inputs=['int_val' + suffix],
        outputs=['int_val32' + suffix],
        to=TensorProto.INT32
    ))
    
    nodes.append(helper.make_node(
        'Sub',
        inputs=['flipped_int32' + suffix, 'int_val32' + suffix],
        outputs=['perturbation_int32' + suffix]
    ))
    
    # Cast perturbation to output format
    nodes.append(helper.make_node(
        'Cast',
        inputs=['perturbation_int32' + suffix],
        outputs=[output_name],
        to=prec
    ))
    
    return nodes

def create_weight16_mask(matmul_output="y", masked_output="y_masked", block_length=4, fp16=True):
    nodes = []
    suffix = "_mask"
    
    # 1. Get the shape of the input tensor
    nodes.append(helper.make_node(
        "Shape",
        inputs=[matmul_output],
        outputs=["y_shape" + suffix]
    ))
    
    # 2. Get the rank (number of dimensions)
    nodes.append(helper.make_node(
        "Shape",
        inputs=["y_shape" + suffix],
        outputs=["rank_shape" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Gather",
        inputs=["rank_shape" + suffix, "zero_const" + suffix],
        outputs=["rank" + suffix],
        axis=0
    ))
    
    # Create a constant for index 0
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero_const" + suffix],
        value=helper.make_tensor(
            name="zero_const_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[0]
        )
    ))
    
   
    
    # 3. Calculate the second-to-last dimension index (rank - 2)
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["two_const" + suffix],
        value=helper.make_tensor(
            name="two_const_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[2]
        )
    ))
    
    nodes.append(helper.make_node(
        "Sub",
        inputs=["rank" + suffix, "two_const" + suffix],
        outputs=["second_last_dim_idx" + suffix]
    ))
    
    # 4. Get the size of the second-to-last dimension
    nodes.append(helper.make_node(
        "Gather",
        inputs=["y_shape" + suffix, "second_last_dim_idx" + suffix],
        outputs=["seq_len_tensor" + suffix],
        axis=0
    ))
    
    # 5. Create scalar constants for range
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
    
    # 6. Create a range of indices (0 to seq_len-1)
    nodes.append(helper.make_node(
        "Range",
        inputs=["zero_scalar" + suffix, "seq_len_tensor" + suffix, "one_scalar" + suffix],
        outputs=["seq_indices" + suffix]
    ))
    
    # 7. Create block length constant
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
    
    # 8. Calculate valid block length (min of block_length and seq_len)
    nodes.append(helper.make_node(
        "Min",
        inputs=["block_len" + suffix, "seq_len_tensor" + suffix],
        outputs=["valid_block" + suffix]
    ))
    
    # 9. Calculate max start index
    nodes.append(helper.make_node(
        "Sub",
        inputs=["seq_len_tensor" + suffix, "valid_block" + suffix],
        outputs=["max_start" + suffix]
    ))
    
    # 10. Generate random start index
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
        inputs=["rand_tensor" + suffix],
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
    
    # 11. Calculate end index
    nodes.append(helper.make_node(
        "Add",
        inputs=["start_idx" + suffix, "valid_block" + suffix],
        outputs=["end_idx" + suffix]
    ))
    
    # 12. Create boolean mask
    nodes.append(helper.make_node(
        "GreaterOrEqual",
        inputs=["seq_indices" + suffix, "start_idx" + suffix],
        outputs=["ge_mask" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "LessOrEqual",  # Changed from "Less"
        inputs=["feature_indices" + suffix, "end_idx" + suffix],
        outputs=["lt_mask" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "And",
        inputs=["ge_mask" + suffix, "lt_mask" + suffix],
        outputs=["bool_mask_1d" + suffix]
    ))
    
    # 13. Create shape for reshaping the mask
    # We'll create a shape tensor with rank number of dimensions
    # All dimensions are 1 except for the second-to-last which is -1 (inferred)
    
    # First create a tensor of ones with length equal to the rank
    nodes.append(helper.make_node(
        "ConstantOfShape",
        inputs=["rank_shape" + suffix],
        outputs=["ones_tensor" + suffix],
        value=helper.make_tensor(
            name="value_info" + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]
        )
    ))
    
    # Now update the second-to-last dimension to -1
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["neg_one_const" + suffix],
        value=helper.make_tensor(
            name="neg_one_const_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[-1]
        )
    ))
    
    # Create an update tensor for ScatterND
    nodes.append(helper.make_node(
        "Unsqueeze",
        inputs=["second_last_dim_idx" + suffix],
        outputs=["second_last_dim_idx_unsqueezed" + suffix],
        axes=[0]
    ))
    
    nodes.append(helper.make_node(
        "ScatterND",
        inputs=["ones_tensor" + suffix, "second_last_dim_idx_unsqueezed" + suffix, "neg_one_const" + suffix],
        outputs=["reshape_shape" + suffix]
    ))
    
    # 14. Reshape the boolean mask for proper broadcasting
    nodes.append(helper.make_node(
        "Reshape",
        inputs=["bool_mask_1d" + suffix, "reshape_shape" + suffix],
        outputs=["bool_mask_broadcast" + suffix]
    ))
    
    # 15. Create zeros tensor for masked values
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
    
    # 16. Use Where instead of Mul for proper broadcasting
    nodes.append(helper.make_node(
        "Where",
        inputs=["bool_mask_broadcast" + suffix, matmul_output, "zeros" + suffix],
        outputs=[masked_output]
    ))
    
    return nodes

def create_input16_mask(matmul_output="y", masked_output="y_masked", block_length=16):
    nodes = []
    suffix = "_mask"
    
    # 1. Get the shape of the input tensor
    nodes.append(helper.make_node(
        "Shape",
        inputs=[matmul_output],
        outputs=["y_shape" + suffix]
    ))
    
    # 2. Get the rank (number of dimensions)
    nodes.append(helper.make_node(
        "Shape",
        inputs=["y_shape" + suffix],
        outputs=["rank_shape" + suffix]
    ))
    
    # Create a constant for index 0
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero_const" + suffix],
        value=helper.make_tensor(
            name="zero_const_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[0]
        )
    ))
    
    nodes.append(helper.make_node(
        "Gather",
        inputs=["rank_shape" + suffix, "zero_const" + suffix],
        outputs=["rank" + suffix],
        axis=0
    ))
    
    # 3. Calculate the last dimension index (rank - 1)
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["one_const" + suffix],
        value=helper.make_tensor(
            name="one_const_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[1]
        )
    ))
    
    nodes.append(helper.make_node(
        "Sub",
        inputs=["rank" + suffix, "one_const" + suffix],
        outputs=["last_dim_idx" + suffix]
    ))
    
    # 4. Get the size of the last dimension
    nodes.append(helper.make_node(
        "Gather",
        inputs=["y_shape" + suffix, "last_dim_idx" + suffix],
        outputs=["feature_len" + suffix],
        axis=0
    ))
    
    # 5. Create a fixed block length of 16
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["block_len" + suffix],
        value=helper.make_tensor(
            name="block_len_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[16]  # Force exactly 16 elements
        )
    ))
    
    # 6. Calculate max start index safely
    nodes.append(helper.make_node(
        "Sub",
        inputs=["feature_len" + suffix, "block_len" + suffix],
        outputs=["max_start_tmp" + suffix]
    ))
    
    # Ensure max_start is not negative
    nodes.append(helper.make_node(
        "Max",
        inputs=["max_start_tmp" + suffix, "zero_const" + suffix],
        outputs=["max_start" + suffix]
    ))
    
    # 7. Generate random start index
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
        inputs=["rand_tensor" + suffix],
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
    
    # 8. Calculate end index
    nodes.append(helper.make_node(
        "Add",
        inputs=["start_idx" + suffix, "block_len" + suffix],
        outputs=["end_idx" + suffix]
    ))
    
    # 9. Create feature indices
    nodes.append(helper.make_node(
        "Range",
        inputs=["zero_const" + suffix, "feature_len" + suffix, "one_const" + suffix],
        outputs=["feature_indices" + suffix]
    ))
    
    # 10. Create strictly boolean mask (exactly 16 positions)
    nodes.append(helper.make_node(
        "GreaterOrEqual",
        inputs=["feature_indices" + suffix, "start_idx" + suffix],
        outputs=["ge_mask" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "Less",
        inputs=["feature_indices" + suffix, "end_idx" + suffix],
        outputs=["lt_mask" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "And",
        inputs=["ge_mask" + suffix, "lt_mask" + suffix],
        outputs=["bool_mask_1d" + suffix]
    ))
    
    # 11. Create shape for reshaping the mask
    nodes.append(helper.make_node(
        "ConstantOfShape",
        inputs=["rank_shape" + suffix],
        outputs=["ones_tensor" + suffix],
        value=helper.make_tensor(
            name="value_info" + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]
        )
    ))
    
    # Create axes input for Unsqueeze
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unsqueeze_axes" + suffix],
        value=helper.make_tensor(
            name="unsqueeze_axes_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))
    
    # Now update the last dimension to -1
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["neg_one_const" + suffix],
        value=helper.make_tensor(
            name="neg_one_const_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[],
            vals=[-1]
        )
    ))
    
    # Use Unsqueeze with two inputs
    nodes.append(helper.make_node(
        "Unsqueeze",
        inputs=["last_dim_idx" + suffix, "unsqueeze_axes" + suffix],
        outputs=["last_dim_idx_unsqueezed" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "ScatterND",
        inputs=["ones_tensor" + suffix, "last_dim_idx_unsqueezed" + suffix, "neg_one_const" + suffix],
        outputs=["reshape_shape" + suffix]
    ))
    
    # 12. Reshape the boolean mask for proper broadcasting
    nodes.append(helper.make_node(
        "Reshape",
        inputs=["bool_mask_1d" + suffix, "reshape_shape" + suffix],
        outputs=["bool_mask_broadcast" + suffix]
    ))
    
    # 13. Create a fixed value tensor (instead of using ZerosLike)
    nodes.append(helper.make_node(
        "ConstantOfShape",
        inputs=["y_shape" + suffix],
        outputs=["zeros_tensor" + suffix],
        value=helper.make_tensor(
            name="zeros_value" + suffix,
            data_type=TensorProto.FLOAT16,
            dims=[1],
            vals=[0.0]
        )
    ))
    
    # 15. Use the exact mask to extract values (avoid multiplying by zero values)
    nodes.append(helper.make_node(
        "Where",
        inputs=["bool_mask_broadcast" + suffix, matmul_output, "zeros_tensor" + suffix],
        outputs=[masked_output]
    ))
    
    return nodes

def create_random_fault_injection(output_name: str, random_value: float):
    nodes = []
    suffix = "_random"
    
    nodes.append(helper.make_node(
        'Shape',
        inputs=[output_name],
        outputs=['runtime_shape' + suffix]
    ))
    
    nodes.append(helper.make_node(
        'Cast',
        inputs=['runtime_shape' + suffix],
        outputs=['runtime_shape_float' + suffix],
        to=TensorProto.FLOAT
    ))
    
    nodes.append(helper.make_node(
        'RandomUniformLike',
        inputs=['runtime_shape' + suffix],
        outputs=['random_vals' + suffix],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0
    ))
    
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals' + suffix, 'runtime_shape_float' + suffix],
        outputs=['scaled_indices' + suffix]
    ))
    
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_indices' + suffix],
        outputs=['floored_indices' + suffix]
    ))
    
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_indices' + suffix],
        outputs=['indices_int64' + suffix],
        to=TensorProto.INT64
    ))
    
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
    
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['fault_value' + suffix],
        value=helper.make_tensor(
            name='fault_value_tensor' + suffix,
            data_type=TensorProto.FLOAT,
            dims=[1],
            vals=[random_value]
        )
    ))
    
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

def create_fp16_fault_injection(input_name, output_name, bit_position, fp32=True):
    nodes = []
    suffix = ""  # No suffix needed for the integrated operator
    
    intermediate_input = input_name
    intermediate_output = output_name
    
    # If input is FP32, cast to FP16 first
    if fp32:
        fp16_input = input_name + "_fp16"
        nodes.append(helper.make_node(
            'Cast',
            inputs=[input_name],
            outputs=[fp16_input],
            to=TensorProto.FLOAT16
        ))
        intermediate_input = fp16_input
        
        # We'll need to cast back to FP32 at the end
        intermediate_output = output_name + "_fp16"

    # 1. Create a constant node for the bit position
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
    
    # 2. Create the perturb node that does everything internally
    perturb_node = helper.make_node(
        'Perturb',  # Custom operator
        inputs=[intermediate_input, 'bit_pos_const' + suffix],
        outputs=[intermediate_output],
        domain='custom.perturb'
    )
    nodes.append(perturb_node)
    
    # If input was FP32, cast result back to FP32
    if fp32:
        nodes.append(helper.make_node(
            'Cast',
            inputs=[intermediate_output],
            outputs=[output_name],
            to=TensorProto.FLOAT
        ))
    
    return nodes

@onnx_op(op_type="DirectBitToggleFp32",
         inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_int32],
         outputs=[PyCustomOpDef.dt_float])
def direct_bit_toggle_fp32_op(x, bit_position):
    result = np.empty_like(x)
    if bit_position.size == 1:
        bit_pos = bit_position.item() if bit_position.ndim == 0 else bit_position[0]
        single_bit_pos = True
    else:
        single_bit_pos = False
    for idx in np.ndindex(x.shape):
        val = x[idx]
        if single_bit_pos:
            pos = bit_pos
        else:
            pos = bit_position[idx] if idx < bit_position.shape else bit_position[0]
        
        if 0 <= pos < 32:
            bytes_data = np.array(val, dtype=np.float32).tobytes()
            bits = struct.unpack('I', bytes_data)[0]
            toggled_bits = bits ^ (1 << pos)
            toggled_bytes = struct.pack('I', toggled_bits)
            result[idx] = np.frombuffer(toggled_bytes, dtype=np.float32)[0]
        else:
            result[idx] = val
    
    return result
def create_random_bitflip_fp32( output_name, bit_position):
    faulty_output = f"{output_name}_faulty"
    nodes = []
    suffix = "_rbf"
    nodes.append(helper.make_node(
        'Shape',
        inputs=[output_name],
        outputs=['input_shape' + suffix]
    ))
    nodes.append(helper.make_node(
        'Cast',
        inputs=['input_shape' + suffix],
        outputs=['input_shape_float' + suffix],
        to=TensorProto.FLOAT
    ))
    nodes.append(helper.make_node(
        'RandomUniformLike',
        inputs=['input_shape_float' + suffix],
        outputs=['random_vals' + suffix],
        low=0.0,
        high=1.0
    ))
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals' + suffix, 'input_shape_float' + suffix],
        outputs=['scaled_indices' + suffix]
    ))
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_indices' + suffix],
        outputs=['floored_indices_float' + suffix]
    ))
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_indices_float' + suffix],
        outputs=['rand_index' + suffix],
        to=TensorProto.INT64
    ))
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
        inputs=['rand_index' + suffix, 'unsqueeze_axes' + suffix],
        outputs=['rand_index_unsqueezed' + suffix]
    ))
    nodes.append(helper.make_node(
        'GatherND',
        inputs=[output_name, 'rand_index_unsqueezed' + suffix],
        outputs=['selected_val' + suffix]
    ))
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
    nodes.append(helper.make_node(
        'DirectBitToggleFp32',
        inputs=['selected_val' + suffix, 'bit_pos_const' + suffix],
        outputs=['toggled_val' + suffix],
        domain="ai.onnx.contrib"
    ))
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=[output_name, 'rand_index_unsqueezed' + suffix, 'toggled_val' + suffix],
        outputs=[faulty_output]
    ))
    
    return nodes

