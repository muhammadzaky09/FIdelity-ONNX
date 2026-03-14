from onnx import helper, TensorProto
import numpy as np
from typing import List
from onnxruntime_extensions import onnx_op, PyCustomOpDef, get_library_path as _get_library_path
import struct

def create_quantized_fault_injection(input_name, output_name, bit_position,
                                     fp16=False, is_signed=True,
                                     rand_idx_name="rand_idx_inject"):
    nodes = []
    suffix   = "_inject"
    int_type = TensorProto.INT8   if is_signed else TensorProto.UINT8
    prec     = TensorProto.FLOAT16 if fp16     else TensorProto.FLOAT

    # 1) Cast original to int
    nodes.append(helper.make_node(
        "Cast",
        inputs=[input_name],
        outputs=["int_val" + suffix],
        to=int_type
    ))

    # 2) Record original shape [d0, d1, ..., dN]
    nodes.append(helper.make_node(
        "Shape",
        inputs=[input_name],
        outputs=["orig_shape" + suffix]
    ))

    # Create a constant tensor for the reduce axes
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["reduce_axes" + suffix],
        value=helper.make_tensor(
            name="reduce_axes_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))
    
    # 3) Compute flat size = product of all dims (with axes as input)
    nodes.append(helper.make_node(
        "ReduceProd",
        inputs=["orig_shape" + suffix, "reduce_axes" + suffix],  # Add axes as input
        outputs=["flat_size" + suffix],
        keepdims=0  # keepdims can remain as an attribute
    ))

    # 4) Flatten to 1-D
    nodes.append(helper.make_node(
        "Flatten",
        inputs=[input_name],
        outputs=["flat_val" + suffix],
        axis=0
    ))

    # Create a constant tensor for the axes
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
    
    # 5) Unsqueeze flat_size → [flat_size] (compatible with opset 13+)
    nodes.append(helper.make_node(
        "Unsqueeze",
        inputs=["flat_size" + suffix, "unsqueeze_axes" + suffix],  # Add axes as input
        outputs=["flat_shape" + suffix]
        # Remove the "axes" attribute
    ))

    # 6) Create zero vector of length flat_size
    nodes.append(helper.make_node(
        "ConstantOfShape",
        inputs=["flat_shape" + suffix],
        outputs=["zero_flat" + suffix],
        value=helper.make_tensor(
            name="zero_val" + suffix,
            data_type=int_type,
            dims=[1],
            vals=[0]
        )
    ))

    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["axes_0" + suffix],
        value=helper.make_tensor(
            name="axes_0_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[0]
        )
    ))
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["axes_1" + suffix],
        value=helper.make_tensor(
            name="axes_1_tensor" + suffix,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1]
        )
    ))
    nodes.append(helper.make_node(
        "Unsqueeze",
        inputs=[rand_idx_name, "axes_0" + suffix],
        outputs=["idx_ud" + suffix]
    ))
    nodes.append(helper.make_node(
        "Unsqueeze",
        inputs=["idx_ud" + suffix, "axes_1" + suffix],
        outputs=["idx_sc" + suffix]
    ))

    # 9) Build the bitmask to flip a single bit
    nodes.append(helper.make_node(
        "Constant",
        inputs=[],
        outputs=["bitmask" + suffix],
        value=helper.make_tensor(
            name="bitmask_val" + suffix,
            data_type=int_type,
            dims=[1],
            vals=[1 << bit_position]
        )
    ))

    # 10) Scatter the bitmask into our zero vector
    nodes.append(helper.make_node(
        "ScatterND",
        inputs=["zero_flat"  + suffix,
                "idx_sc"     + suffix,
                "bitmask"    + suffix],
        outputs=["mask_flat" + suffix]
    ))

    # 11) Reshape mask_flat → mask_nd of original shape
    nodes.append(helper.make_node(
        "Reshape",
        inputs=["mask_flat"  + suffix,
                "orig_shape" + suffix],
        outputs=["mask"       + suffix]
    ))

    # 12) Flip the bit with XOR
    nodes.append(helper.make_node(
        "BitwiseXor",
        inputs=["int_val" + suffix, "mask" + suffix],
        outputs=["flipped_int" + suffix]
    ))

    # 13) Compute perturbation = flipped - original
    nodes.append(helper.make_node(
        "Cast",
        inputs=["flipped_int" + suffix],
        outputs=["flipped_i32" + suffix],
        to=TensorProto.INT32
    ))
    nodes.append(helper.make_node(
        "Cast",
        inputs=["int_val"       + suffix],
        outputs=["orig_i32"      + suffix],
        to=TensorProto.INT32
    ))
    nodes.append(helper.make_node(
        "Sub",
        inputs=["flipped_i32" + suffix, "orig_i32" + suffix],
        outputs=["perturb_i32" + suffix]
    ))

    # 14) Cast perturbation back to float16/float
    nodes.append(helper.make_node(
        "Cast",
        inputs=["perturb_i32" + suffix],
        outputs=[output_name],
        to=prec
    ))

    return nodes

def create_weight16_mask(matmul_output="y", masked_output="y_masked", block_length=4, fp16=True):
    nodes = []
    suffix = "_mask"
    
    # original shape of the input
    nodes.append(helper.make_node(
        "Shape",
        inputs=[matmul_output],
        outputs=["orig_shape" + suffix]
    ))

    # flatten everything except the last (hidden) dimension; after this
    # the tensor has shape (N, H) where N = prod(original_dims[:-1])
    nodes.append(helper.make_node(
        "Flatten",
        inputs=[matmul_output],
        outputs=["y_flat" + suffix],
        axis=-1  # keep last dim intact
    ))

    # We now operate on y_flat instead of the original tensor
    working_tensor = "y_flat" + suffix
    
    # Define all constants at the beginning
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
    
    # Create a constant for index 1
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
    
    # Create a constant for index 2
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
    
    # Create a constant for -1
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
    
    # 1. Get the shape of the flattened tensor
    nodes.append(helper.make_node(
        "Shape",
        inputs=[working_tensor],
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
    
    # 3. Calculate the second-to-last dimension index (rank - 2)
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
    
    # FIXED: Changed input from "feature_indices" to "seq_indices"
    nodes.append(helper.make_node(
        "Less",  # Using "Less" for exclusive upper bound
        inputs=["seq_indices" + suffix, "end_idx" + suffix],
        outputs=["lt_mask" + suffix]
    ))
    
    nodes.append(helper.make_node(
        "And",
        inputs=["ge_mask" + suffix, "lt_mask" + suffix],
        outputs=["bool_mask_1d" + suffix]
    ))
    
    # 13. Create shape for reshaping the mask
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
    
    # FIXED: Use the axes input tensor for Unsqueeze
    nodes.append(helper.make_node(
        "Unsqueeze",
        inputs=["second_last_dim_idx" + suffix, "unsqueeze_axes" + suffix],
        outputs=["second_last_dim_idx_unsqueezed" + suffix]
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
    
    # 15. Create zeros tensor for masked values with correct data type
    nodes.append(helper.make_node(
        "ConstantOfShape",
        inputs=["y_shape" + suffix],
        outputs=["zeros" + suffix],
        value=helper.make_tensor(
            name="zeros_value" + suffix,
            data_type=TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT,  # FIXED: Use FLOAT16 when fp16=True
            dims=[1],
            vals=[0.0]
        )
    ))
    
    # 16. Apply the mask on the FLATTENED tensor
    nodes.append(helper.make_node(
        "Where",
        inputs=["bool_mask_broadcast" + suffix, working_tensor, "zeros" + suffix],
        outputs=["masked_flat" + suffix]
    ))
    
    # 17. Reshape the masked tensor back to the original shape so that
    #     downstream layers remain unaffected
    nodes.append(helper.make_node(
        "Reshape",
        inputs=["masked_flat" + suffix, "orig_shape" + suffix],
        outputs=[masked_output]
    ))
    
    return nodes

def create_input16_mask(matmul_output="y", masked_output="y_masked", block_length=16, fp16=False):
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
    
    # 5. Block length (16 per FIdelity spec, but parameterised for flexibility)
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
            data_type=TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT,
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

def create_random_fault_injection(output_name: str, random_value: float, fp16: bool = True,
                                  rand_idx_name: str = "rand_idx_inject"):
    nodes = []
    suffix = "_random"
    prec = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT

    # Save original shape to restore after flat ScatterND
    nodes.append(helper.make_node(
        'Shape', inputs=[output_name], outputs=['orig_shape' + suffix]))

    # Flatten to 1D: shape [N]
    nodes.append(helper.make_node(
        'Constant', inputs=[], outputs=['flat_shape' + suffix],
        value=helper.make_tensor('flat_shape_t' + suffix, TensorProto.INT64, [1], [-1])))
    nodes.append(helper.make_node(
        'Reshape', inputs=[output_name, 'flat_shape' + suffix],
        outputs=['flat' + suffix]))

    # rand_idx_name (INT64 scalar) → [[idx]] shape [1,1] for ScatterND on 1D tensor
    nodes.append(helper.make_node(
        'Constant', inputs=[], outputs=['idx_shape' + suffix],
        value=helper.make_tensor('idx_shape_t' + suffix, TensorProto.INT64, [2], [1, 1])))
    nodes.append(helper.make_node(
        'Reshape', inputs=[rand_idx_name, 'idx_shape' + suffix], outputs=['idx_2d' + suffix]))

    # Fault value to scatter
    nodes.append(helper.make_node(
        'Constant', inputs=[], outputs=['fault_val' + suffix],
        value=helper.make_tensor('fault_val_t' + suffix, prec, [1], [random_value])))

    # Scatter fault_value at rand_idx in flat tensor, then reshape back
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=['flat' + suffix, 'idx_2d' + suffix, 'fault_val' + suffix],
        outputs=['flat_faulty' + suffix]))
    nodes.append(helper.make_node(
        'Reshape', inputs=['flat_faulty' + suffix, 'orig_shape' + suffix],
        outputs=[f'{output_name}_faulty']))

    return nodes


def create_random_bitflip_injection(output_name: str, bit_position: int,
                                    fp16: bool = True,
                                    rand_idx_name: str = "rand_idx_inject"):
    """
    RANDOM_BITFLIP using the BitFlip custom op (custom.bitflip domain).
    Inputs: (fp16_tensor, bit_position:int32, fault_index:int64)
    BitFlip copies the full tensor and flips bit_position at fault_index —
    bit-exact, no float arithmetic, no rounding error.
    fault_index is supplied externally (rand_idx_name) for reproducibility.
    """
    suffix = "_rbf"
    faulty_output = f"{output_name}_faulty"
    nodes = []

    nodes.append(helper.make_node(
        'Constant', inputs=[], outputs=['bit_pos_const' + suffix],
        value=helper.make_tensor('bit_pos_tensor' + suffix, TensorProto.INT32, [1], [bit_position])))

    nodes.append(helper.make_node(
        'BitFlip',
        inputs=[output_name, 'bit_pos_const' + suffix, rand_idx_name],
        outputs=[faulty_output],
        domain='custom.bitflip'))

    return nodes





def create_fp16_fault_injection(input_name, output_name, bit_position,
                                fp32=False, rand_idx_name="rand_idx_inject"):
    nodes = []
    suffix = "_fp16fi"

    intermediate_input = input_name
    intermediate_output = output_name

    # Cast FP32 → FP16 if needed
    if fp32:
        fp16_input = input_name + "_fp16" + suffix
        nodes.append(helper.make_node(
            'Cast', inputs=[input_name], outputs=[fp16_input],
            to=TensorProto.FLOAT16))
        intermediate_input = fp16_input
        intermediate_output = output_name + "_fp16" + suffix

    # Bit-position constant (INT32 scalar)
    nodes.append(helper.make_node(
        'Constant', inputs=[], outputs=['bit_pos' + suffix],
        value=helper.make_tensor(
            'bit_pos_t' + suffix, TensorProto.INT32, [1], [bit_position])))

    # BitFlip → full perturbed tensor (bit-exact, no rounding)
    nodes.append(helper.make_node(
        'BitFlip',
        inputs=[intermediate_input, 'bit_pos' + suffix, rand_idx_name],
        outputs=['perturbed' + suffix],
        domain='custom.bitflip'))

    # Delta = perturbed − original  (computed in FP32 to avoid FP16 cancellation)
    nodes.append(helper.make_node(
        'Cast', inputs=['perturbed' + suffix],
        outputs=['perturbed_f32' + suffix], to=TensorProto.FLOAT))
    nodes.append(helper.make_node(
        'Cast', inputs=[intermediate_input],
        outputs=['orig_f32' + suffix], to=TensorProto.FLOAT))
    nodes.append(helper.make_node(
        'Sub',
        inputs=['perturbed_f32' + suffix, 'orig_f32' + suffix],
        outputs=['delta_f32' + suffix]))
    nodes.append(helper.make_node(
        'Cast', inputs=['delta_f32' + suffix],
        outputs=[intermediate_output], to=TensorProto.FLOAT16))

    # Cast delta FP16 → FP32 if the original input was FP32
    if fp32:
        nodes.append(helper.make_node(
            'Cast', inputs=[intermediate_output],
            outputs=[output_name], to=TensorProto.FLOAT))

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
def create_random_bitflip_fp32(output_name, bit_position,
                               rand_idx_name: str = "rand_idx_inject"):
    faulty_output = f"{output_name}_faulty"
    nodes = []
    suffix = "_rbf"

    # Save original shape to restore after flat operations
    nodes.append(helper.make_node(
        'Shape', inputs=[output_name], outputs=['orig_shape' + suffix]))

    # Flatten to 1D: shape [N]
    nodes.append(helper.make_node(
        'Constant', inputs=[], outputs=['flat_shape' + suffix],
        value=helper.make_tensor('flat_shape_t' + suffix, TensorProto.INT64, [1], [-1])))
    nodes.append(helper.make_node(
        'Reshape', inputs=[output_name, 'flat_shape' + suffix],
        outputs=['flat' + suffix]))

    # rand_idx_name (INT64 scalar) → [[idx]] shape [1,1] for GatherND / ScatterND on 1D tensor
    nodes.append(helper.make_node(
        'Constant', inputs=[], outputs=['ax0' + suffix],
        value=helper.make_tensor('ax0_t' + suffix, TensorProto.INT64, [1], [0])))
    nodes.append(helper.make_node(
        'Constant', inputs=[], outputs=['ax1' + suffix],
        value=helper.make_tensor('ax1_t' + suffix, TensorProto.INT64, [1], [1])))
    nodes.append(helper.make_node(
        'Unsqueeze', inputs=[rand_idx_name, 'ax0' + suffix], outputs=['idx_1d' + suffix]))
    nodes.append(helper.make_node(
        'Unsqueeze', inputs=['idx_1d' + suffix, 'ax1' + suffix], outputs=['idx_2d' + suffix]))

    # GatherND: select one element from flat tensor → shape [1]
    nodes.append(helper.make_node(
        'GatherND', inputs=['flat' + suffix, 'idx_2d' + suffix],
        outputs=['selected_val' + suffix]))

    # Bit-flip the single selected element
    nodes.append(helper.make_node(
        'Constant', inputs=[], outputs=['bit_pos_const' + suffix],
        value=helper.make_tensor('bit_pos_tensor' + suffix, TensorProto.INT32, [1], [bit_position])))
    nodes.append(helper.make_node(
        'DirectBitToggleFp32',
        inputs=['selected_val' + suffix, 'bit_pos_const' + suffix],
        outputs=['toggled_val' + suffix], domain="ai.onnx.contrib"))

    # ScatterND the toggled value back into flat tensor, then reshape
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=['flat' + suffix, 'idx_2d' + suffix, 'toggled_val' + suffix],
        outputs=['flat_faulty' + suffix]))
    nodes.append(helper.make_node(
        'Reshape', inputs=['flat_faulty' + suffix, 'orig_shape' + suffix],
        outputs=[faulty_output]))

    return nodes

# -------------------------------------------------------------
# Convolution-specific 16-neuron masks
# -------------------------------------------------------------

def create_conv_input16_mask(conv_output="y", masked_output="y_masked", block_length=16, fp16=True):
    """Keep 16 consecutive channels across all spatial positions.

    Input tensor assumed NCHW. Mask selects a slice along C of length
    min(block_length, C) starting at a random channel index, and
    broadcasts over N,H,W.
    """
    nodes = []
    suffix = "_cim"

    # 1. shape
    nodes.append(helper.make_node("Shape", [conv_output], ["shape"+suffix]))

    # gather C dim (index 1)
    nodes.append(helper.make_node("Constant", [], ["idx1"+suffix],
        value=helper.make_tensor("idx1_t"+suffix, TensorProto.INT64, [], [1])))
    nodes.append(helper.make_node("Gather", ["shape"+suffix, "idx1"+suffix], ["Cdim"+suffix], axis=0))

    # valid block = min(block_length, C)
    nodes.append(helper.make_node("Constant", [], ["blk"+suffix],
        value=helper.make_tensor("blk_t"+suffix, TensorProto.INT64, [], [block_length])))
    nodes.append(helper.make_node("Min", ["blk"+suffix, "Cdim"+suffix], ["vblock"+suffix]))

    # max_start = C - vblock
    nodes.append(helper.make_node("Sub", ["Cdim"+suffix, "vblock"+suffix], ["maxst"+suffix]))

    # random start
    nodes.append(helper.make_node("Cast", ["maxst"+suffix], ["maxst_f"+suffix], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("RandomUniform", [], ["rstart"+suffix], dtype=TensorProto.FLOAT, high=1.0, low=0.0, shape=[1]))
    nodes.append(helper.make_node("Mul", ["rstart"+suffix, "maxst_f"+suffix], ["rst_scaled"+suffix]))
    nodes.append(helper.make_node("Floor", ["rst_scaled"+suffix], ["rst_floor"+suffix]))
    nodes.append(helper.make_node("Cast", ["rst_floor"+suffix], ["sidx"+suffix], to=TensorProto.INT64))

    # build channel indices 0..C-1
    nodes.append(helper.make_node("Range", inputs=["zero_scalar"+suffix if False else None], outputs=[]))
    # but we need zero_scalar const and one_scalar
    nodes.append(helper.make_node("Constant", [], ["z"+suffix], value=helper.make_tensor("z"+suffix, TensorProto.INT64, [], [0])))
    nodes.append(helper.make_node("Constant", [], ["one"+suffix], value=helper.make_tensor("one"+suffix, TensorProto.INT64, [], [1])))
    nodes.append(helper.make_node("Range", ["z"+suffix, "Cdim"+suffix, "one"+suffix], ["cidx"+suffix]))

    # boolean mask: cidx >= sidx AND cidx < sidx+vblock
    nodes.append(helper.make_node("Add", ["sidx"+suffix, "vblock"+suffix], ["endidx"+suffix]))
    nodes.append(helper.make_node("GreaterOrEqual", ["cidx"+suffix, "sidx"+suffix], ["ge"+suffix]))
    nodes.append(helper.make_node("Less", ["cidx"+suffix, "endidx"+suffix], ["lt"+suffix]))
    nodes.append(helper.make_node("And", ["ge"+suffix, "lt"+suffix], ["mask1d"+suffix]))

    # reshape to broadcast [1,C,1,1]
    nodes.append(helper.make_node("Constant", [], ["shape4"+suffix], value=helper.make_tensor("shape4t"+suffix, TensorProto.INT64, [4], [1,0,1,1])))
    # The 0 will be replaced by C via ScatterND
    nodes.append(helper.make_node("ConstantOfShape", ["shape4"+suffix], ["shapevec"+suffix], value=helper.make_tensor("sv"+suffix, TensorProto.INT64, [1], [1])))
    # scatter
    nodes.append(helper.make_node("Constant", [], ["axis1"+suffix], value=helper.make_tensor("ax1"+suffix, TensorProto.INT64, [1], [1])))
    nodes.append(helper.make_node("Unsqueeze", ["idx1"+suffix, "axis1"+suffix], ["idx1u"+suffix]))
    nodes.append(helper.make_node("ScatterND", ["shapevec"+suffix, "idx1u"+suffix, "Cdim"+suffix], ["bshape"+suffix]))
    nodes.append(helper.make_node("Reshape", ["mask1d"+suffix, "bshape"+suffix], ["mask4d"+suffix]))

    # zeros tensor
    dtype = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    nodes.append(helper.make_node("ConstantOfShape", ["shape"+suffix], ["zeros"+suffix], value=helper.make_tensor("zv"+suffix, dtype, [1], [0.0])))

    nodes.append(helper.make_node("Where", ["mask4d"+suffix, conv_output, "zeros"+suffix], [masked_output]))

    return nodes


def create_conv_weight16_mask(conv_output="y", masked_output="y_masked", block_length=16, fp16=True):
    nodes = []
    suffix = "_cwm"

    # shape
    nodes.append(helper.make_node("Shape", [conv_output], ["shape"+suffix]))

    # Gather H (idx 2) and W (idx3)
    for idx,val in [(2,"H"), (3,"W")]:
        nodes.append(helper.make_node("Constant", [], [f"idx{idx}{suffix}"], value=helper.make_tensor(f"i{idx}{suffix}", TensorProto.INT64, [], [idx])))
        nodes.append(helper.make_node("Gather", ["shape"+suffix, f"idx{idx}{suffix}"], [f"{val}{suffix}"], axis=0))

    # flatten spatial size N = H*W
    nodes.append(helper.make_node("Mul", ["H"+suffix, "W"+suffix], ["Nsp"+suffix]))

    # valid_block
    nodes.append(helper.make_node("Constant", [], ["blk"+suffix], value=helper.make_tensor("blk_t"+suffix, TensorProto.INT64, [], [block_length])))
    nodes.append(helper.make_node("Min", ["blk"+suffix, "Nsp"+suffix], ["vblock"+suffix]))

    nodes.append(helper.make_node("Sub", ["Nsp"+suffix, "vblock"+suffix], ["maxst"+suffix]))

    nodes.append(helper.make_node("Cast", ["maxst"+suffix], ["maxst_f"+suffix], to=TensorProto.FLOAT))
    nodes.append(helper.make_node("RandomUniform", [], ["rstart"+suffix], dtype=TensorProto.FLOAT, high=1.0, low=0.0, shape=[1]))
    nodes.append(helper.make_node("Mul", ["rstart"+suffix, "maxst_f"+suffix], ["rst_scaled"+suffix]))
    nodes.append(helper.make_node("Floor", ["rst_scaled"+suffix], ["rst_floor"+suffix]))
    nodes.append(helper.make_node("Cast", ["rst_floor"+suffix], ["sidx"+suffix], to=TensorProto.INT64))

    # Build indices 0..Nsp-1
    nodes.append(helper.make_node("Constant", [], ["z"+suffix], value=helper.make_tensor("z"+suffix, TensorProto.INT64, [], [0])))
    nodes.append(helper.make_node("Constant", [], ["one"+suffix], value=helper.make_tensor("1"+suffix, TensorProto.INT64, [], [1])))
    nodes.append(helper.make_node("Range", ["z"+suffix, "Nsp"+suffix, "one"+suffix], ["spidx"+suffix]))

    nodes.append(helper.make_node("Add", ["sidx"+suffix, "vblock"+suffix], ["endidx"+suffix]))
    nodes.append(helper.make_node("GreaterOrEqual", ["spidx"+suffix, "sidx"+suffix], ["ge"+suffix]))
    nodes.append(helper.make_node("Less", ["spidx"+suffix, "endidx"+suffix], ["lt"+suffix]))
    nodes.append(helper.make_node("And", ["ge"+suffix, "lt"+suffix], ["mask1d"+suffix]))

    # reshape to [1,1,H,W]
    nodes.append(helper.make_node("Constant", [], ["shape4"+suffix], value=helper.make_tensor("sh"+suffix, TensorProto.INT64, [4], [1,1,0,0])))
    # scatter dims for H,W
    nodes.append(helper.make_node("Constant", [], ["axes0"+suffix], value=helper.make_tensor("ax0"+suffix, TensorProto.INT64, [1], [2])))
    nodes.append(helper.make_node("Unsqueeze", [f"idx2{suffix}", "axes0"+suffix], ["idx2u"+suffix]))
    nodes.append(helper.make_node("ScatterND", ["shape4"+suffix, "idx2u"+suffix, "H"+suffix], ["shape_hw1"+suffix]))
    nodes.append(helper.make_node("Constant", [], ["axes1"+suffix], value=helper.make_tensor("ax1b"+suffix, TensorProto.INT64, [1], [3])))
    nodes.append(helper.make_node("Unsqueeze", [f"idx3{suffix}", "axes1"+suffix], ["idx3u"+suffix]))
    nodes.append(helper.make_node("ScatterND", ["shape_hw1"+suffix, "idx3u"+suffix, "W"+suffix], ["bshape"+suffix]))
    nodes.append(helper.make_node("Reshape", ["mask1d"+suffix, "bshape"+suffix], ["mask4d"+suffix]))

    dtype = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    nodes.append(helper.make_node("ConstantOfShape", ["shape"+suffix], ["zeros"+suffix], value=helper.make_tensor("zv"+suffix, dtype, [1], [0.0])))

    nodes.append(helper.make_node("Where", ["mask4d"+suffix, conv_output, "zeros"+suffix], [masked_output]))
    
    return nodes

