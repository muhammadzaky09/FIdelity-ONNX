from onnx import helper, TensorProto
import numpy as np
from typing import List

def create_int8_fault_injection(input_name, input_shape, target_indices, bit_position,output_name):
    """Create fault injection using pure ONNX operations.
    
    Args:
        input_name: Name of input tensor
        input_shape: Shape of input tensor (e.g., [1, 32, 512])
        target_indices: Index per dimension (e.g., [0, 16, 256])
        bit_position: Which bit to flip (0-7)
    """
    nodes = [
        # 1. Create shape tensor for zeros
        helper.make_node("Constant",
            inputs=[],
            outputs=["shape_tensor"],
            value=helper.make_tensor(
                name="const_shape",
                data_type=TensorProto.INT64,
                dims=[len(input_shape)],
                vals=input_shape
            )
        ),
        
        # 2. Create zeros tensor using ConstantOfShape
        helper.make_node("ConstantOfShape",
            inputs=["shape_tensor"],
            outputs=["zeros"],
            value=helper.make_tensor(
                name="const_zero_value",
                data_type=TensorProto.INT8,
                dims=[1],
                vals=[0]
            )
        ),
        
        # 3. Create indices tensor for target position
        helper.make_node("Constant",
            inputs=[],
            outputs=["indices"],
            value=helper.make_tensor(
                name="const_indices",
                data_type=TensorProto.INT64,
                dims=[1, len(target_indices)],  # Shape required by ScatterND
                vals=target_indices
            )
        ),
        
        # 4. Create bit mask value (1 << bit_position)
        helper.make_node("Constant",
            inputs=[],
            outputs=["bit_mask"],
            value=helper.make_tensor(
                name="const_mask",
                data_type=TensorProto.INT8,
                dims=[1],
                vals=[1 << bit_position]
            )
        ),
        
        # 5. Create mask tensor using ScatterND
        helper.make_node("ScatterND",
            inputs=["zeros", "indices", "bit_mask"],
            outputs=["mask_tensor"]
        ),
        
        # 6. BitWiseXor with input tensor
        helper.make_node("BitwiseXor",
            inputs=[input_name, "mask_tensor"],
            outputs=[output_name]
        )
    ]
    
    return nodes

def create_random_int8_fault_injection(output_name: str, output_shape: List[int], target_indices: List[int], random_value: int):
    """Create random fault injection into layer output.
    
    Args:
        output_name: Name of layer output tensor to fault
        output_shape: Shape of output tensor
        target_indices: Where to inject fault
        random_value: INT8 value to inject
    """
    nodes = [
        # 1. Create zeros same shape as output
        helper.make_node("Constant",
            inputs=[],
            outputs=["zeros"],
            value=helper.make_tensor(
                name="const_zeros",
                data_type=TensorProto.INT8,
                dims=output_shape,
                vals=[0] * np.prod(output_shape)
            )
        ),
        
        # 2. Create indices for target position
        helper.make_node("Constant",
            inputs=[],
            outputs=["indices"],
            value=helper.make_tensor(
                name="const_indices",
                data_type=TensorProto.INT64,
                dims=[1, len(target_indices)],
                vals=target_indices
            )
        ),
        
        # 3. Create random value
        helper.make_node("Constant",
            inputs=[],
            outputs=["random_value"],
            value=helper.make_tensor(
                name="const_random",
                data_type=TensorProto.INT8,
                dims=[1],
                vals=[random_value]
            )
        ),
        
        # 4. Create mask using ScatterND
        helper.make_node("ScatterND",
            inputs=["zeros", "indices", "random_value"],
            outputs=["random_tensor"]
        ),
        
        # 5. Cast for addition
        helper.make_node("Cast",
            inputs=[output_name],
            outputs=["output_int32"],
            to=TensorProto.INT32
        ),
        helper.make_node("Cast",
            inputs=["random_tensor"],
            outputs=["random_int32"],
            to=TensorProto.INT32
        ),
        
        # 6. Add as INT32
        helper.make_node("Add",
            inputs=["output_int32", "random_int32"],
            outputs=["sum_int32"]
        ),
        
        # 7. Cast back to INT8 for final output
        helper.make_node("Cast",
            inputs=["sum_int32"],
            outputs=[output_name],
            to=TensorProto.INT8
        )
    ]
    return nodes

def create_float16_fault_injection(input_name, input_shape, target_indices, bit_position):
    nodes = [
        # 1. Cast input to uint16 for bit manipulation
        helper.make_node("Cast",
            inputs=[input_name],
            outputs=["input_uint16"],
            to=TensorProto.UINT16
        ),

        # 2. Create zeros tensor with same shape as input
        helper.make_node("Constant",
            inputs=[],
            outputs=["zeros"],
            value=helper.make_tensor(
                name="const_zeros",
                data_type=TensorProto.UINT16,
                dims=input_shape,
                vals=np.zeros(input_shape, dtype=np.uint16).tobytes()
            )
        ),

        # 3. Create target_indices tensor
        helper.make_node("Constant",
            inputs=[],
            outputs=["target_indices"],
            value=helper.make_tensor(
                name="const_indices",
                data_type=TensorProto.INT64,
                dims=[len(target_indices)],
                vals=target_indices
            )
        ),

        # 4. Create the mask value as a 1D tensor matching indices rank
        helper.make_node("Constant",
            inputs=[],
            outputs=["mask_value"],
            value=helper.make_tensor(
                name="const_mask",
                data_type=TensorProto.UINT16,
                dims=[1],  # 1D to match target_indices
                vals=[1 << bit_position]  # bit mask for float16
            )
        ),

        # 5. Place mask value at target indices in zeros tensor
        helper.make_node("ScatterElements",
            inputs=["zeros", "target_indices", "mask_value"],
            outputs=["mask_tensor"],
            axis=0
        ),

        # 6. BitWise XOR input with mask tensor
        helper.make_node("BitwiseXor",
            inputs=["input_uint16", "mask_tensor"],
            outputs=["output_uint16"]
        ),

        # 7. Cast back to float16
        helper.make_node("Cast",
            inputs=["output_uint16"],
            outputs=["output_tensor"],
            to=TensorProto.FLOAT16
        )
    ]

    return nodes