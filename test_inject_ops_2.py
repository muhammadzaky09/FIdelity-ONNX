# import numpy as np
# import onnx
# from onnx import helper, TensorProto, numpy_helper
# import onnxruntime as ort

# import numpy as np
# import onnx
# from onnx import helper, TensorProto, numpy_helper
# import onnxruntime as ort

# def create_quantized_fault_injection(bit_position):
#     input_name = "input"
#     output_name = "output"
    
#     nodes = []
    
#     # Get input shape
#     nodes.append(helper.make_node('Shape',
#             inputs=[input_name],
#             outputs=['runtime_shape']
#             ))
    
#     # Create split sizes tensor
#     split_tensor = helper.make_tensor(
#         name='split_sizes',
#         data_type=TensorProto.INT64,
#         dims=[3],
#         vals=[1, 1, 1]  # Split into 3 parts of size 1
#     )
    
#     # Split runtime_shape into individual dimensions
#     nodes.append(helper.make_node('Split',
#             inputs=['runtime_shape', 'split_sizes'],
#             outputs=['dim0', 'dim1', 'dim2'],
#             axis=0
#             ))
    
#     # Generate three random values
#     nodes.append(helper.make_node('RandomUniform',
#             inputs=[],
#             outputs=['random_vals'],
#             dtype=TensorProto.FLOAT,
#             high=1.0,
#             low=0.0,
#             shape=[1, 3]  # One row, three columns for coordinates
#             ))
    
#     # Cast dimensions to float
#     for i in range(3):
#         nodes.append(helper.make_node('Cast',
#                 inputs=[f'dim{i}'],
#                 outputs=[f'dim{i}_float'],
#                 to=TensorProto.FLOAT
#             ))
    
#     # Scale random values
#     nodes.append(helper.make_node('Concat',
#             inputs=['dim0_float', 'dim1_float', 'dim2_float'],
#             outputs=['concat_dims'],
#             axis=0
#         ))
#     nodes.append(helper.make_node(
#         'Constant',
#         inputs=[],
#         outputs=['unsqueeze_axes'],
#         value=helper.make_tensor(
#             name='const_unsqueeze_axes',
#             data_type=TensorProto.INT64,
#             dims=[1],
#             vals=[0]  # This means "insert a new axis at position 0"
#         )
#     ))
#     # Unsqueeze for broadcasting
#     nodes.append(helper.make_node(
#         'Unsqueeze',
#         inputs=['concat_dims', 'unsqueeze_axes'],  # Now supplying both required inputs
#         outputs=['dims_unsqueezed']
#     ))
    
#     # Scale random values by dimensions
#     nodes.append(helper.make_node('Mul',
#             inputs=['random_vals', 'dims_unsqueezed'],
#             outputs=['scaled_indices']
#         ))
    
#     # Floor and cast to INT64
#     nodes.append(helper.make_node('Floor',
#             inputs=['scaled_indices'],
#             outputs=['floored_indices']
#         ))
    
#     nodes.append(helper.make_node('Cast',
#             inputs=['floored_indices'],
#             outputs=['indices_int64'],
#             to=TensorProto.INT64
#         ))
    
#     # Cast input to INT8
#     nodes.append(helper.make_node('Cast', 
#             inputs=[input_name], 
#             outputs=['int8_val'], 
#             to=TensorProto.INT8
#         ))
    
#     # Create bitmask
#     nodes.append(helper.make_node('Constant',
#             inputs=[],
#             outputs=['bitmask'],
#             value=helper.make_tensor(
#                 name='bitmask_val',
#                 data_type=TensorProto.INT8,
#                 dims=[1],
#                 vals=[1 << bit_position]
#             )))
    
#     # Create zero tensor
#     nodes.append(helper.make_node('ConstantOfShape',
#             inputs=['runtime_shape'],
#             outputs=['zero_base'],
#             value=helper.make_tensor(
#                 name='zero_value',
#                 data_type=TensorProto.INT8,
#                 dims=[1],
#                 vals=[0]
#             )))
    
#     # Use ScatterND
#     nodes.append(helper.make_node('ScatterND',
#             inputs=['zero_base', 'indices_int64', 'bitmask'],
#             outputs=['bit_mask']))
    
#     # Perform bit flip
#     nodes.append(helper.make_node('BitwiseXor',
#             inputs=['int8_val', 'bit_mask'],
#             outputs=['flipped_int']))
    
#     # Cast to INT32 for subtraction
#     nodes.append(helper.make_node('Cast',
#             inputs=['flipped_int'],
#             outputs=['flipped_int32'],
#             to=TensorProto.INT32
#         ))
    
#     nodes.append(helper.make_node('Cast',
#             inputs=['int8_val'],
#             outputs=['int8_val32'],
#             to=TensorProto.INT32
#         ))
    
#     # Calculate perturbation
#     nodes.append(helper.make_node('Sub',
#             inputs=['flipped_int32', 'int8_val32'],
#             outputs=['perturbation_int32']
#         ))
    
#     # Final cast to float
#     nodes.append(helper.make_node('Cast',
#             inputs=['perturbation_int32'],
#             outputs=[output_name],
#             to=TensorProto.FLOAT))

#     inputs = [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ['batch', 'sequence', 'hidden'])]
#     outputs = [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ['batch', 'sequence', 'hidden'])]
    
#     graph = helper.make_graph(
#         nodes=nodes,
#         name='fault_injection_test',
#         inputs=inputs,
#         outputs=outputs,
#         initializer=[split_tensor]  # Add split_tensor as initializer
#     )
    
#     model = helper.make_model(graph)
#     model.opset_import[0].version = 18
    
#     return model

# # For testing
# if __name__ == "__main__":
#     model = create_quantized_fault_injection(bit_position=3)
#     onnx.checker.check_model(model)
#     session = ort.InferenceSession(model.SerializeToString(), providers=['CUDAExecutionProviders'])
    
#     # Test with sample input
#     test_input = np.random.randn(1, 2, 3).astype(np.float32)
#     outputs = session.run(None, {"input": test_input})
#     perturbation = outputs[0]
    
#     print(f"Input shape: {test_input.shape}")
#     print(f"Perturbation shape: {perturbation.shape}")
#     print(f"Number of modified elements: {np.count_nonzero(perturbation)}")
#     print(perturbation)

import numpy as np
import onnx
from onnx import helper, TensorProto

def create_simplified_quantized_fault_injection(bit_position):
    input_name = "input"
    output_name = "output"
    
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
        outputs=[output_name],
        to=TensorProto.FLOAT
    ))
    
    # -------------------------------
    # 3. Define Graph Inputs and Outputs
    # -------------------------------
    
    inputs = [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ['batch', 'sequence', 'hidden'])]
    outputs = [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, ['batch', 'sequence', 'hidden'])]
    
    # Create the graph.
    graph = helper.make_graph(
        nodes=nodes,
        name='simplified_fault_injection_test',
        inputs=inputs,
        outputs=outputs
    )
    
    # Create the model, setting an opset that supports our operators (e.g. opset 18).
    model = helper.make_model(graph)
    model.opset_import[0].version = 18
    
    return model

# For testing:
if __name__ == "__main__":
    model = create_simplified_quantized_fault_injection(bit_position=1)
    onnx.checker.check_model(model)
    import onnxruntime as ort
    session = ort.InferenceSession(model.SerializeToString(), providers=['CUDAExecutionProvider'])
    
    # Test with a sample input tensor, e.g. shape (1, 2, 3)
    test_input = np.random.randn(1, 2, 3).astype(np.float32)
    print(test_input)
    outputs = session.run(None, {"input": test_input})
    perturbation = outputs[0]
    
    print("Input shape:", test_input.shape)
    print("Perturbation shape:", perturbation.shape)
    print("Number of modified elements:", np.count_nonzero(perturbation))
    print(perturbation)

from onnx import helper, TensorProto

def create_quantized_fault_injection16(input_name, output_name, block_axis, input_dtype, bit_position):
    nodes = []
    # 1. Get runtime shape of the input tensor.
    nodes.append(helper.make_node(
        'Shape',
        inputs=[input_name],
        outputs=['runtime_shape']
    ))
    
    # 2. Get the size of the dimension along the fault injection axis.
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['axis_idx'],
        value=helper.make_tensor(
            name='axis_val',
            data_type=TensorProto.INT64,
            dims=[],  # scalar
            vals=[block_axis if block_axis >= 0 else -1]  # (Note: negative handling may be improved as needed)
        )
    ))
    nodes.append(helper.make_node(
        'Gather',
        inputs=['runtime_shape', 'axis_idx'],
        outputs=['dim_size'],
        axis=0
    ))
    
    # 3. Define constants: 16, 0, and 1.
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['sixteen'],
        value=helper.make_tensor(
            name='sixteen_val',
            data_type=TensorProto.INT64,
            dims=[], 
            vals=[16]
        )
    ))
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['zero'],
        value=helper.make_tensor(
            name='zero_val',
            data_type=TensorProto.INT64,
            dims=[], 
            vals=[0]
        )
    ))
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['one'],
        value=helper.make_tensor(
            name='one_val',
            data_type=TensorProto.INT64,
            dims=[], 
            vals=[1]
        )
    ))
    
    # 4. Calculate the number of 16-element blocks along the fault axis.
    nodes.append(helper.make_node(
        'Div',
        inputs=['dim_size', 'sixteen'],
        outputs=['num_blocks']
    ))
    # Generate a random float scalar in [0, 1)
    nodes.append(helper.make_node(
        'RandomUniform',
        inputs=[],  # no inputs required; shape is provided as an attribute
        outputs=['random_block_float'],
        dtype=TensorProto.FLOAT,
        low=0.0,
        high=1.0,
        shape=[1]
    ))
    # Scale the random number by num_blocks
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_block_float', 'num_blocks'],
        outputs=['scaled_random_block']
    ))
    # Cast the scaled random float to INT64 to select a block index
    nodes.append(helper.make_node(
        'Cast',
        inputs=['scaled_random_block'],
        outputs=['random_block'],
        to=TensorProto.INT64
    ))
    # Multiply the block index by 16 to get the start index of the block.
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_block', 'sixteen'],
        outputs=['start_idx']
    ))
    
    # 5. Determine block length:
    #    For INPUT16 (block_axis == -1) we use a fixed block length of 16;
    #    For WEIGHT16 (block_axis != -1) we generate a random block length in [1,16).
    if block_axis == -1:  # INPUT16 fault model: fixed 16-element block.
        nodes.append(helper.make_node(
            'Constant',
            inputs=[],
            outputs=['block_len'],
            value=helper.make_tensor(
                name='block_len_val',
                data_type=TensorProto.INT64,
                dims=[],
                vals=[16]
            )
        ))
    else:  # WEIGHT16 fault model: random block length between 1 and 16.
        nodes.append(helper.make_node(
            'RandomUniform',
            inputs=[],
            outputs=['random_len_float'],
            dtype=TensorProto.FLOAT,
            low=1.0,
            high=16.0,
            shape=[1]
        ))
        nodes.append(helper.make_node(
            'Cast',
            inputs=['random_len_float'],
            outputs=['block_len'],
            to=TensorProto.INT64
        ))
    
    # 6. Create bitmask for bit flipping.
    bitmask_value = 1 << bit_position  # e.g. for bit_position 3, this is 8.
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['bitmask'],
        value=helper.make_tensor(
            name='bitmask_val',
            data_type=TensorProto.INT8,
            dims=[],  # scalar
            vals=[bitmask_value]
        )
    ))
    
    # 7. Gather a block element along the specified axis and perform the bit flip.
    nodes.append(helper.make_node(
        'Gather',
        inputs=[input_name, 'start_idx'],
        outputs=['gathered_val'],
        axis=block_axis
    ))
    
    # For subtraction, we need to work with INT32 values.
    if input_dtype == TensorProto.FLOAT:
        # Cast the gathered FLOAT value to INT8, then flip the bit.
        nodes.append(helper.make_node(
            'Cast',
            inputs=['gathered_val'],
            outputs=['gathered_int8'],
            to=TensorProto.INT8
        ))
        nodes.append(helper.make_node(
            'BitwiseXor',
            inputs=['gathered_int8', 'bitmask'],
            outputs=['flipped_int8']
        ))
        # Cast the flipped result back to FLOAT.
        nodes.append(helper.make_node(
            'Cast',
            inputs=['flipped_int8'],
            outputs=['flipped_val'],
            to=TensorProto.FLOAT
        ))
        # Also cast both gathered and flipped values to INT32 for subtraction.
        nodes.append(helper.make_node(
            'Cast',
            inputs=['gathered_val'],
            outputs=['gathered_val_int32'],
            to=TensorProto.INT32
        ))
        nodes.append(helper.make_node(
            'Cast',
            inputs=['flipped_val'],
            outputs=['flipped_val_int32'],
            to=TensorProto.INT32
        ))
        nodes.append(helper.make_node(
            'Sub',
            inputs=['flipped_val_int32', 'gathered_val_int32'],
            outputs=['delta_scalar']
        ))
    else:
        # Assume the input is already an INT8 type.
        nodes.append(helper.make_node(
            'BitwiseXor',
            inputs=['gathered_val', 'bitmask'],
            outputs=['flipped_int8']
        ))
        # Cast the flipped value to FLOAT.
        nodes.append(helper.make_node(
            'Cast',
            inputs=['flipped_int8'],
            outputs=['flipped_val'],
            to=TensorProto.FLOAT
        ))
        # Cast both gathered and flipped values to INT32.
        nodes.append(helper.make_node(
            'Cast',
            inputs=['gathered_val'],
            outputs=['gathered_val_int32'],
            to=TensorProto.INT32
        ))
        nodes.append(helper.make_node(
            'Cast',
            inputs=['flipped_val'],
            outputs=['flipped_val_int32'],
            to=TensorProto.INT32
        ))
        nodes.append(helper.make_node(
            'Sub',
            inputs=['flipped_val_int32', 'gathered_val_int32'],
            outputs=['delta_scalar']
        ))
    
    # 8. Unsqueeze the delta along the block axis so it can be broadcast.
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['delta_scalar'],
        outputs=['delta_unsqueezed'],
        axes=[block_axis]
    ))
    
    # 9. Expand the delta to the full input shape.
    nodes.append(helper.make_node(
        'Expand',
        inputs=['delta_unsqueezed', 'runtime_shape'],
        outputs=['expanded_delta']
    ))
    
    # 10. Create indices for the block mask:
    #     Generate a range [0, dim_size) along the fault axis.
    nodes.append(helper.make_node(
        'Range',
        inputs=['zero', 'dim_size', 'one'],
        outputs=['indices']
    ))
    
    # 11. Create the block mask:
    #     Compute the upper bound of the block: start_idx + block_len.
    nodes.append(helper.make_node(
        'Add',
        inputs=['start_idx', 'block_len'],
        outputs=['upper_bound']
    ))
    nodes.append(helper.make_node(
        'GreaterOrEqual',
        inputs=['indices', 'start_idx'],
        outputs=['mask_lower']
    ))
    nodes.append(helper.make_node(
        'Less',
        inputs=['indices', 'upper_bound'],
        outputs=['mask_upper']
    ))
    nodes.append(helper.make_node(
        'And',
        inputs=['mask_lower', 'mask_upper'],
        outputs=['mask_bool']
    ))
    
    # 12. Create axes for unsqueezing the mask:
    #     Get the rank (number of dimensions) of the runtime shape.
    nodes.append(helper.make_node(
        'Shape',
        inputs=['runtime_shape'],
        outputs=['shape_len']
    ))
    # Create a range [0, rank) to represent all axes.
    nodes.append(helper.make_node(
        'Range',
        inputs=['zero', 'shape_len', 'one'],
        outputs=['all_axes']
    ))
    nodes.append(helper.make_node(
        'NonZero',
        inputs=['all_axes'],
        outputs=['axes_nonzero']
    ))
    nodes.append(helper.make_node(
        'Squeeze',
        inputs=['axes_nonzero'],
        outputs=['unsqueeze_axes'],
        axes=[0]
    ))
    
    # 13. Unsqueeze and expand the mask to the full input shape.
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['mask_bool', 'unsqueeze_axes'],
        outputs=['mask_expanded']
    ))
    
    # 14. Create a zero tensor of the same shape as the input.
    nodes.append(helper.make_node(
        'ConstantOfShape',
        inputs=['runtime_shape'],
        outputs=['zero_tensor'],
        value=helper.make_tensor(
            name='const_value',
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[0.0]
        )
    ))
    
    # 15. Use the mask to select where to apply the delta:
    #     Where(mask, expanded_delta, zero_tensor) produces the final delta tensor.
    nodes.append(helper.make_node(
        'Where',
        inputs=['mask_expanded', 'expanded_delta', 'zero_tensor'],
        outputs=[output_name]
    ))
    
    

