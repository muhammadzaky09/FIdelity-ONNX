import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

def create_random_fault_injection(output_name: str, random_value: float):
    """
    Modified implementation of random fault injection that works with 3D tensors.
    
    Args:
        output_name: Name of the tensor to modify
        random_value: Value to inject at the random position
        
    Returns:
        List of ONNX nodes for fault injection
    """
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
    
    # 3. Generate random indices using RandomUniformLike (doesn't need explicit shape)
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
    
    # 7. Create the dimensions for reshape
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['reshape_dims' + suffix],
        value=helper.make_tensor(
            name='reshape_dims_tensor' + suffix,
            data_type=TensorProto.INT64,
            dims=[2],
            vals=[1, 3]  # For a 3D tensor - reshape to [1, 3]
        )
    ))
    
    # 8. Reshape the indices to be compatible with ScatterND
    nodes.append(helper.make_node(
        'Reshape',
        inputs=['indices_int64' + suffix, 'reshape_dims' + suffix],
        outputs=['indices_reshaped' + suffix]
    ))
    
    # 9. Create a constant for the fault value (FP16)
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
    
    # 10. Use ScatterND to inject the fault
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=[output_name, 'indices_reshaped' + suffix, 'fault_value' + suffix],
        outputs=[f'{output_name}_faulty']
    ))
    
    return nodes  # This was missing!

def test_fp16_fault_injection_3d():
    # Create a 3D tensor with FP16 data
    shape = [3, 4, 5]  # [channels, height, width]
    data = np.ones(shape, dtype=np.float16) * 1.5
    
    # Create ONNX graph inputs and outputs
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT16, shape)
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT16, shape)
    faulty_output = helper.make_tensor_value_info('output_faulty', TensorProto.FLOAT16, shape)
    
    # Create a simple identity op to pass through the input
    identity_node = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['output']
    )
    
    # Create fault injection nodes - inject value 42.0
    fault_value = 42.0
    fault_nodes = create_random_fault_injection('output', fault_value)
    
    # Create the graph with all nodes
    graph = helper.make_graph(
        [identity_node] + fault_nodes,
        'test_fault_injection_3d',
        [input_tensor],
        [output_tensor, faulty_output]
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 14  # Use a compatible opset version
    
    # Check and save the model
    onnx.checker.check_model(model)
    model_path = 'test_fault_injection_3d.onnx'
    onnx.save(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Run inference
    session = ort.InferenceSession(model_path)
    outputs = session.run(['output', 'output_faulty'], {'input': data})
    
    original = outputs[0]
    faulty = outputs[1]
    
    # Find where the fault was injected (where the tensors differ)
    diff = faulty - original
    fault_indices = np.where(np.abs(diff) > 0)
    
    print(f"Original tensor shape: {original.shape}")
    print(f"Original tensor dtype: {original.dtype}")
    print(f"Original tensor sample: {original.flatten()[:5]}")
    
    if len(fault_indices[0]) > 0:
        # Convert the indices to a tuple for indexing
        idx = tuple(i[0] for i in fault_indices)
        
        print(f"\nFault injected at position: {idx}")
        print(f"Original value at position: {original[idx]}")
        print(f"Faulty value at position: {faulty[idx]}")
        print(f"Injected fault value: {fault_value}")
        
        # Print a small slice of the tensor around the fault location
        slices = []
        for i, pos in enumerate(idx):
            start = max(0, pos - 1)
            end = min(original.shape[i], pos + 2)
            slices.append(slice(start, end))
        
        print("\nOriginal tensor slice around fault:")
        print(original[tuple(slices)])
        
        print("\nFaulty tensor slice around fault:")
        print(faulty[tuple(slices)])
    else:
        print("No difference found between original and faulty tensors!")

if __name__ == "__main__":
    test_fp16_fault_injection_3d()