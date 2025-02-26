import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto
# Make sure this function is importable from your module, e.g.:
# from inject_ops import create_random_fault_injection

def create_random_fault_injection(output_name: str, random_value: float):
    from onnx import helper, TensorProto
    import numpy as np
    nodes = []
    
    # 1. Get the runtime shape of the tensor.
    # For a tensor of shape [1, N, 4096], this returns a tensor [1, N, 4096] (a vector of 3 elements).
    nodes.append(helper.make_node(
        'Shape',
        inputs=[output_name],
        outputs=['runtime_shape']
    ))
    
    # 2. Generate random uniform values of shape [3].
    # This produces a 1D tensor of 3 elements.
    nodes.append(helper.make_node(
        'RandomUniform',
        inputs=[],  # shape provided as attribute.
        outputs=['random_vals'],
        dtype=TensorProto.FLOAT,
        low=0.0,
        high=1.0,
        shape=[3]
    ))
    
    # 3. Cast the runtime shape from INT64 to FLOAT so that we can multiply.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['runtime_shape'],
        outputs=['runtime_shape_float'],
        to=TensorProto.FLOAT
    ))
    
    # 4. Multiply the random values by the runtime shape.
    # This scales the random values into the valid index ranges.
    nodes.append(helper.make_node(
        'Mul',
        inputs=['random_vals', 'runtime_shape_float'],
        outputs=['scaled_random']
    ))
    
    # 5. Floor the scaled random values to get integer indices.
    nodes.append(helper.make_node(
        'Floor',
        inputs=['scaled_random'],
        outputs=['floored_random']
    ))
    
    # 6. Cast the floored values to INT64.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_random'],
        outputs=['random_indices_raw'],
        to=TensorProto.INT64
    ))
    
    # 7. Unsqueeze the random indices so that their shape becomes [1,3],
    # as ScatterND requires indices shape to be [num_updates, rank(data)].
    nodes.append(helper.make_node(
        'Unsqueeze',
        inputs=['random_indices_raw'],
        outputs=['random_indices'],
        axes=[0]
    ))
    
    # 8. Create a constant node for the fault value in FP16.
    fault_val = np.array([random_value], dtype=np.float16).tolist()
    nodes.append(helper.make_node(
        'Constant',
        inputs=[],
        outputs=['random_value'],
        value=helper.make_tensor(
            name='const_random',
            data_type=TensorProto.FLOAT16,
            dims=[1],
            vals=fault_val
        )
    ))
    
    # 9. Use ScatterND to inject the fault value into the tensor at the computed random index.
    nodes.append(helper.make_node(
        'ScatterND',
        inputs=[output_name, 'random_indices', 'random_value'],
        outputs=[f'{output_name}_faulty']
    ))
    
    return nodes





def build_and_run_onnx_model(nodes, input_shape=(1,3,4), input_dtype=np.float16):
    """
    Build an ONNX model with a single input and a single output.
    The model's input name is "input_tensor" and the output is produced by your fault injection
    subgraph (which appends "_faulty" to the input name).
    """
    input_name = "input_tensor"
    output_name = f"{input_name}_faulty"
    
    # Create the graph.
    graph = helper.make_graph(
        nodes=nodes,
        name="RandomFaultInjectionGraph",
        inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, list(input_shape))],
        outputs=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, list(input_shape))]
    )
    
    # Create the model.
    model = helper.make_model(graph, producer_name="RandomFaultInjectionTest")
    onnx.checker.check_model(model)
    
    # Run inference using ONNX Runtime.
    sess = ort.InferenceSession(model.SerializeToString())
    input_data = np.random.rand(*input_shape).astype(input_dtype)
    output = sess.run(None, {input_name: input_data})[0]
    return input_data, output

if __name__ == "__main__":
    # Choose a random fault value.
    random_fault_value = 0.1
    # We'll use "input_tensor" as both the input and as the argument to create_random_fault_injection.
    nodes = create_random_fault_injection("input_tensor", random_value=random_fault_value)
    
    # Build and run the model.
    input_data, output = build_and_run_onnx_model(nodes, input_shape=(1,3,4), input_dtype=np.float16)
    
    print("Input Data:")
    print(input_data)
    print("\nFaulty Output:")
    print(output)
    
    nonzero_indices = np.argwhere(output != 0)
    nonzero_values = output[output != 0]
    print("\nNonzero Indices:")
    print(nonzero_indices)
    print("\nNonzero Values:")
    print(nonzero_values)
