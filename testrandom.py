import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper

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


def build_test_model():
    # Create a simple model:
    # Input: "x" with shape [1,10]
    # Weight: identity matrix of shape [10,10]
    # MatMul: produces "y" which is roughly equal to x (for testing)
    # Then apply random fault injection on y.
    input_tensor = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 10, 10])
    # Identity weight
    weight_array = np.eye(10, dtype=np.float16)
    weight = helper.make_tensor("W", TensorProto.FLOAT16, [10, 10], weight_array.flatten().tolist())
    weight_node = helper.make_node("Constant", inputs=[], outputs=["W"], value=weight)
    matmul_node = helper.make_node("MatMul", inputs=["x", "W"], outputs=["y"], name="MatMul_Node")
    
    # Random fault injection subgraph on y.
    fault_nodes = create_random_fault_injection("y", 7.0)  # injecting fault value 7.0
    # The fault injection produces "y_faulty"
    
    output_tensor = helper.make_tensor_value_info("y_faulty", TensorProto.FLOAT16, [1, 10, 10])
    
    nodes = [weight_node, matmul_node] + fault_nodes
    
    graph = helper.make_graph(nodes, "TestGraph", [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="RandomFaultInjectionTest")
    onnx.checker.check_model(model)
    return model

if __name__ == "__main__":
    model = build_test_model()
    onnx.save(model, "random_fault_test.onnx")
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.random.randn(1,10, 10).astype(np.float16)
    y = sess.run(None, {"x": x})[0]
    print("Input x:")
    print(x)
    print("Output y_faulty (one element replaced by fault value):")
    print(y)
