import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

# Assume your create_quantized_fault_injection function is defined as given:
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
    
    # Cast the whole runtime_shape to FLOAT.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['runtime_shape'],
        outputs=['runtime_shape_float'],
        to=TensorProto.FLOAT
    ))
    
    # Generate random uniform values with shape [3].
    nodes.append(helper.make_node(
        'RandomUniform',
        inputs=[],
        outputs=['random_vals'],
        dtype=TensorProto.FLOAT,
        high=1.0,
        low=0.0,
        shape=[3]
    ))
    
    # Multiply random values with runtime_shape_float.
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
    
    # Cast to INT64.
    nodes.append(helper.make_node(
        'Cast',
        inputs=['floored_indices'],
        outputs=['indices_int64'],
        to=TensorProto.INT64
    ))
    
    # -------------------------------
    # 2. Fault Injection Operations
    # -------------------------------
    
    # Cast the input to INT8.
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
            dims=[],  # scalar
            vals=[1 << bit_position]
        )
    ))
    
    # Create a zero tensor of the same shape using ConstantOfShape.
    nodes.append(helper.make_node(
        'ConstantOfShape',
        inputs=['runtime_shape'],  # use the same shape as the input
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
    
    # Apply BitwiseXor.
    nodes.append(helper.make_node(
        'BitwiseXor',
        inputs=['int8_val', 'bit_mask'],
        outputs=['flipped_int']
    ))
    
    # Cast both flipped and original INT8 values to INT32.
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
    
    # Subtract to compute perturbation.
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
    return nodes

def create_test_model(bit_position=3):
    # Define a 3D input tensor, e.g., [1, 4, 8]
    input_info = helper.make_tensor_value_info("input_tensor", TensorProto.FLOAT, [1, 4, 8])
    output_info = helper.make_tensor_value_info("perturbation_output", TensorProto.FLOAT, [1, 4, 8])
    
    # Create an Identity node so that the input flows into our subgraph.
    identity_node = helper.make_node("Identity", inputs=["input_tensor"], outputs=["id_out"])
    
    # Build the injection subgraph for input (this function is for 3D input).
    injection_nodes = create_quantized_fault_injection("id_out", "perturbation_output", bit_position)
    
    # Create the graph.
    graph = helper.make_graph(
        nodes=[identity_node] + injection_nodes,
        name="TestInjectionGraph",
        inputs=[input_info],
        outputs=[output_info]
    )
    
    model = helper.make_model(graph)
    onnx.save(model, "test_injection.onnx")
    return "test_injection.onnx"

def run_test_model(model_path):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    # Create a random input tensor with shape [1, 4, 8]
    x = np.random.rand(1, 4, 8).astype(np.float32)
    output_name = sess.get_outputs()[0].name
    outputs = sess.run([output_name], {"input_tensor": x})
    print("Perturbation output shape:", outputs[0].shape)
    print("Perturbation output (first 10 values):", outputs[0].flatten()[:10])

if __name__ == "__main__":
    model_path = create_test_model(bit_position=3)
    run_test_model(model_path)
