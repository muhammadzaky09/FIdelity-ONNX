import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

from onnx import helper, TensorProto

def create_quantized_fault_injection_weight(input_name, output_name, bit_position):
    """
    Injection subgraph for a weight tensor (assumed 2D) in FLOAT16.

    The pipeline is as follows:
      1. Cast the weight from FLOAT16 to FLOAT32.
      2. Dynamically extract its shape.
      3. Generate a random index vector (of length 2) and multiply by the shape.
      4. Floor and cast the result to INT64.
      5. Unsqueeze the index (so it becomes shape [1,2]) as required by ScatterND.
      6. Cast the promoted weight (float32) to INT8.
      7. Create a constant bitmask (int8) with shape [1] containing the desired bit flipped.
      8. Create a zero tensor (of the same shape) using ConstantOfShape.
      9. Use ScatterND to scatter the bitmask into the zero tensor at the computed index.
     10. Apply BitwiseXor between the int8 weight and the bit_mask.
     11. Cast both the flipped and original int8 values to INT32 and subtract to obtain the perturbation.
     12. Cast the perturbation first to FLOAT32 and then to FLOAT16.

    The final output (a perturbation tensor in FLOAT16) is produced in output_name.
    """
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



# Create a minimal model with a constant weight in FP16.
def create_test_model():
    # Create a random weight tensor in FP16 (for example, shape [4096, 4096]).
    weight_np = np.random.randn(4096, 4096).astype(np.float16)
    # Build a Constant node for the weight.
    weight_const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["weight_const_out"],
        value=helper.make_tensor(
            name="weight_const_tensor",
            data_type=TensorProto.FLOAT16,
            dims=weight_np.shape,
            vals=weight_np.flatten().tolist()
        )
    )
    # For testing, simply create an Identity node so that the weight passes through.
    identity = helper.make_node("Identity", inputs=["weight_const_out"], outputs=["model_output"])
    # Build graph.
    graph = helper.make_graph(
        [weight_const, identity],
        "test_graph",
        inputs=[],  # no external inputs
        outputs=[helper.make_tensor_value_info("model_output", TensorProto.FLOAT16, [4096, 4096])]
    )
    model = helper.make_model(graph)
    onnx.save(model, "test_model.onnx")
    return "test_model.onnx"

# Insert the injection subgraph into the model.
def insert_injection_subgraph(model_path, injected_output_name, bit_position=3):
    model = onnx.load(model_path)
    # Assume the weight constant node is our injection source; its output is "weight_const_out"
    injection_nodes = create_quantized_fault_injection_weight("weight_const_out", injected_output_name, bit_position)
    # Append injection nodes to the graph.
    model.graph.node.extend(injection_nodes)
    # For testing, set the model's output to be the injection subgraph output.
    model.graph.output[0].name = injected_output_name
    onnx.save(model, "test_model_injected.onnx")
    return "test_model_injected.onnx"

# Run inference using ONNX Runtime.
def run_inference(model_path):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    output_name = sess.get_outputs()[0].name
    # Our model is fully constant, so no inputs are needed.
    outputs = sess.run([output_name], {})
    print("Output shape:", outputs[0].shape)
    print("Output sample (first 10 elements):", outputs[0].flatten()[:10])
    print("nonzero indices: ",np.argwhere(outputs[0] != 0))
    print("nonzero values", outputs[0][outputs[0] != 0] )

if __name__ == "__main__":
    # Create the test model.
    test_model_path = create_test_model()
    # Insert the injection subgraph; the output tensor is named "perturbation_output".
    injected_model_path = insert_injection_subgraph(test_model_path, "perturbation_output", bit_position=5)
    # Run inference on the modified model.
    run_inference(injected_model_path)
