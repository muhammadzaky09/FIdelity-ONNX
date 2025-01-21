import numpy as np
import onnx
import onnxruntime
from onnx import helper, TensorProto
from inject_ops import create_random_int8_fault_injection

def test_random_int8_fault_injection():
    """Test random fault injection on simple 2x2 INT8 tensor."""
    input_shape = [1, 512, 2048]
    target_indices = [0, 0, 256]  # Target position [0,1]
    random_value = 42
    input_name = "test_input"
    
    # Create nodes
    nodes = create_random_int8_fault_injection(
        input_name=input_name,
        input_shape=input_shape,
        target_indices=target_indices,
        random_value=random_value
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=nodes,
        name="test_graph",
        inputs=[helper.make_tensor_value_info(
            input_name, TensorProto.INT8, input_shape
        )],
        outputs=[helper.make_tensor_value_info(
            "output_tensor", TensorProto.INT8, input_shape
        )],
        initializer=[]
    )
    
    # Create and run model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    
    # Test input
    test_input = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
    session = onnxruntime.InferenceSession(model.SerializeToString())
    output = session.run(None, {input_name: test_input})[0]
    
    print("Input:", test_input)
    print("Output:", output)
    print("Injected fault at position", target_indices, "with value", random_value)

if __name__ == "__main__":
    test_random_int8_fault_injection()