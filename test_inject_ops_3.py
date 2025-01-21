import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto, GraphProto
from inject_ops import create_int8_fault_injection

def create_fault_layer(layer_graph: GraphProto, injection_graph: GraphProto, 
                      target_tensor_name: str) -> GraphProto:
    """Merge injection nodes into layer graph at target tensor."""
    
    # Validate target tensor exists in graph
    def find_tensor_in_graph(graph, tensor_name):
        for node in graph.node:
            if tensor_name in node.input or tensor_name in node.output:
                return True
        for tensor in graph.input + graph.output:
            if tensor_name == tensor.name:
                return True
        return False
        
    if not find_tensor_in_graph(layer_graph, target_tensor_name):
        raise ValueError(f"Target tensor {target_tensor_name} not found in graph")
        
    if len(injection_graph.input) != 1 or len(injection_graph.output) != 1:
        raise ValueError("Injection graph must have exactly one input and output")
    
    # Create mapping
    io_map = [(target_tensor_name, injection_graph.input[0].name)]
    
    # Merge graphs
    merged_graph = onnx.compose.merge_graphs(
        g1=layer_graph,
        g2=injection_graph,
        io_map=io_map,
        inputs=[input.name for input in layer_graph.input],
        outputs=[injection_graph.output[0].name]
    )
    
    return merged_graph

def test_fault_injection_pipeline():
    """End-to-end test of fault injection graph merging"""
    
    # 1. Base graph with proper type casting
    base_nodes = [
        helper.make_node("Cast",
            inputs=["float_input"],
            outputs=["int8_pre_add"],
            to=TensorProto.INT8
        ),
        helper.make_node("Cast", 
            inputs=["int8_pre_add"],
            outputs=["float_for_add"],
            to=TensorProto.FLOAT
        ),
        helper.make_node("Add",
            inputs=["float_for_add", "float_for_add"],
            outputs=["float_result"]
        ),
        helper.make_node("Cast",
            inputs=["float_result"],
            outputs=["add_output"],
            to=TensorProto.INT8
        )
    ]
    
    input_shape = [1, 3, 4, 4]
    base_graph = helper.make_graph(
        base_nodes,
        "test_base",
        [helper.make_tensor_value_info("float_input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("add_output", TensorProto.INT8, input_shape)]
    )

    # Save base graph
    base_model = helper.make_model(base_graph)
    base_model.opset_import[0].version = 18
    onnx.save(base_model, "base_graph.onnx")

    # 2. Injection graph
    injection_nodes = create_int8_fault_injection(
        "add_output",
        input_shape, 
        [0,1,2,2],
        3
    )
    
    injection_graph = helper.make_graph(
        injection_nodes,
        "test_injection",
        [helper.make_tensor_value_info("add_output", TensorProto.INT8, input_shape)],
        [helper.make_tensor_value_info("output_tensor", TensorProto.INT8, input_shape)]
    )

    # Save injection graph
    injection_model = helper.make_model(injection_graph)
    injection_model.opset_import[0].version = 18
    onnx.save(injection_model, "injection_graph.onnx")

    # 3. Merge and create model
    merged_graph = create_fault_layer(base_graph, injection_graph, "add_output")
    model = helper.make_model(merged_graph)
    model.opset_import[0].version = 18
    
    # Save merged graph
    onnx.save(model, "merged_graph.onnx")
    
    # 4. Run inference
    session = ort.InferenceSession(model.SerializeToString())
    test_input = np.ones(input_shape).astype(np.float32)
    outputs = session.run(None, {"float_input": test_input})
    
    # 5. Verify results
    result = outputs[0]
    print(f"Original value at target: {result[0,1,2,2]}")
    print(f"Values around target: \n{result[0,1,2,:]}")
    
    return model, result

# Run test and save models
model, result = test_fault_injection_pipeline()
print("Models saved as base_graph.onnx, injection_graph.onnx and merged_graph.onnx")

# Run test
if __name__ == "__main__":
    model, result = test_fault_injection_pipeline()
    print("Test passed successfully!")