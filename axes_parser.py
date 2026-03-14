import onnx
from onnx import helper, TensorProto, save_model, numpy_helper

def patch_reduce_ops(model, reduce_ops=("ReduceMean", "ReduceMax")):
    graph = model.graph
    const_nodes = []
    original_nodes = list(graph.node)  # capture the original node list
    new_nodes = []
    
    for node in original_nodes:
        if node.op_type in reduce_ops:
            axes_attr = None
            for attr in node.attribute:
                if attr.name == "axes":
                    axes_attr = attr
                    break
            if axes_attr is not None:
                axes_values = list(axes_attr.ints)
                # Remove the axes attribute.
                node.attribute.remove(axes_attr)
                # Create a new Constant node for these axes.
                axes_const_name = node.name + "_axes_const"
                const_node = helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_const_name],
                    value=helper.make_tensor(
                        name=axes_const_name,
                        data_type=TensorProto.INT64,
                        dims=[len(axes_values)],
                        vals=axes_values
                    )
                )
                const_nodes.append(const_node)
                node.input.append(axes_const_name)
        new_nodes.append(node)
    
    # Prepend the constant nodes.
    all_nodes = const_nodes + new_nodes
    graph.ClearField("node")
    graph.node.extend(all_nodes)
    return model

def move_initializers_to_constant_for_matmul(model):
    graph = model.graph
    # Build a dictionary for quick lookup of initializer values.
    init_dict = {init.name: init for init in graph.initializer}
    constant_nodes = []
    replaced_inits = {}  # map original initializer name -> new constant name

    # Iterate over all nodes in the graph.
    for node in graph.node:
        if node.op_type == "MatMul":
            new_inputs = []
            for inp in node.input:
                if inp in init_dict:
                    # If not already replaced, create a Constant node.
                    if inp not in replaced_inits:
                        new_const_name = inp 
                        replaced_inits[inp] = new_const_name
                        const_node = helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[new_const_name],
                            value=init_dict[inp]
                        )
                        constant_nodes.append(const_node)
                    new_inputs.append(replaced_inits[inp])
                else:
                    new_inputs.append(inp)
            # Update node inputs by modifying the list element‐by‐element.
            for i in range(len(node.input)):
                node.input[i] = new_inputs[i]

    # Prepend the new constant nodes.
    original_nodes = list(graph.node)
    graph.ClearField("node")
    graph.node.extend(constant_nodes + original_nodes)

    # Remove replaced initializers from the initializer list.
    new_initializers = [init for init in graph.initializer if init.name not in replaced_inits]
    graph.ClearField("initializer")
    graph.initializer.extend(new_initializers)

    # Also remove them from graph inputs if present.
    new_inputs = [inp for inp in graph.input if inp.name not in replaced_inits]
    graph.ClearField("input")
    graph.input.extend(new_inputs)

    return model

# if __name__ == "__main__":
#     original_model_path = "decoders/decoder-merge-20.onnx"       # your original model
#     updated_model_path = "decoders/decoder-merge-20-patched.onnx"   # destination for patched model

#     # Load the model.
#     model = onnx.load(original_model_path)
#     # First, patch reduce operators.
#     model = patch_reduce_ops(model, reduce_ops=("ReduceMean","ReduceMax"))
#     # Next, move initializers to Constant nodes for MatMul.
#     model = move_initializers_to_constant_for_matmul(model)

#     save_model(model, updated_model_path)
#     print("Saved patched model as", updated_model_path)
