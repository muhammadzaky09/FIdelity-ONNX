import onnx
import json
import os
import glob

def find_weight_constant_node(graph, candidate):
    print(f"\nSearching for weight constant node for candidate: {candidate}")
    
    # For debugging: list all MatMul and Constant nodes.
    for node in graph.node:
        if node.op_type == "MatMul":
            node_id = node.name if node.name else (node.output[0] if node.output else "[Unnamed]")
            print(f"MatMul node {node_id}: inputs={node.input}")
        if node.op_type == "Constant":
            node_id = node.name if node.name else (node.output[0] if node.output else "[Unnamed]")
            print(f"Constant node {node_id}: outputs={node.output}")
    
    # Look for a Constant node that produces the candidate.
    for node in graph.node:
        if node.op_type == "Constant":
            if candidate in node.output:
                # Return just the candidate name (the tensor name)
                return candidate
    return None

def parse_transformer_pairs(model_path: str):
    model = onnx.load(model_path)
    graph = model.graph

    # Patterns for different projection subgraphs.
    patterns = {
        'q_proj': '/self_attn/q_proj',
        'k_proj': '/self_attn/k_proj',
        'v_proj': '/self_attn/v_proj',
        'o_proj': '/self_attn/o_proj',
        'gate_proj': '/mlp/gate_proj',
        'up_proj': '/mlp/up_proj',
        'down_proj': '/mlp/down_proj'
    }
    
    # Build a dictionary for initializers.
    init_dict = {init.name: init for init in graph.initializer}
    
    for pattern_name, pattern in patterns.items():
        round_node = None
        round_output = None
        matmul_node = None
        weight_tensor = None
        
        # Iterate over nodes to find nodes that are part of the subgraph.
        for node in graph.node:
            # Determine if the node is relevant by checking both its name and outputs.
            node_relevant = False
            if node.name and pattern in node.name:
                node_relevant = True
            else:
                for out in node.output:
                    if pattern in out:
                        node_relevant = True
                        break
            if not node_relevant:
                continue
            
            # If the node is a Round node, record it and its output tensor name
            if node.op_type == "Round":
                round_node = node.name if node.name else None
                if node.output and len(node.output) > 0:
                    round_output = node.output[0]
                
            # If the node is a MatMul, record its identity and try to identify its weight input.
            elif node.op_type == "MatMul":
                matmul_node = node.name if node.name else (node.output[0] if node.output else None)
                if len(node.input) >= 2:
                    candidate = node.input[1]
                    # First, check if the candidate is still in the initializer dictionary.
                    if candidate in init_dict:
                        weight_tensor = candidate
                    else:
                        # Otherwise, try to find a Constant node that produces this candidate.
                        weight_tensor = find_weight_constant_node(graph, candidate)
            
            # If we've found all necessary components, break early.
            if round_output and matmul_node and weight_tensor:
                break
        
        # Save info to JSON with the new format if all components are found.
        if round_output and matmul_node and weight_tensor:
            info = {
                "input_tensor": round_output,
                "target_layer": matmul_node,
                "weight_tensor": weight_tensor,
                "model_name": model_path
            }
            decoder_name = os.path.basename(model_path).replace('.onnx', '')
            json_filename = f'injection_llm/{decoder_name}_{pattern_name}.json'
            with open(json_filename, 'w') as f:
                json.dump(info, f, indent=4)
            print(f"Saved JSON for pattern '{pattern_name}' as {json_filename}")
        else:
            print(f"Could not find all components for pattern '{pattern_name}':")
            print("  input_tensor (round output):", round_output)
            print("  target_layer (matmul):", matmul_node)
            print("  weight_tensor:", weight_tensor)

if __name__ == "__main__":
    # Directory containing your ONNX files.
    onnx_dir = "decoders/7B16"
    # Use glob to find all .onnx files in the specified directory.
    onnx_files = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    print(f"Found {len(onnx_files)} ONNX files in directory '{onnx_dir}'")
    
    # Process each ONNX file.
    for model_path in onnx_files:
        print(f"\nProcessing {model_path}")
        parse_transformer_pairs(model_path)