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
                # Use the node name if available; otherwise, return the candidate (i.e. the constant's output)
                return node.name if node.name != "" else candidate
    return None

def parse_transformer_pairs(model_path: str):
    model = onnx.load(model_path)
    graph = model.graph

    # Patterns for different projection subgraphs.
    patterns = {
        'q_proj': '/self_attn/q_proj',
        'k_proj': '/self_attn/k_proj',
        'v_proj': '/self_attn/v_proj',
        'gate_proj': '/mlp/gate_proj',
        'up_proj': '/mlp/up_proj',
        'down_proj': '/mlp/down_proj'
    }
    
    # Build a dictionary for initializers.
    init_dict = {init.name: init for init in graph.initializer}
    
    for pattern_name, pattern in patterns.items():
        round_node_name = None
        matmul_node_name = None
        weight_node_name = None
        
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
            
            # If the node is a Round node (or has Round in its op type), record it.
            if node.op_type == "Round":
                round_node_name = node.name if node.name else (node.output[0] if node.output else None)
            # If the node is a MatMul, record its identity and try to identify its weight input.
            elif node.op_type == "MatMul":
                matmul_node_name = node.name if node.name else (node.output[0] if node.output else None)
                if len(node.input) >= 2:
                    candidate = node.input[1]
                    # First, check if the candidate is still in the initializer dictionary.
                    if candidate in init_dict:
                        weight_node_name = candidate
                    else:
                        # Otherwise, try to find a Constant node that produces this candidate.
                        weight_node_name = find_weight_constant_node(graph, candidate)
            
            # If we’ve found all three, break early.
            if round_node_name and matmul_node_name and weight_node_name:
                break
        
        # Save info to JSON if all nodes are found.
        if round_node_name and matmul_node_name and weight_node_name:
            info = {
                "round_node": round_node_name,
                "matmul_node": matmul_node_name,
                "weight_node": weight_node_name,
                "decoder_path": model_path
            }
            decoder_name = os.path.basename(model_path).replace('.onnx', '')
            json_filename = f'input_llm/{decoder_name}_{pattern_name}.json'
            with open(json_filename, 'w') as f:
                json.dump(info, f, indent=4)
            print(f"Saved JSON for pattern '{pattern_name}' as {json_filename}")
        else:
            print(f"Could not find all nodes for pattern '{pattern_name}':")
            print("  round_node:", round_node_name)
            print("  matmul_node:", matmul_node_name)
            print("  weight_node:", weight_node_name)

if __name__ == "__main__":
    # Directory containing your ONNX files.
    onnx_dir = "decoders"
    # Use glob to find all .onnx files in the specified directory.
    onnx_files = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    print(f"Found {len(onnx_files)} ONNX files in directory '{onnx_dir}'")
    
    # Process each ONNX file.
    for model_path in onnx_files:
        print(f"\nProcessing {model_path}")
        parse_transformer_pairs(model_path)
