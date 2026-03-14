import onnx
import json
import os
import glob
from collections import deque, defaultdict
import argparse

def trace_tensor_to_round(graph, tensor_name):

    producer_map = {}
    for node in graph.node:
        for output in node.output:
            producer_map[output] = node
    
    # BFS search to find Round node
    visited = set()
    queue = deque([tensor_name])
    
    while queue:
        current_tensor = queue.popleft()
        if current_tensor in visited:
            continue
        visited.add(current_tensor)
        
        # Check if this tensor is produced by a Round node
        if current_tensor in producer_map:
            producer = producer_map[current_tensor]
            if producer.op_type == "Round":
                return current_tensor
            
            # Not a Round node, add its inputs to the queue
            for input_tensor in producer.input:
                queue.append(input_tensor)
    
    # No path to a Round node found
    return None

def parse_transformer_pairs(model_path: str):
    model = onnx.load(model_path)
    graph = model.graph

    # Update patterns to include all specified MatMul layers
    patterns = {
        'q_proj': '/self_attn/q_proj/MatMul',
        'k_proj': '/self_attn/k_proj/MatMul',
        'v_proj': '/self_attn/v_proj/MatMul',
        'self_attn1': '/self_attn/MatMul',
        'self_attn2': '/self_attn/MatMul_1',
        'o_proj': '/self_attn/o_proj/MatMul',
        'gate_proj': '/mlp/gate_proj/MatMul',
        'up_proj': '/mlp/up_proj/MatMul',
        'down_proj': '/mlp/down_proj/MatMul'
    }
    
    # Build a dictionary for initializers
    init_dict = {init.name: init for init in graph.initializer}
    
    # Build a map of consumers (tensor name -> list of nodes that consume it)
    consumers = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            consumers[inp].append(node)
    
    for pattern_name, pattern in patterns.items():
        print(f"\nProcessing pattern '{pattern_name}' ({pattern})")
        
        matmul_node = None
        input_tensor = None
        weight_tensor = None
        
        # Find the MatMul node for this pattern
        for node in graph.node:
            if node.op_type == "MatMul" and pattern in node.name:
                matmul_node = node
                break
        
        if not matmul_node:
            print(f"Could not find MatMul node for pattern '{pattern_name}'")
            continue
        
        print(f"Found MatMul node: {matmul_node.name}")
        print(f"MatMul inputs: {matmul_node.input}")
        
        # For each input to the MatMul node, trace back to a Round node
        for i, inp in enumerate(matmul_node.input):
            round_output = trace_tensor_to_round(graph, inp)
            
            if round_output:
                if i == 0:  # First input is traditionally the activation input
                    input_tensor = round_output
                    print(f"Found input tensor from Round: {input_tensor}")
                else:  # Second input is traditionally the weight
                    weight_tensor = round_output
                    print(f"Found weight tensor from Round: {weight_tensor}")
        
        # Save info to JSON if both tensors were found
        if input_tensor and weight_tensor:
            info = {
                "input_tensor": input_tensor,
                "target_layer": matmul_node.name,
                "weight_tensor": weight_tensor,
                "model_name": model_path
            }
            decoder_name = os.path.basename(model_path).replace('.onnx', '')
            json_filename = f'injection_llm/{decoder_name}_{pattern_name}.json'
            os.makedirs('injection_llm', exist_ok=True)
            with open(json_filename, 'w') as f:
                json.dump(info, f, indent=4)
            print(f"Saved JSON for pattern '{pattern_name}' as {json_filename}")
        else:
            print(f"Could not find all tensors from Round nodes for pattern '{pattern_name}':")
            print("  input_tensor:", input_tensor)
            print("  weight_tensor:", weight_tensor)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process ONNX files to identify transformer patterns.')
    parser.add_argument('onnx_dir', type=str, help='Directory containing ONNX model files')
    
    args = parser.parse_args()
    
    onnx_dir = args.onnx_dir
  
    onnx_files = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    print(f"Found {len(onnx_files)} ONNX files in directory '{onnx_dir}'")
    
    for model_path in onnx_files:
        print(f"\nProcessing {model_path}")
        parse_transformer_pairs(model_path)