import onnx
import json
import os
import glob

def parse_matmul_nodes(model_path: str):
    """
    Parse an ONNX model to identify all MatMul nodes and extract their input, weight, and output tensors.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        A list of dictionaries containing MatMul information
    """
    model = onnx.load(model_path)
    graph = model.graph
    
    # Dictionary to map initializer names to their info
    init_dict = {init.name: init for init in graph.initializer}
    
    # Dictionary to track which nodes produce which tensors
    producers = {}
    for node in graph.node:
        for output in node.output:
            producers[output] = node
    
    matmul_nodes = []
    
    for node in graph.node:
        if node.op_type == "MatMul":
            # Get node ID (use name if available, output otherwise)
            node_id = node.name if node.name else node.output[0]
            
            # Get input tensor name (first input)
            input_tensor = node.input[0] if len(node.input) > 0 else None
            
            # Get weight tensor name (second input)
            weight_tensor = node.input[1] if len(node.input) > 1 else None
            
            # Find the constant node that produces the weight tensor
            weight_const_name = None
            if weight_tensor and weight_tensor in producers:
                weight_producer = producers[weight_tensor]
                if weight_producer.op_type == "Constant":
                    weight_const_name = f"{weight_tensor}"
            # If not found in producers, it might be an initializer
            elif weight_tensor and weight_tensor in init_dict:
                weight_const_name = weight_tensor
            
            # Get output tensor name
            output_tensor = node.output[0] if len(node.output) > 0 else None
            
            # Only add if we have all the required information
            if input_tensor and weight_tensor and output_tensor:
                matmul_info = {
                    "node_id": node_id,
                    "input_tensor": input_tensor,
                    "weight_tensor": weight_const_name or weight_tensor,
                    "output_tensor": output_tensor,
                    "decoder_path": model_path
                }
                matmul_nodes.append(matmul_info)
                
                print(f"Found MatMul node: {node_id}")
                print(f"  Input tensor: {input_tensor}")
                print(f"  Weight tensor: {weight_const_name or weight_tensor}")
                print(f"  Output tensor: {output_tensor}")
    
    return matmul_nodes

def save_matmul_info(matmul_nodes, output_dir="matmul_info"):
    """
    Save MatMul node information to JSON files.
    
    Args:
        matmul_nodes: List of dictionaries containing MatMul information
        output_dir: Directory to save JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, matmul_info in enumerate(matmul_nodes):
        # Extract model name from path
        model_name = os.path.basename(matmul_info["decoder_path"]).replace('.onnx', '')
        
        # Create a JSON filename based on model name and node ID or index
        if "node_id" in matmul_info and matmul_info["node_id"]:
            # Replace any characters that might be problematic in filenames
            safe_node_id = matmul_info["node_id"].replace('/', '_').replace('\\', '_')
            json_filename = f"{model_name}_{safe_node_id}.json"
        else:
            json_filename = f"{model_name}_matmul_{i}.json"
        
        json_path = os.path.join(output_dir, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(matmul_info, f, indent=4)
        
        print(f"Saved MatMul info to {json_path}")

if __name__ == "__main__":
    # Directory containing your ONNX files
    onnx_dir = "decoders/fp16"
    # Output directory for JSON files
    output_dir = "input_llm_fp16"
    
    # Use glob to find all .onnx files in the specified directory
    onnx_files = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    print(f"Found {len(onnx_files)} ONNX files in directory '{onnx_dir}'")
    
    # Process each ONNX file
    for model_path in onnx_files:
        print(f"\nProcessing {model_path}")
        matmul_nodes = parse_matmul_nodes(model_path)
        save_matmul_info(matmul_nodes, output_dir)