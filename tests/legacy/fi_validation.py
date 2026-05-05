import onnx
import numpy as np
import onnxruntime as ort
import os
import csv
import json
from onnx import helper, shape_inference
from find_op_pairs import modify_onnx_graph_input, modify_onnx_graph_weight, modify_onnx_graph_random
import re

def add_faulty_matmul_output(model_path, config, fault_model, output_path=None):
    """
    Modify a fault-injected ONNX model to expose the faulty MatMul output tensor.
    
    Args:
        model_path: Path to the fault-injected model
        config: Configuration dictionary
        fault_model: The fault model used (INPUT, WEIGHT, INPUT16, WEIGHT16)
        output_path: Path to save the modified model
    """
    if output_path is None:
        output_path = model_path.replace(".onnx", "_expose_faulty.onnx")
    
    # Load the model
    model = onnx.load(model_path)
    
    # Extract the target layer name and base name
    target_layer = config["target_layer"]
    target_base = target_layer.replace("/MatMul", "")
    
    # Find all node outputs
    all_outputs = []
    for node in model.graph.node:
        all_outputs.extend(node.output)
    
    # Determine the expected tensor suffix based on fault model
    if "16" in fault_model:
        # For INPUT16 and WEIGHT16, we're looking for a masked output
        suffix = "_fault_injected_masked"
    else:
        # For INPUT and WEIGHT, we're looking for the direct fault-injected output
        suffix = "_fault_injected"
    
    # Look for the specific output tensor
    pattern = f"{target_base}/MatMul_output_0{suffix}"
    matches = [output for output in all_outputs if pattern in output]
    
    # Print debugging info
    print(f"Searching for output pattern: {pattern}")
    print(f"Found {len(matches)} matching outputs")
    
    # Find fault-injected outputs (for diagnosis)
    fault_injected_outputs = [output for output in all_outputs if "_fault_injected" in output]
    print(f"Found {len(fault_injected_outputs)} outputs with '_fault_injected':")
    for output in fault_injected_outputs[:10]:  # Print up to 10 examples
        print(f"  - {output}")
    
    # Find masked outputs (for diagnosis)
    masked_outputs = [output for output in all_outputs if "_masked" in output]
    print(f"Found {len(masked_outputs)} outputs with '_masked':")
    for output in masked_outputs[:10]:  # Print up to 10 examples
        print(f"  - {output}")
    
    # Select the appropriate output tensor
    if matches:
        faulty_output = matches[0]
        print(f"Using specific match: {faulty_output}")
    else:
        # Fall back to broader matching if specific pattern not found
        if "16" in fault_model:
            # For *16 models, try to find any masked output related to the target
            broader_pattern = f"{target_base}/.*_masked"
            broader_matches = [out for out in all_outputs if re.search(broader_pattern, out)]
            if broader_matches:
                faulty_output = broader_matches[0]
                print(f"Using broader masked match: {faulty_output}")
            else:
                raise ValueError(f"No masked outputs found for {fault_model} model")
        else:
            # For regular models, try to find any fault_injected output related to the target
            broader_pattern = f"{target_base}/.*_fault_injected"
            broader_matches = [out for out in all_outputs if re.search(broader_pattern, out)]
            if broader_matches:
                faulty_output = broader_matches[0]
                print(f"Using broader fault_injected match: {faulty_output}")
            else:
                raise ValueError(f"No fault_injected outputs found for {fault_model} model")
    
    # Create a new ValueInfo for the faulty output
    faulty_output_value_info = helper.make_tensor_value_info(
        faulty_output,
        10,  # FLOAT16
        None  # Let shape inference handle this
    )
    
    # Add the new output to the graph
    model.graph.output.append(faulty_output_value_info)
    
    # Run shape inference to update output shapes
    model = shape_inference.infer_shapes(model)
    
    # Save the modified model
    onnx.save(model, output_path)
    print(f"Modified model saved with output {faulty_output} for {fault_model} model")
    
    return output_path, faulty_output

def run_single_experiment(config, fault_model, bit_position):
    """
    Run a single fault injection experiment and return the results.
    """
    # Llama config
    llama_config = {
        "fp16": True,
        "precision": "int8"
    }
    
    print(f"Running experiment with fault model: {fault_model}, bit position: {bit_position}")
    
    # Make a copy of the config to avoid modifying the original
    config_copy = config.copy()
    
    # Apply the appropriate fault injection technique
    try:
        if fault_model in ['INPUT', 'INPUT16']:
            faulty_path = modify_onnx_graph_input(config_copy, llama_config, fault_model, bit_position)
        elif fault_model in ['WEIGHT', 'WEIGHT16']:
            faulty_path = modify_onnx_graph_weight(config_copy, llama_config, fault_model, bit_position)
        else:
            faulty_path = modify_onnx_graph_random(config_copy, llama_config, fault_model, bit_position)
    except Exception as e:
        print(f"Error in fault injection: {str(e)}")
        raise
    
    # Add the faulty MatMul output to the model outputs
    try:
        final_path, actual_faulty_output = add_faulty_matmul_output(faulty_path, config_copy, fault_model)
    except Exception as e:
        print(f"Error adding output: {str(e)}")
        raise
    
    # Generate test input
    N = 10           
    lastN = 5        
    totalN = N + lastN  

    # Set random seed for reproducibility
    np.random.seed(bit_position + 42)
    
    # Create input tensors
    attn_mask = np.zeros((1, 1, N, totalN), dtype=np.float16)
    for i in range(N):
        attn_mask[0, 0, i, :lastN+i+1] = 1.0

    position_ids = np.arange(lastN, lastN + N, dtype=np.int64).reshape(1, N)
    
    inputs = {
        'hidden_in': np.random.rand(1, N, 4096).astype(np.float16),
        'attn_mask': attn_mask,
        'position_ids': position_ids,
        'past_key_in': np.random.rand(1, 32, lastN, 128).astype(np.float16),
        'past_value_in': np.random.rand(1, 32, lastN, 128).astype(np.float16)
    }
    
    # Create an ONNX Runtime session for the modified model
    try:
        print(f"Creating session for {final_path}")
        session = ort.InferenceSession(final_path, providers=['CUDAExecutionProvider'])
        
        # Get all output names
        output_names = [output.name for output in session.get_outputs()]
        print(f"Available outputs: {output_names}")
        
        # Check if our faulty output is in the list
        if actual_faulty_output not in output_names:
            print(f"WARNING: {actual_faulty_output} not in model outputs!")
            print(f"Available outputs: {output_names}")
        
        # Run the model
        print(f"Running inference with inputs: {list(inputs.keys())}")
        outputs = session.run(output_names, inputs)
        
        if outputs is None:
            print("ERROR: session.run returned None")
            return None
        
        print(f"Inference completed. Got {len(outputs)} outputs")
        
        # Create a dictionary mapping output names to tensors
        output_dict = {}
        for i, (name, tensor) in enumerate(zip(output_names, outputs)):
            print(f"Output {i}: {name}, shape={tensor.shape if tensor is not None else 'None'}")
            output_dict[name] = tensor
        
        # Get standard outputs and the faulty MatMul output
        hidden_out = output_dict.get('hidden_out')
        past_key = output_dict.get('present.key')
        past_value = output_dict.get('present.value')
        faulty_output = output_dict.get(actual_faulty_output)
        
        if faulty_output is None:
            print(f"WARNING: {actual_faulty_output} is None in results")
            return None
        
        # Success - return the results
        print(f"Successfully retrieved faulty output with shape {faulty_output.shape}")
        return {
            "fault_model": fault_model,
            "bit_position": bit_position,
            "hidden_out": hidden_out,
            "past_key": past_key,
            "past_value": past_value,
            "faulty_output": faulty_output
        }
    
    except Exception as e:
        print(f"Error in inference: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full stack trace
        return None
    
    # ... rest of the function remains the same ...
if __name__ == "__main__":
    # Setup Llama configuration with layer files directory
    llama_config = {
        "fp16": True,
        "precision": "int8",
        "layer_files": "input_llm"  # Directory containing layer configuration files
    }
    
    # Create directory for results
    results_dir = "fault_injection_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # CSV to save results metrics
    results_csv = os.path.join(results_dir, "fault_metrics.csv")
    
    with open(results_csv, 'w', newline='') as csvfile:
        fieldnames = ["layer_config", "fault_model", "bit_position", "faulty_norm", "max_abs_val", "nonzero_count"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Bit position for testing
        bit_position = 3  # Example bit position
        
        # Outer loop: Iterate over all layer configuration files
        for layer_file in os.listdir(llama_config['layer_files']):
            config_path = os.path.join(llama_config['layer_files'], layer_file)
            
            # Skip directories
            if os.path.isdir(config_path):
                continue
            
            try:
                # Load the layer configuration
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                print(f"\n=== Processing layer configuration: {layer_file} ===")
                print(f"Target layer: {config['target_layer']}")
                
                # Inner loop: Run experiments for all fault models with this layer config
                fault_models = ['WEIGHT16']
                
                for fault_model in fault_models:
                    try:
                        # Run the experiment
                        results = run_single_experiment(config, fault_model, bit_position)
                        
                        # Extract the faulty output
                        faulty_output = results["faulty_output"]
                        
                        if faulty_output is not None:
                            # Calculate metrics
                            faulty_norm = np.linalg.norm(faulty_output)
                            max_abs_val = np.max(np.abs(faulty_output))
                            nonzero_count = np.count_nonzero(faulty_output)
                            
                            # Write to CSV
                            writer.writerow({
                                "layer_config": layer_file,
                                "fault_model": fault_model,
                                "bit_position": bit_position,
                                "faulty_norm": faulty_norm,
                                "max_abs_val": max_abs_val,
                                "nonzero_count": nonzero_count
                            })
                            
                            
                            
                            print(f"Completed experiment for {layer_file}, {fault_model}. Faulty norm: {faulty_norm:.6f}")
                        else:
                            print(f"Skipping metrics for {layer_file}, {fault_model} due to missing faulty output")
                        
                    except Exception as e:
                        print(f"Error in {layer_file}, {fault_model} experiment: {str(e)}")
                        
            except Exception as e:
                print(f"Error processing config file {layer_file}: {str(e)}")
    
    print(f"All experiments completed. Results saved to {results_dir}")