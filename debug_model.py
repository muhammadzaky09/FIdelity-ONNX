import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper
import time
import os

# Create output directory if it doesn't exist
output_dir = "experiment_results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"matmul_experiments_{time.strftime('%Y%m%d_%H%M%S')}.txt")

# Define model paths
model1_path = 'decoders/decoder-merge-20-patched.onnx'
model2_path = 'decoders/decoder-merge-20_injected.onnx'

# Define the target node names
target1_node_name = "self_attn/o_proj/MatMul"
target2_node_name = "self_attn/o_proj/MatMul_output_0_Add"
target2_output = "/self_attn/o_proj/MatMul_output_0_final"

with open(output_file, 'w') as f:
    f.write("Experiment Results: Extracting MatMul and Add Outputs\n")
    f.write("="*50 + "\n\n")
    
    # Step 1: Modify model 1 to include MatMul output
    f.write("Step 1: Modifying model 1 to expose MatMul output\n")
    try:
        model1 = onnx.load(model1_path)
        model1.opset_import[0].version = 18
        
        # Find the MatMul node
        matmul_found = False
        for node in model1.graph.node:
            if node.op_type == "MatMul" and any("self_attn/o_proj" in output for output in node.output):
                target1_output = node.output[0]
                f.write(f"Found target MatMul node, output: {target1_output}\n")
                matmul_found = True
                break
        
        if not matmul_found:
            f.write("MatMul node not found in model 1. Listing all MatMul nodes:\n")
            for node in model1.graph.node:
                if node.op_type == "MatMul":
                    f.write(f"MatMul node: outputs={node.output}\n")
            raise ValueError("Target MatMul node not found")
        
        # Add MatMul output to the model outputs
        matmul_info = helper.make_tensor_value_info(
            name=target1_output,
            elem_type=onnx.TensorProto.FLOAT16,
            shape=["?", "?", "?"]
        )
        model1.graph.output.append(matmul_info)
        
        # Save the modified model
        model1_mod_path = os.path.join(output_dir, "model1_modified.onnx")
        onnx.save(model1, model1_mod_path)
        f.write(f"Modified model 1 saved to {model1_mod_path}\n")
        
    except Exception as e:
        f.write(f"Error modifying model 1: {e}\n")
        model1_mod_path = model1_path
    
    # Step 2: Modify model 2 to include Add output
    f.write("\nStep 2: Modifying model 2 to expose Add output\n")
    try:
        model2 = onnx.load(model2_path)
        model2.opset_import[0].version = 18
        
        # Find the Add node
        add_found = False
        for node in model2.graph.node:
            if node.name == target2_node_name or target2_output in node.output:
                f.write(f"Found target Add node: {node.name}, outputs: {node.output}\n")
                add_found = True
                break
        
        if not add_found:
            f.write("Target Add node not found. Listing all Add nodes:\n")
            for node in model2.graph.node:
                if node.op_type == "Add":
                    f.write(f"Add node: name={node.name}, outputs={node.output}\n")
            raise ValueError("Target Add node not found")
        
        # Add the output to the model outputs
        add_info = helper.make_tensor_value_info(
            name=target2_output,
            elem_type=onnx.TensorProto.FLOAT16,
            shape=["?", "?", "?"]
        )
        model2.graph.output.append(add_info)
        
        # Save the modified model
        model2_mod_path = os.path.join(output_dir, "model2_modified.onnx")
        onnx.save(model2, model2_mod_path)
        f.write(f"Modified model 2 saved to {model2_mod_path}\n")
        
    except Exception as e:
        f.write(f"Error modifying model 2: {e}\n")
        model2_mod_path = model2_path
    
    # Step 3: Create sessions with the modified models
    f.write("\nStep 3: Creating inference sessions\n")
    
    try:
        # Create session for model 1
        session1 = ort.InferenceSession(model1_mod_path, providers=['CPUExecutionProvider'])
        
        # List all available outputs
        f.write("Model 1 outputs:\n")
        model1_outputs = []
        for output in session1.get_outputs():
            model1_outputs.append(output.name)
            f.write(f"  {output.name}\n")
        
        f.write(f"Target MatMul output '{target1_output}' available: {target1_output in model1_outputs}\n")
    except Exception as e:
        f.write(f"Error creating session for model 1: {e}\n")
        raise ValueError("Failed to create session for model 1")
    
    try:
        # Create session for model 2
        session2 = ort.InferenceSession(model2_mod_path, providers=['CPUExecutionProvider'])
        
        # List all available outputs
        f.write("\nModel 2 outputs:\n")
        model2_outputs = []
        for output in session2.get_outputs():
            model2_outputs.append(output.name)
            f.write(f"  {output.name}\n")
        
        f.write(f"Target Add output '{target2_output}' available: {target2_output in model2_outputs}\n")
    except Exception as e:
        f.write(f"Error creating session for model 2: {e}\n")
        raise ValueError("Failed to create session for model 2")
    
    # Step 4: Run experiments
    f.write("\nStep 4: Running experiments\n")
    f.write("="*50 + "\n\n")
    
    for exp_num in range(20):
        f.write(f"Experiment {exp_num+1}/20:\n")
        f.write("-"*40 + "\n")
        
        # Generate random input
        N = 10           
        lastN = 5        
        totalN = N + lastN  

        # Set random seed for reproducibility
        np.random.seed(exp_num + 42)
        
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
        
        try:
            # Run model 1
            print(f"Running experiment {exp_num+1}/20 - Model 1")
            f.write("Model 1 execution:\n")
            
            # Run with all outputs
            outputs1 = session1.run(model1_outputs, inputs)
            
            # Create a dictionary mapping output names to tensors
            output_dict1 = {name: tensor for name, tensor in zip(model1_outputs, outputs1)}
            
            # Extract the standard outputs and the MatMul output
            hidden_out1 = output_dict1.get('hidden_out')
            past_key1 = output_dict1.get('present.key')
            past_value1 = output_dict1.get('present.value')
            matmul_output1 = output_dict1.get(target1_output)
            
            # Verify we got all outputs
            f.write(f"  Got hidden_out: {hidden_out1 is not None}\n")
            f.write(f"  Got past_key: {past_key1 is not None}\n")
            f.write(f"  Got past_value: {past_value1 is not None}\n")
            f.write(f"  Got MatMul output: {matmul_output1 is not None}\n")
            
            if hidden_out1 is not None:
                f.write(f"  hidden_out shape: {hidden_out1.shape}\n")
            if past_key1 is not None:
                f.write(f"  past_key shape: {past_key1.shape}\n")
            if past_value1 is not None:
                f.write(f"  past_value shape: {past_value1.shape}\n")
            if matmul_output1 is not None:
                f.write(f"  MatMul output shape: {matmul_output1.shape}\n")
                f.write(f"  MatMul output first few values: {matmul_output1.flatten()[:5]}\n")
                f.write(f"  MatMul output stats: min={np.min(matmul_output1)}, max={np.max(matmul_output1)}, mean={np.mean(matmul_output1)}\n")
            
            # Run model 2
            print(f"Running experiment {exp_num+1}/20 - Model 2")
            f.write("\nModel 2 execution:\n")
            
            # Run with all outputs
            outputs2 = session2.run(model2_outputs, inputs)
            
            # Create a dictionary mapping output names to tensors
            output_dict2 = {name: tensor for name, tensor in zip(model2_outputs, outputs2)}
            
            # Extract the standard outputs and the Add output
            hidden_out2 = output_dict2.get('hidden_out')
            past_key2 = output_dict2.get('present.key')
            past_value2 = output_dict2.get('present.value')
            add_output2 = output_dict2.get(target2_output)
            
            # Verify we got all outputs
            f.write(f"  Got hidden_out: {hidden_out2 is not None}\n")
            f.write(f"  Got past_key: {past_key2 is not None}\n")
            f.write(f"  Got past_value: {past_value2 is not None}\n")
            f.write(f"  Got Add output: {add_output2 is not None}\n")
            
            if hidden_out2 is not None:
                f.write(f"  hidden_out shape: {hidden_out2.shape}\n")
            if past_key2 is not None:
                f.write(f"  past_key shape: {past_key2.shape}\n")
            if past_value2 is not None:
                f.write(f"  past_value shape: {past_value2.shape}\n")
            if add_output2 is not None:
                f.write(f"  Add output shape: {add_output2.shape}\n")
                f.write(f"  Add output first few values: {add_output2.flatten()[:5]}\n")
                f.write(f"  Add output stats: min={np.min(add_output2)}, max={np.max(add_output2)}, mean={np.mean(add_output2)}\n")
            
            # Compare the standard outputs between models
            if hidden_out1 is not None and hidden_out2 is not None:
                f.write(f"\nhidden_out equal: {np.array_equal(hidden_out1, hidden_out2)}\n")
            if past_key1 is not None and past_key2 is not None:
                f.write(f"past_key equal: {np.array_equal(past_key1, past_key2)}\n")
            if past_value1 is not None and past_value2 is not None:
                f.write(f"past_value equal: {np.array_equal(past_value1, past_value2)}\n")
            
            # Compare MatMul and Add outputs
            if matmul_output1 is not None and add_output2 is not None and matmul_output1.shape == add_output2.shape:
                diff = np.abs(matmul_output1 - add_output2)
                f.write(f"\nDifferences between MatMul and Add outputs:\n")
                f.write(f"  Max difference: {np.max(diff)}\n")
                f.write(f"  Mean difference: {np.mean(diff)}\n")
                f.write(f"  Std difference: {np.std(diff)}\n")
            
        except Exception as e:
            f.write(f"\nError during experiment {exp_num+1}: {e}\n")
            import traceback
            f.write(traceback.format_exc() + "\n")
        
        f.write("\n" + "="*50 + "\n\n")

print(f"All experiments completed. Results saved to {output_file}")