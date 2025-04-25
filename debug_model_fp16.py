import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import time
import os

# Create output directory if it doesn't exist
output_dir = "experiment_results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "matmul_add_comparison_experiments_fp16.txt")

# Define model paths and custom ops library if required
golden_model_path = 'decoders/fp16/decoder-merge-20-patched.onnx'  # Golden model (without injection)
faulty_model_path = 'decoders/fp16/decoder-merge-20-patched_injected.onnx'  # Faulty model (with injection)
perturb_lib_path = 'llama/onnx_perturb.so'

# Target node names for exposing outputs
golden_matmul_name = "/self_attn/MatMul_1"        # Name of the MatMul node in the golden model
golden_matmul_output = "/self_attn/MatMul_1_output_0"  # Expected output name for MatMul
faulty_add_name = "_Add"                             # Substring to identify the Add node in the faulty model

timestamp = time.strftime('%Y%m%d_%H%M%S')

with open(output_file, 'a') as f:
    # Write header
    f.write("\n\n" + "="*70 + "\n")
    f.write(f"NEW RUN: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*70 + "\n\n")
    
    f.write("Golden vs Faulty Model: Exposing MatMul (golden) and _Add (faulty) outputs and comparing values\n")
    f.write("="*50 + "\n\n")
    f.write(f"Golden Model: {golden_model_path}\n")
    f.write(f"Faulty Model: {faulty_model_path}\n")
    f.write(f"Custom Ops Library: {perturb_lib_path}\n\n")
    
    # ---------------------------------------------------------------------
    # Step 1: Modify the golden model to expose the MatMul output.
    f.write("Step 1: Modifying golden model to expose MatMul output\n")
    try:
        golden_model = onnx.load(golden_model_path)
        golden_model.opset_import[0].version = 18
        
        matmul_found = False
        for node in golden_model.graph.node:
            if node.op_type == "MatMul" and node.name == golden_matmul_name:
                matmul_found = True
                f.write(f"Found target MatMul node: {node.name}, outputs: {node.output}\n")
                if golden_matmul_output not in node.output:
                    f.write(f"Warning: Expected output {golden_matmul_output} not found. Using {node.output[0]} instead.\n")
                    golden_matmul_output = node.output[0]
                break
        
        if not matmul_found:
            f.write("Target MatMul node not found in golden model. Listing all MatMul nodes:\n")
            for node in golden_model.graph.node:
                if node.op_type == "MatMul":
                    f.write(f"  MatMul node: {node.name}, outputs: {node.output}\n")
            raise ValueError(f"Target MatMul node '{golden_matmul_name}' not found")
        
        # Append MatMul output if not already present
        if golden_matmul_output not in [o.name for o in golden_model.graph.output]:
            matmul_info = helper.make_tensor_value_info(
                name=golden_matmul_output,
                elem_type=TensorProto.FLOAT16,
                shape=None  # Dynamic shape
            )
            golden_model.graph.output.append(matmul_info)
            f.write(f"Appended MatMul output '{golden_matmul_output}' to the model outputs.\n")
        else:
            f.write(f"MatMul output '{golden_matmul_output}' already exists in the model outputs.\n")
        
        golden_mod_path = os.path.join(output_dir, f"golden_modified_{timestamp}.onnx")
        onnx.save(golden_model, golden_mod_path)
        f.write(f"Modified golden model saved to {golden_mod_path}\n")
        
    except Exception as e:
        f.write(f"Error modifying golden model: {e}\n")
        import traceback
        f.write(traceback.format_exc() + "\n")
        raise

    # ---------------------------------------------------------------------
    # Step 2: Modify the faulty model to expose the _Add output.
    f.write("\nStep 2: Modifying faulty model to expose _Add output\n")
    try:
        faulty_model = onnx.load(faulty_model_path)
        faulty_model.opset_import[0].version = 18
        
        add_found = False
        faulty_add_output = None
        for node in faulty_model.graph.node:
            if node.op_type == "Add" and faulty_add_name in node.name:
                add_found = True
                faulty_add_output = node.output[0]
                f.write(f"Found target Add node: {node.name}, outputs: {node.output}\n")
                break
        
        if not add_found:
            f.write("Target Add node not found in faulty model. Listing all Add nodes:\n")
            for node in faulty_model.graph.node:
                if node.op_type == "Add":
                    f.write(f"  Add node: {node.name}, outputs: {node.output}\n")
            raise ValueError(f"Target Add node containing '{faulty_add_name}' not found")
        
        if faulty_add_output not in [o.name for o in faulty_model.graph.output]:
            add_info = helper.make_tensor_value_info(
                name=faulty_add_output,
                elem_type=TensorProto.FLOAT16,
                shape=None
            )
            faulty_model.graph.output.append(add_info)
            f.write(f"Appended Add output '{faulty_add_output}' to the model outputs.\n")
        else:
            f.write(f"Add output '{faulty_add_output}' already exists in the model outputs.\n")
        
        faulty_mod_path = os.path.join(output_dir, f"faulty_modified_{timestamp}.onnx")
        onnx.save(faulty_model, faulty_mod_path)
        f.write(f"Modified faulty model saved to {faulty_mod_path}\n")
        
    except Exception as e:
        f.write(f"Error modifying faulty model: {e}\n")
        import traceback
        f.write(traceback.format_exc() + "\n")
        raise

    # ---------------------------------------------------------------------
    # Step 3: Create CUDA sessions with the modified models.
    f.write("\nStep 3: Creating inference sessions using CUDAExecutionProvider\n")
    try:
        sess_options = ort.SessionOptions()
        f.write(f"Registering custom ops library: {perturb_lib_path}\n")
        sess_options.register_custom_ops_library(perturb_lib_path)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        golden_session = ort.InferenceSession(golden_mod_path, sess_options, providers=['CUDAExecutionProvider'])
        golden_outputs = [output.name for output in golden_session.get_outputs()]
        f.write("Golden model outputs:\n")
        for name in golden_outputs:
            f.write(f"  {name}\n")
        f.write(f"Target MatMul output '{golden_matmul_output}' available: {golden_matmul_output in golden_outputs}\n")
        
        faulty_session = ort.InferenceSession(faulty_mod_path, sess_options, providers=['CUDAExecutionProvider'])
        faulty_outputs = [output.name for output in faulty_session.get_outputs()]
        f.write("\nFaulty model outputs:\n")
        for name in faulty_outputs:
            f.write(f"  {name}\n")
        f.write(f"Target Add output '{faulty_add_output}' available: {faulty_add_output in faulty_outputs}\n")
        
    except Exception as e:
        f.write(f"Error creating sessions: {e}\n")
        import traceback
        f.write(traceback.format_exc() + "\n")
        raise

    # ---------------------------------------------------------------------
    # Step 4: Run experiments with identical inputs and compare outputs.
    f.write("\nStep 4: Running experiments with identical inputs and comparing outputs\n")
    f.write("="*50 + "\n\n")
    
    for exp_num in range(5):
        f.write(f"Experiment {exp_num+1}/5:\n")
        f.write("-"*40 + "\n")
        
        # Generate fixed input
        N = 10
        lastN = 5
        totalN = N + lastN

        seed = exp_num + 42
        np.random.seed(seed)
        
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
            # Run inference for golden model
            f.write("Golden model execution:\n")
            golden_results = golden_session.run(None, inputs)
            golden_out = {name: tensor for name, tensor in zip(golden_outputs, golden_results)}
            
            golden_hidden_out = golden_out.get('hidden_out')
            golden_past_key = golden_out.get('present.key') or golden_out.get('past_key')
            golden_past_value = golden_out.get('present.value') or golden_out.get('past_value')
            golden_matmul_tensor = golden_out.get(golden_matmul_output)
            
            f.write(f"  hidden_out exists: {golden_hidden_out is not None}\n")
            f.write(f"  past_key exists: {golden_past_key is not None}\n")
            f.write(f"  past_value exists: {golden_past_value is not None}\n")
            f.write(f"  MatMul output exists: {golden_matmul_tensor is not None}\n")
            if golden_matmul_tensor is not None:
                flat_matmul = golden_matmul_tensor.flatten()
                f.write(f"  MatMul output shape: {golden_matmul_tensor.shape}\n")
                f.write(f"  MatMul first 5 values: {flat_matmul[:5]}\n")
            
            # Run inference for faulty model
            f.write("\nFaulty model execution (same input):\n")
            faulty_results = faulty_session.run(None, inputs)
            faulty_out = {name: tensor for name, tensor in zip(faulty_outputs, faulty_results)}
            
            faulty_hidden_out = faulty_out.get('hidden_out')
            faulty_past_key = faulty_out.get('present.key') or faulty_out.get('past_key')
            faulty_past_value = faulty_out.get('present.value') or faulty_out.get('past_value')
            faulty_add_tensor = faulty_out.get(faulty_add_output)
            
            f.write(f"  hidden_out exists: {faulty_hidden_out is not None}\n")
            f.write(f"  past_key exists: {faulty_past_key is not None}\n")
            f.write(f"  past_value exists: {faulty_past_value is not None}\n")
            f.write(f"  Add output exists: {faulty_add_tensor is not None}\n")
            if faulty_add_tensor is not None:
                flat_add = faulty_add_tensor.flatten()
                f.write(f"  Add output shape: {faulty_add_tensor.shape}\n")
                f.write(f"  Add first 5 values: {flat_add[:5]}\n")
            
            # -----------------------------------------------------------------
            # Compare the target outputs: MatMul (golden) vs Add (faulty)
            if golden_matmul_tensor is not None and faulty_add_tensor is not None:
                tensors_equal = np.array_equal(golden_matmul_tensor, faulty_add_tensor)
                close_enough = np.allclose(golden_matmul_tensor, faulty_add_tensor, rtol=1e-03, atol=1e-05)
                f.write("\nComparison of MatMul (golden) and Add (faulty) outputs:\n")
                f.write(f"  Exact equality: {tensors_equal}\n")
                f.write(f"  Close enough (np.allclose): {close_enough}\n")
            
            # Compare standard outputs
            def compare_output(name, golden_val, faulty_val):
                if golden_val is not None and faulty_val is not None:
                    equal = np.array_equal(golden_val, faulty_val)
                    close = np.allclose(golden_val, faulty_val, rtol=1e-03, atol=1e-05)
                    f.write(f"\nComparison for {name}:\n")
                    f.write(f"  Exact equality: {equal}\n")
                    f.write(f"  Close enough (np.allclose): {close}\n")
                    # Print first few values if available
                    f.write(f"  Golden first 5: {golden_val.flatten()[:5]}\n")
                    f.write(f"  Faulty first 5: {faulty_val.flatten()[:5]}\n")
                else:
                    f.write(f"\nCannot compare {name} as one of them is None.\n")
            
            compare_output("hidden_out", golden_hidden_out, faulty_hidden_out)
            compare_output("past_key", golden_past_key, faulty_past_key)
            compare_output("past_value", golden_past_value, faulty_past_value)
            
        except Exception as e:
            error_msg = f"Error during experiment {exp_num+1}: {e}"
            f.write(f"\n{error_msg}\n")
            import traceback
            f.write(traceback.format_exc() + "\n")
        
        f.write("\n" + "="*50 + "\n\n")

print(f"All experiments completed. Results appended to {output_file}")
