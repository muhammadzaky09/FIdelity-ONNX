import onnx
from collections import deque, defaultdict
from onnx import helper, shape_inference, numpy_helper, TensorProto
from inject_ops import create_quantized_fault_injection,  create_random_bitflip_injection, create_random_fault_injection, create_quantized_fault_injection_weight, create_input16_mask, create_weight16_mask
from typing import List
from itertools import chain
import numpy as np
from inject_utils.utils import delta_init, analyze_onnx_path
from axes_parser import patch_reduce_ops, move_initializers_to_constant_for_matmul




def modify_onnx_graph_input(config, fault_model, bit_position=3):

    model_path = config["decoder_path"]
    output_path = config.get("output_path", model_path.replace(".onnx", "_injected.onnx"))

    # Load model and run shape inference.
    model = onnx.load(model_path)
    model = patch_reduce_ops(model, reduce_ops=("ReduceMean", "ReduceMax"))
    model = shape_inference.infer_shapes(model)

    # Analyze the graph to get the path info.
    path_info = analyze_onnx_path(model_path, config["round_node"], config["matmul_node"])
    if path_info is None:
        raise ValueError("Could not find a path matching the given patterns.")
    src_node, target_node, full_path, external_inputs = path_info

    clone_suffix = "_fault_injected"
    original_matmul_output = target_node.output[0]

    # --- Clone the subgraph from src_node onward ---
    tensor_map = {}
    cloned_nodes = []
    # Map the source node's output.
    tensor_map[src_node.output[0]] = f"{src_node.output[0]}{clone_suffix}"
    for node in full_path[1:]:
        new_inputs = [tensor_map.get(inp, inp) for inp in node.input]
        new_outputs = [f"{out}{clone_suffix}" for out in node.output]
        cloned_node = helper.make_node(node.op_type, new_inputs, new_outputs, f"{node.name}{clone_suffix}")
        cloned_nodes.append(cloned_node)
        for orig_out, new_out in zip(node.output, new_outputs):
            tensor_map[orig_out] = new_out

    injection_nodes = create_quantized_fault_injection(
        input_name=src_node.output[0],
        output_name=tensor_map[src_node.output[0]],
        bit_position=bit_position,
    )

    # --- Build new node list ---
    original_nodes = list(model.graph.node)
    insert_pos = next(i for i, n in enumerate(original_nodes) if n.name == src_node.name) + 1
    new_nodes = (
        original_nodes[:insert_pos] +
        injection_nodes +
        cloned_nodes +
        original_nodes[insert_pos:]
    )
    
    # --- Add a final Add node to sum the original and cloned outputs ---
    cloned_matmul_output = tensor_map[original_matmul_output]
    
    if "16" in fault_model:
        mask_nodes = create_input16_mask(
            matmul_output=cloned_matmul_output,  # using MatMul output
            masked_output=f"{cloned_matmul_output}_masked",
            block_length=16
        )
        new_nodes.extend(mask_nodes)
        cloned_matmul_output = f"{cloned_matmul_output}_masked"

        
    add_node = helper.make_node(
        'Add',
        [original_matmul_output, cloned_matmul_output],
        [f"{original_matmul_output}_final"],
        f"{original_matmul_output}_Add"
    )
    new_nodes.append(add_node)
    
    # --- Rewire downstream nodes ---
    for node in new_nodes:
        if node != add_node and original_matmul_output in node.input:
            node.input[:] = [
                f"{original_matmul_output}_final" if inp == original_matmul_output else inp
                for inp in node.input
            ]
    
    # 6. Update model and validate
    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)

    
    # --- Update initializers for external inputs (if any) ---
    for inp in external_inputs:
        if inp in [i.name for i in model.graph.initializer]:
            orig_init = next(i for i in model.graph.initializer if i.name == inp)
            cloned_init = numpy_helper.from_array(numpy_helper.to_array(orig_init), name=f"{inp}{clone_suffix}")
            model.graph.initializer.append(cloned_init)
    
    # Replace nodes in the graph.

 
    model = shape_inference.infer_shapes(model)
    model.opset_import[0].version = 18
    onnx.save(model, output_path)
    print(f"Modified model saved to {output_path}")
    return output_path

def modify_onnx_graph_weight(config, fault_model, bit_position=3):
    # 1. Load model and run shape inference.
    model_path = config["decoder_path"]
    model = onnx.load(model_path)
    model = patch_reduce_ops(model, reduce_ops=("ReduceMean", "ReduceMax"))
    model = move_initializers_to_constant_for_matmul(model)
    model = shape_inference.infer_shapes(model)
    output_path = config.get("output_path", model_path.replace(".onnx", "_injected.onnx"))
    print(output_path)
    
    # 2. Locate target MatMul node using config["matmul_node"].
    target_node = None
    for node in model.graph.node:
        if node.op_type == 'MatMul' and config["matmul_node"] in node.name:
            target_node = node
            break
    if target_node is None:
        raise ValueError(f"Target MatMul node with pattern '{config['matmul_node']}' not found.")
    
    # 3. Identify activation and weight inputs.
    activation_input = target_node.input[0]
    weight_input = target_node.input[1]  # Should match config["weight_node"]
    
    # 4. Locate the weight constant node using config["weight_node"].
    weight_const_node = None
    for node in model.graph.node:
        # Use exact match on output name
        if node.op_type == 'Constant' and config["weight_node"] in node.output:
            weight_const_node = node
            break
    if weight_const_node is None:
        raise ValueError("Weight constant node not found for injection.")
    print(f"Found weight constant node '{weight_const_node.name}' with output '{weight_const_node.output[0]}'.")
    
    # 4.5 Extract weight tensor to get its shape dynamically.
    weight_tensor = None
    for attr in weight_const_node.attribute:
        if attr.name == "value":
            weight_tensor = numpy_helper.to_array(attr.t)
            break
    if weight_tensor is None:
        raise ValueError("Unable to extract weight tensor from constant node.")
    weight_shape = list(weight_tensor.shape)
    
    # 5. Build the injection subgraph.
    injection_src = weight_const_node.output[0]
    faulty_weight = injection_src + "_fault"
    
    injection_nodes = create_quantized_fault_injection_weight(
        input_name=injection_src,
        output_name=faulty_weight,
        bit_position=bit_position
    )
    
    # 6. Create a new MatMul node that uses the faulty weight.
    cloned_matmul_output = target_node.output[0] + "_fault"
    cloned_matmul_node = helper.make_node(
        "MatMul",
        inputs=[activation_input, faulty_weight],
        outputs=[cloned_matmul_output],
        name=target_node.name + "_fault"
    )
    
    if "16" in fault_model:
        size = np.random.randint(1, 16)
        mask_nodes = create_weight16_mask(
            matmul_output=cloned_matmul_output,
            masked_output=cloned_matmul_output + "_masked",
            block_length=size
        )
        injection_nodes.extend(mask_nodes)
        cloned_matmul_output = cloned_matmul_output + "_masked"
    
    # 7. Create an Add node that sums the outputs of the original and cloned MatMul.
    final_output = target_node.output[0] + "_final"
    add_node = helper.make_node(
        "Add",
        inputs=[target_node.output[0], cloned_matmul_output],
        outputs=[final_output],
        name=target_node.name + "_Add"
    )
    
    # 8. Update downstream consumers: replace any input equal to target_node.output[0] with final_output.
    for node in model.graph.node:
        new_inputs = []
        for inp in node.input:
            if inp == target_node.output[0]:
                new_inputs.append(final_output)
            else:
                new_inputs.append(inp)
        node.input[:] = new_inputs
    
    # 9. Insert the new nodes into the graph.
    new_nodes = []
    injected = False
    for node in model.graph.node:
        new_nodes.append(node)
        if not injected and node.name == weight_const_node.name:
            new_nodes.extend(injection_nodes)
            injected = True
    new_nodes.append(cloned_matmul_node)
    new_nodes.append(add_node)
    
    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    model = shape_inference.infer_shapes(model)
    model.opset_import[0].version = 18
    onnx.save(model, output_path)
    print(f"Modified WEIGHT injection model saved to {config['decoder_path'].replace('.onnx', '_injected.onnx')}")
    return output_path

def modify_onnx_graph_random(config, fault_model, bit_position=None):

    model_path = config["decoder_path"]
    output_path = config.get("output_path", model_path.replace(".onnx", "_random.onnx"))
    matmul_pattern = config["matmul_node"]

    model = onnx.load(model_path)
    model = patch_reduce_ops(model, reduce_ops=("ReduceMean", "ReduceMax"))
    model = shape_inference.infer_shapes(model)

    target_node = None
    for node in model.graph.node:
        if node.op_type == 'MatMul' and matmul_pattern in node.name:
            target_node = node
            break
    if not target_node:
        raise ValueError(f"MatMul node with pattern {matmul_pattern} not found")

    matmul_output = target_node.output[0]
    consumers = defaultdict(list)
    for node in model.graph.node:
        for inp in node.input:
            consumers[inp].append(node)
    # (downstream_nodes not used further)

    if "BITFLIP" in fault_model:
        injection_nodes = create_random_bitflip_injection(
            output_name=matmul_output,
            bit_position=bit_position
        )
    else:
        injection_nodes = create_random_fault_injection(
            output_name=matmul_output,
            random_value=delta_init()
        )
    

    new_nodes = []
    faulty_output = f"{matmul_output}_faulty"

    for node in model.graph.node:
        if node == target_node:
            new_nodes.append(node)  
            new_nodes.extend(injection_nodes)  
        else:
            if matmul_output in node.input:
                new_inputs = [
                    faulty_output if inp == matmul_output else inp 
                    for inp in node.input
                ]
                new_node = helper.make_node(
                    node.op_type,
                    new_inputs,
                    node.output,
                    node.name
                )
                new_nodes.append(new_node)
            else:
                new_nodes.append(node)
    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)



    model = shape_inference.infer_shapes(model)
    model.opset_import[0].version = 18
    onnx.save(model, output_path)
    print(f"Modified random fault injection model saved to {output_path}")
    return output_path

            
if __name__ == "__main__":
    config = {
        "round_node": "/self_attn/k_proj/Round",
        "matmul_node": "/self_attn/k_proj/MatMul",
        "weight_node": "onnx::MatMul_390_const",
        "decoder_path": "decoders/decoder-merge-20.onnx"
    }
    fault_model = "WEIGHT"  # or "WEIGHT16"
    bit_position = 3
    # modify_onnx_graph_input(config, fault_model, bit_position)
    # print('doing injection')
    modify_onnx_graph_weight(config, fault_model, bit_position)
    # fault_model = "RANDOM"
    # modify_onnx_graph_random(config, fault_model)
    
    
    # # RUNNING ONNXRUNTIME
    # N = 128               
    # sumN = 256             
    # lastN = 32            
    # Concatpast_key_dim_2 = 64   
    # Concatpast_value_dim_2 = 64 


    # hidden_in = np.random.randn(1, N, 4096).astype(np.float16)
    # attn_mask = np.random.randn(1, 1, N, sumN).astype(np.float16)
    # position_ids = np.random.randint(0, N, size=(1, N)).astype(np.int64)
    # past_key_in = np.random.randn(1, 32, lastN, 128).astype(np.float16)
    # past_value_in = np.random.randn(1, 32, lastN, 128).astype(np.float16)

  
    # inputs = {
    #     "hidden_in": hidden_in,
    #     "attn_mask": attn_mask,
    #     "position_ids": position_ids,
    #     "past_key_in": past_key_in,
    #     "past_value_in": past_value_in
    # }


    # # session = ort.InferenceSession('modified_model_16.onnx')
    # session = ort.InferenceSession('try/decoder-merge-20.onnx')

    # # Run inference
    # outputs = session.run(None, inputs)


 
    # # - hidden_out: float16[1, N, 4096]
    # # - past_key: float16[1, 32, Concatpast_key_dim_2, 128]
    # # - past_value: float16[1, 32, Concatpast_value_dim_2, 128]
    # for i, output in enumerate(outputs):
    #     print(f"Output {i} shape: {output.shape}")
    #     print(output)
    
    # num_of_decoders = 20
    
    # input_data = {
    #     "hidden_in": np.random.rand(1, 1, 512).astype(np.float16),
    #     "attn_mask": np.random.rand(1, 72, 512).astype(np.float16),
    #     "position_ids": np.random.randint(0, size=(1, 128)).astype(np.int64),
    #     "past_key_in": np.random.randint(0, 10, size=(1, 1, 1)).astype(np.int64),
    #     "past_value_in": np.random.randint(0, 10, size=(1, 1, 1)).astype(np.int64),
    # }
    # gpu_inputs = {}
    # for name, data in input_data.items():
    #     gpu_inputs[name] = torch.from_numpy(data).cuda()
    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    # outputs = None
    # for decoder_idx in range(num_of_decoders):
    #     model_path = model_path_template.format(decoder_idx)
    #     session = ort.InferenceSession(model_path, providers=providers)
        
        # io_binding = session.io_binding()
        
        # for input_name in session.get_inputs():
        #     name = input_name.name
        #     tensor = gpu_inputs[name].contiguous()
        #     io_binding.bind_input(
        #         name=name,
        #         device_type='cuda',
        #         device_id=0,
        #         element_type=np.float32,
        #         shape=tuple(tensor.shape),
        #         buffer_ptr=tensor.data_ptr()
        #     )
        
        # for output in session.get_outputs():
        #     io_binding.bind_output(output.name, 'cuda')
        
        # # Run inference
        # session.run_with_iobinding(io_binding)
        
        # outputs = io_binding.get_outputs()
        
        # gpu_inputs.update({
        #     'input_tensor': outputs[0]  # Assuming first output feeds into next decoder
        # })
    
    # # Convert final outputs to CPU if needed
    # cpu_outputs = [out.numpy() for out in outputs]