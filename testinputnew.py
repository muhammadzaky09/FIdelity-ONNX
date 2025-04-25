import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
from inject_ops import create_quantized_fault_injection
import struct

def build_test_model(bit_position=3, is_signed=True):
    # Create model input/output
    input_name = "input"
    output_name = "perturbation"
    
    # Generate nodes for bit flipping
    nodes = create_quantized_fault_injection(
        input_name=input_name,
        output_name=output_name,
        bit_position=bit_position,
        fp32=False,
        is_signed=is_signed
    )
    
    # Define input and output tensors
    inputs = [helper.make_tensor_value_info(input_name, TensorProto.INT8 if is_signed else TensorProto.UINT8, [1, 2, 3])]
    outputs = [helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, [1, 2, 3])]
    
    # Create graph and model
    graph = helper.make_graph(nodes, "BitFlipTest", inputs, outputs)
    model = helper.make_model(graph, producer_name="BitFlipTester")
    model.opset_import[0].version = 18
    
    # Save model
    onnx.save(model, "bit_flip_test.onnx")
    return "bit_flip_test.onnx"

def run_test(model_path, bit_position, is_signed=True):
    # Create test input data
    if is_signed:
        # INT8 range: -128 to 127
        data = np.array([[[-1, 0, 1], [64, -64, 127]]], dtype=np.int8)
    else:
        # UINT8 range: 0 to 255
        data = np.array([[[1, 2, 3], [128, 192, 255]]], dtype=np.uint8)
    
    # Create session and run inference
    session = ort.InferenceSession(model_path)
    inputs = {session.get_inputs()[0].name: data}
    perturbation = session.run(None, inputs)[0]
    
    # Verify results
    print(f"\nInput data:\n{data}")
    print(f"\nPerturbation (result of bit flip at position {bit_position}):\n{perturbation}")
    
    # Check which values changed (nonzero perturbation)
    changed = np.abs(perturbation) > 0
    print(f"\nChanged elements (should be exactly one):\n{changed}")
    print(f"Number of changed elements: {np.sum(changed)}")
    
    # For the changed element, calculate what happened
    if np.sum(changed) > 0:
        indices = np.argwhere(changed)
        idx = tuple(indices[0])
        original = int(data[idx])
        
        # Manually calculate expected result after bit flip
        bit_mask = 1 << bit_position
        expected_int = original ^ bit_mask
        
        # For signed INT8, handle sign extension
        if is_signed and original >= 0 and expected_int < 0:
            expected_float = -(-expected_int & 0xFF)
        elif is_signed and original < 0 and expected_int >= 0:
            expected_float = expected_int
        else:
            expected_float = expected_int
        
        print(f"\nDetailed analysis of changed element at {idx}:")
        print(f"Original value: {original} (binary: {format(original & 0xFF, '08b')})")
        print(f"Bit position to flip: {bit_position}")
        print(f"Expected after bit flip: {expected_int} (binary: {format(expected_int & 0xFF, '08b')})")
        print(f"Actual perturbation: {perturbation[idx]}")
        print(f"Expected perturbation: {float(expected_float - original)}")
        
        diff = abs(perturbation[idx] - float(expected_float - original))
        if diff < 0.001:
            print("\n✓ Test PASSED: Perturbation matches expected value")
        else:
            print("\n✗ Test FAILED: Perturbation doesn't match expected value")
    else:
        print("\n✗ Test FAILED: No bits were flipped")
    
    return np.sum(changed) == 1

if __name__ == "__main__":
    # Test for INT8 (signed)
    print("=" * 50)
    print("Testing INT8 (signed) with bit position 3")
    model_path = build_test_model(bit_position=7, is_signed=True)
    run_test(model_path, bit_position=7, is_signed=True)
    
    # Test for UINT8 (unsigned)
    print("\n" + "=" * 50)
    print("Testing UINT8 (unsigned) with bit position 5")
    model_path = build_test_model(bit_position=7, is_signed=False)
    run_test(model_path, bit_position=7, is_signed=False)