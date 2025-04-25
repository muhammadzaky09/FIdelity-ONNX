import os
import struct
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

def create_fp16_fault_injection(input_name, output_name, bit_position, fp32=False):
    """
    Build a small ONNX graph that injects an FP16 bit-flip perturbation.
    If fp32=True, casts input to FP16 and back to FP32.
    Returns the list of nodes to include.
    """
    nodes = []
    intermediate_input = input_name
    intermediate_output = output_name

    if fp32:
        # Cast input FP32 -> FP16
        fp16_input = input_name + "_fp16"
        nodes.append(helper.make_node(
            'Cast', inputs=[input_name], outputs=[fp16_input], to=TensorProto.FLOAT16
        ))
        intermediate_input = fp16_input
        intermediate_output = output_name + "_fp16"

    # 1. Constant node for bit position
    nodes.append(
        helper.make_node(
            'Constant', inputs=[], outputs=['bit_pos_const'],
            value=helper.make_tensor(
                name='bit_pos_tensor', data_type=TensorProto.INT32,
                dims=[1], vals=[bit_position]
            )
        )
    )
    # 2. Custom perturb operator: outputs the perturbation delta in FP16
    nodes.append(
        helper.make_node(
            'Perturb', inputs=[intermediate_input, 'bit_pos_const'],
            outputs=[intermediate_output], domain='custom.perturb'
        )
    )

    if fp32:
        # Cast output FP16 -> FP32
        nodes.append(helper.make_node(
            'Cast', inputs=[intermediate_output], outputs=[output_name], to=TensorProto.FLOAT
        ))

    return nodes


def test_fp16_fault_injection():
    """Test that the Perturb op returns exactly one FP16 delta corresponding to a single bit flip."""
    # Configuration
    input_name = "input_tensor"
    output_name = "output_delta"
    bit_position = 14  # bit to flip in FP16 layout
    model_file = "fp16_fault_injection_test.onnx"

    # Define model I/O
    input_info = helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, [1,1])
    output_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, [1,1])

    # Build graph
    nodes = create_fp16_fault_injection(input_name, output_name, bit_position)
    graph = helper.make_graph(nodes, 'fp16_fault_test', [input_info], [output_info])
    model = helper.make_model(
        graph, producer_name='perturb_test',
        opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('custom.perturb', 1)]
    )
    onnx.save(model, model_file)

    # Setup ONNX Runtime session with custom op
    sess_opts = ort.SessionOptions()
    sess_opts.register_custom_ops_library('onnx-bitflip/build/onnx_perturb.so')
    sess = ort.InferenceSession(model_file, sess_opts, providers=['CUDAExecutionProvider'])

    # Prepare an all-ones FP16 input
    orig_array = np.random.randn(1,1).astype(np.float16)
    print(f"Original array: {orig_array}")
    outputs = sess.run(None, {input_name: orig_array})
    delta = outputs[0]
    
    nz = np.count_nonzero(delta)
    assert nz == 1, f"Expected 1 non-zero delta, got {nz}."

    # Locate the flipped index
    idx = np.nonzero(delta)
    perturb_val = delta[idx][0]
    orig_val = np.float16(orig_array)

    # Compute expected flipped value and delta
    orig_bits = orig_val.view(np.uint16)
    mask = np.uint16(1 << bit_position)
    flipped_bits = orig_bits ^ mask
    expected_flipped = np.frombuffer(flipped_bits.tobytes(), dtype=np.float16)[0]
    print(f"Expected flipped value: {expected_flipped}")
    expected_delta = np.float16(expected_flipped - orig_val)

    # Validate the perturbation matches expected delta
    print(f"Perturbation value: {perturb_val}, Expected delta: {expected_delta}")
    assert perturb_val == expected_delta, (
        f"Delta mismatch: got {perturb_val}, expected {expected_delta}."
    )

    print("FP16 fault injection test passed.")
    os.remove(model_file)


if __name__ == "__main__":
    test_fp16_fault_injection()