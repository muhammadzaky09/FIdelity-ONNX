import os
import numpy as np
import onnxruntime as ort
import struct
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PerturbOp")

class PerturbOp:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(current_dir, 'llama/onnx_perturb.so')  # Assuming new .so file is named onnx_perturb.so
        
        if not os.path.exists(lib_path):
            parent_dir = os.path.dirname(current_dir)
            lib_path = os.path.join(parent_dir, 'build', 'onnx_perturb.so')

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Could not find onnx_perturb.so library at {lib_path}")

        logger.info(f"Loading custom op library from: {lib_path}")

        try:
            self.sess_options = ort.SessionOptions()
            self.sess_options.register_custom_ops_library(lib_path)
        except Exception as e:
            logger.error(f"Error registering custom ops library: {e}")
            raise

    def create_model(self, input_shape):
        """Create an ONNX model with the 'perturb' operator."""
        import onnx
        from onnx import helper, TensorProto

        elem_type = TensorProto.FLOAT16

        # Create inputs
        input_tensor = helper.make_tensor_value_info('input', elem_type, input_shape)
        bit_position = helper.make_tensor_value_info('bit_position', TensorProto.INT32, [1])

        # Create outputs
        output_tensor = helper.make_tensor_value_info('output', elem_type, input_shape)

        # Create perturb node
        node = helper.make_node(
            'Perturb',
            ['input', 'bit_position'],
            ['output'],
            domain='custom.perturb'
        )

        # Create graph
        graph = helper.make_graph(
            [node],
            'PerturbModel',
            [input_tensor, bit_position],
            [output_tensor]
        )

        # Create model
        model = helper.make_model(
            graph,
            producer_name='Perturb_Producer',
            opset_imports=[
                helper.make_opsetid('', 14),
                helper.make_opsetid('custom.perturb', 1)
            ]
        )
        return model.SerializeToString()

    def run(self, input_data, bit_position):
        """Run the 'perturb' operator."""
        if input_data.dtype != np.float16:
            input_data = input_data.astype(np.float16)

        model_bytes = self.create_model(input_data.shape)
        providers = ['CUDAExecutionProvider']

        logger.info(f"Using providers: {providers}")

        try:
            session = ort.InferenceSession(
                model_bytes,
                self.sess_options,
                providers=providers
            )

            bit_pos_np = np.array([bit_position], dtype=np.int32)
            feeds = {
                'input': input_data,
                'bit_position': bit_pos_np
            }

            outputs = session.run(None, feeds)
            return outputs[0]
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise


def display_binary(value):
    """Return the 16-bit binary representation of a float16 value."""
    return format(np.float16(value).view(np.uint16), '016b')


def test_large_tensor():
    perturb_op = PerturbOp()

    # Define input tensor shapes for testing (2D, 3D, 4D)
    test_shapes = [
        (10, 10),         # 2D tensor
        (5, 5, 5),        # 3D tensor
        (4, 4, 4, 4)      # 4D tensor
    ]

    bit_position = 14  # Flip a bit in the exponent or mantissa

    for shape in test_shapes:
        logger.info(f"Testing tensor of shape: {shape}")

        # Generate FP16 input tensor
        input_data = np.random.uniform(-100, 100, shape).astype(np.float16)

        # Run the perturb operator
        output = perturb_op.run(input_data, bit_position)

        # Count nonzero elements
        nonzero_count = np.count_nonzero(output)
        assert nonzero_count == 1, f"FAIL: Expected exactly 1 nonzero element, got {nonzero_count}"

        # Find the perturbed index
        perturbed_index = np.argwhere(output != 0)
        assert perturbed_index.shape[0] == 1, "FAIL: More than one perturbed index found."

        # Extract original and perturbed values
        perturbed_idx = tuple(perturbed_index[0])
        original_value = input_data[perturbed_idx]
        perturbed_value = output[perturbed_idx]

        # Expected delta from direct bit manipulation
        expected_value_bits = np.float16(original_value).view(np.uint16) ^ (1 << bit_position)
        expected_delta = np.frombuffer(struct.pack('H', expected_value_bits), dtype=np.float16)[0] - original_value

        # Binary representations
        original_bin = display_binary(original_value)
        expected_bin = display_binary(expected_delta)
        computed_bin = display_binary(perturbed_value)

        logger.info(f"Injected at index {perturbed_idx}")
        logger.info(f"Original FP16 value: {original_value} (bin: {original_bin})")
        logger.info(f"Computed delta from op: {perturbed_value} (bin: {computed_bin})")
        logger.info(f"Expected delta:       {expected_delta} (bin: {expected_bin})")

        assert np.isclose(perturbed_value, expected_delta, atol=1e-3), "FAIL: Mismatch in delta value."

        logger.info(f"PASS: Test passed for shape {shape} with bit position {bit_position}.\n")


if __name__ == '__main__':
    test_large_tensor()
