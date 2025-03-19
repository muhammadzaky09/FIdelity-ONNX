import os
import numpy as np
import onnxruntime as ort

class BitFlipOp:
    def __init__(self):
        # Find the path to the shared library
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(current_dir, 'onnx_bitflip.so')
        
        if not os.path.exists(lib_path):
            # Try to find in site-packages
            import site
            for site_dir in site.getsitepackages():
                candidate = os.path.join(site_dir, 'onnx_bitflip.so')
                if os.path.exists(candidate):
                    lib_path = candidate
                    break
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Could not find onnx_bitflip.so library. Current dir: {current_dir}")
            
        # Register the custom op
        self.sess_options = ort.SessionOptions()
        self.sess_options.register_custom_ops_library(lib_path)
        print(f"Registered custom ops library from {lib_path}")

    def create_model(self, input_shape, dtype=np.float32):
        """
        Create an ONNX model with the BitFlip operator
        
        Args:
            input_shape: Shape of the input tensor
            dtype: Data type (np.float32 or np.float16)
        
        Returns:
            onnx_model: ONNX model bytes
        """
        import onnx
        from onnx import helper, TensorProto
        
        # Determine element type for ONNX
        if dtype == np.float32:
            elem_type = TensorProto.FLOAT
        elif dtype == np.float16:
            elem_type = TensorProto.FLOAT16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        # Create inputs
        input_tensor = helper.make_tensor_value_info(
            'input', elem_type, input_shape
        )
        bit_position = helper.make_tensor_value_info(
            'bit_position', TensorProto.INT32, [1]
        )
        
        # Create outputs
        output_tensor = helper.make_tensor_value_info(
            'output', elem_type, input_shape
        )
        
        # Create node
        node = helper.make_node(
            'BitFlip',         # Op type
            ['input', 'bit_position'],  # Inputs
            ['output'],        # Outputs
            domain='contrib.bitflip'
        )
        
        # Create graph
        graph = helper.make_graph(
            [node],
            'BitFlipModel',
            [input_tensor, bit_position],
            [output_tensor]
        )
        
        # Create model
        model = helper.make_model(
            graph,
            producer_name='BitFlip_Producer',
            opset_imports=[
                helper.make_opsetid('', 14),
                helper.make_opsetid('contrib.bitflip', 1)
            ]
        )
        
        # Return the serialized model
        return model.SerializeToString()
    
    def run(self, input_data, bit_position, use_gpu=False):
        """
        Run the BitFlip operator
        
        Args:
            input_data: Input tensor (numpy array)
            bit_position: Bit position to flip (integer)
            use_gpu: Whether to use GPU
        
        Returns:
            output: Result tensor
        """
        # Create model
        model_bytes = self.create_model(input_data.shape, input_data.dtype)
        
        # Set providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        # Create session
        session = ort.InferenceSession(model_bytes, self.sess_options, providers=providers)
        
        # Convert bit position to numpy array
        bit_pos_np = np.array([bit_position], dtype=np.int32)
        
        # Run inference
        outputs = session.run(None, {
            'input': input_data,
            'bit_position': bit_pos_np
        })
        
        return outputs[0]