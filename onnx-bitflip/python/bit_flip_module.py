import os
import numpy as np
import onnxruntime as ort
import ctypes
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BitFlipOp")

class BitFlipOp:
    def __init__(self):
        # Find the path to the shared library
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(current_dir, 'onnx_perturb.so')
        
        if not os.path.exists(lib_path):
            # Try to find in build directory
            parent_dir = os.path.dirname(current_dir)
            lib_path = os.path.join(parent_dir, 'build', 'onnx_perturb.so')
            
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Could not find onnx_bitflip.so library at {lib_path}")
        
        logger.info(f"Loading custom op library from: {lib_path}")
        
        # Check if the library exists and is valid
        try:
            lib = ctypes.CDLL(lib_path)
            # Look for the RegisterCustomOps symbol
            if hasattr(lib, 'RegisterCustomOps'):
                logger.info("Found RegisterCustomOps symbol in library")
            else:
                logger.warning("RegisterCustomOps symbol not found in library")
        except Exception as e:
            logger.error(f"Error loading library: {e}")
        
        # Register the custom op with ONNXRuntime
        try:
            self.sess_options = ort.SessionOptions()
            
            
            # logger.info("Registering custom op library")
            # Register the library
            self.sess_options.register_custom_ops_library(lib_path)
            # logger.info("Successfully registered custom ops library")
        except Exception as e:
            logger.error(f"Error registering custom ops library: {e}")
            raise

    def create_model(self, input_shape):
        """
        Create an ONNX model with the BitFlip operator for FP16
        
        Args:
            input_shape: Shape of the input tensor
        
        Returns:
            onnx_model: ONNX model bytes
        """
        import onnx
        from onnx import helper, TensorProto
        
        # logger.info(f"Creating ONNX model with input shape {input_shape}")
        
        # Use FP16
        elem_type = TensorProto.FLOAT16
        
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
            'Perturb',         # Op type
            ['input', 'bit_position'],  # Inputs
            ['output'],        # Outputs
            domain='custom.bitflip'  # Domain
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
                helper.make_opsetid('custom.bitflip', 1)
            ]
        )
        
        # logger.info("ONNX model created successfully")
        return model.SerializeToString()
    
    def run(self, input_data, bit_position, use_gpu=False):
        """
        Run the BitFlip operator
        
        Args:
            input_data: Input tensor (numpy array in FP16)
            bit_position: Bit position to flip (integer)
            use_gpu: Whether to use GPU
        
        Returns:
            output: Result tensor
        """
        # Ensure input is FP16
        if input_data.dtype != np.float16:
            # logger.info(f"Converting input from {input_data.dtype} to float16")
            input_data = input_data.astype(np.float16)
        
        # logger.info(f"Running BitFlip with bit position {bit_position}")
        
        # Create model
        model_bytes = self.create_model(input_data.shape)
        
        # Set providers
        providers = ['CUDAExecutionProvider'] 
        logger.info(f"Using providers: {providers}")
        
        try:
            # Create session with detailed logging
            # logger.info("Creating inference session")
            session = ort.InferenceSession(
                model_bytes, 
                self.sess_options, 
                providers=providers
            )
            
            # Convert bit position to numpy array
            bit_pos_np = np.array([bit_position], dtype=np.int32)
            
            # Prepare inputs
            feeds = {
                'input': input_data,
                'bit_position': bit_pos_np
            }
            
            # Run inference
            # logger.info("Running inference")
            outputs = session.run(None, feeds)
            # logger.info("Inference completed successfully")
            
            return outputs[0]
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise