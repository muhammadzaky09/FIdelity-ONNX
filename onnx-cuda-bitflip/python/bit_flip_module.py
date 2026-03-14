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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir  = os.path.dirname(current_dir)

        # Repo root is two levels up from this file (python/ → onnx-cuda-bitflip/ → repo)
        repo_root = os.path.dirname(parent_dir)

        # Search directories in priority order:
        #   1. repo llama/  (both BitFlip/custom.bitflip and Perturb/custom.perturb)
        #   2. local build
        #   3. alongside this file
        #   4. every directory already on LD_LIBRARY_PATH
        #   5. legacy server path
        search_dirs = [
            os.path.join(repo_root, "llama"),
            os.path.join(parent_dir, "build"),
            current_dir,
            *os.environ.get("LD_LIBRARY_PATH", "").split(":"),
            "/workspace/onnx-transformer/llama",
        ]

        # Find both libraries — each registers a different op domain:
        #   onnx_bitflip.so  → custom.bitflip : BitFlip   (used by RANDOM_BITFLIP)
        #   onnx_perturb.so  → custom.perturb : Perturb   (used by INPUT/WEIGHT fp16)
        wanted = {"onnx_bitflip.so": None, "onnx_perturb.so": None}
        for d in search_dirs:
            if not d:
                continue
            for name in list(wanted.keys()):
                if wanted[name] is None:
                    candidate = os.path.join(d, name)
                    if os.path.exists(candidate):
                        wanted[name] = candidate
            if all(v is not None for v in wanted.values()):
                break

        found = {name: path for name, path in wanted.items() if path is not None}
        if not found:
            raise FileNotFoundError(
                "Could not find onnx_bitflip.so or onnx_perturb.so in any of: "
                + ", ".join(filter(None, search_dirs))
            )

        self.sess_options = ort.SessionOptions()
        for name, lib_path in found.items():
            logger.info(f"Loading custom op library from: {lib_path}")
            try:
                lib = ctypes.CDLL(lib_path)
                if hasattr(lib, 'RegisterCustomOps'):
                    logger.info(f"Found RegisterCustomOps in {name}")
                else:
                    logger.warning(f"RegisterCustomOps not found in {name}")
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                continue
            try:
                self.sess_options.register_custom_ops_library(lib_path)
            except Exception as e:
                logger.error(f"Error registering {name}: {e}")
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
            domain='custom.perturb'  # Domain
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