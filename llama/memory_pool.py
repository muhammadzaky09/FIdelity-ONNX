from loguru import logger
from .utils import singleton
import onnxruntime as ort
import numpy as np
import os
import sys
import cupy as cp
import psutil
import math

class OrtWrapper:
    def __init__(self, onnxfile: str):
        assert os.path.exists(onnxfile)
        self.onnxfile = onnxfile
        self.sess = ort.InferenceSession(onnxfile,
                                         providers=['CUDAExecutionProvider'],
                                         provider_options=[{'device_id': 0}]
                                         )
        self.io_binding = self.sess.io_binding()
        self.inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.output_names = [output.name for output in outputs]
        logger.debug('{} loaded'.format(onnxfile))
    
    def _get_ort_type(self, dtype):
        if dtype == cp.float32 or dtype == np.float32:
            return np.float32
        elif dtype == cp.float64 or dtype == np.float64:
            return np.float64
        elif dtype == cp.int32 or dtype == np.int32:
            return np.int32
        elif dtype == cp.int64 or dtype == np.int64:
            return np.int64
        elif dtype == cp.uint8 or dtype == np.uint8:
            return np.uint8
        elif dtype == cp.int8 or dtype == np.int8:
            return np.int8
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def forward(self, _inputs: dict):
        assert len(self.inputs) == len(_inputs)
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()

        # Bind input tensors (assuming they are on GPU)
        for name, tensor in _inputs.items():
            self.io_binding.bind_input(
                name=name,
                device_type='cuda',
                device_id=0,
                element_type=self._get_ort_type(tensor.dtype),
                shape=tensor.shape,
                buffer_ptr=tensor.data.ptr
            )

        # Pre-allocate GPU buffers for model outputs.
        # For autoregressive models, each forward pass typically produces one new token,
        # but for the embed model, the output shape depends on the input sequence length.
        output_buffers = {}
        batch_size = list(_inputs.values())[0].shape[0]
        # Get the sequence length from one of the inputs (e.g. input_ids)
        seq_len = list(_inputs.values())[0].shape[1]

        for name in self.output_names:
            # Decide expected shape based on the output name
            if name == 'embed':
                # For the embedding model, output shape is (batch_size, seq_len, embed_dim)
                embed_dim = 4096  # Adjust if necessary
                expected_shape = (batch_size, seq_len, embed_dim)
                expected_dtype = np.float32
            else:
                # For other outputs (like decoder logits), assume one token is generated
                vocab_size = 32000  # Adjust as needed from your tokenizer/model configuration
                expected_shape = (batch_size, 1, vocab_size)
                expected_dtype = np.float32

            output_buffers[name] = ort.OrtValue.ortvalue_from_shape_and_type(
                expected_shape, expected_dtype, 'cuda', 0
            )
            self.io_binding.bind_ortvalue_output(name, output_buffers[name])

        print('binded')
        # Run the model using IOBinding.
        self.sess.run_with_iobinding(self.io_binding)

        # Retrieve outputs directly from the pre-allocated GPU buffers.
        # You can now use them without copying data from CPU back to GPU.
        outputs = {}
        for name in self.output_names:
            # For example, converting to a CuPy array; alternatively, use the OrtValue directly.
            outputs[name] = cp.asarray(output_buffers[name].numpy())
        return outputs

    
    def __del__(self):
        logger.debug('{} unload'.format(self.onnxfile))


@singleton
class MemoryPoolSimple:
    def __init__(self, maxGB):
        if maxGB < 0:
            raise Exception('maxGB must > 0, get {}'.format(maxGB))
        
        self.max_size = maxGB * 1024 * 1024 * 1024
        self.wait_map = {}
        self.active_map = {}

    def submit(self, key: str, onnx_filepath: str):
        if not os.path.exists(onnx_filepath):
            raise Exception('{} not exist!'.format(onnx_filepath))

        if key not in self.wait_map:
            self.wait_map[key] = {
                'onnx': onnx_filepath,
                'file_size': os.path.getsize(onnx_filepath)
            }

    def used(self):
        sum_size = 0
        biggest_k = None
        biggest_size = 0
        for k in self.active_map.keys():
            cur_size = self.wait_map[k]['file_size']
            sum_size += cur_size

            if biggest_k is None:
                biggest_k = k
                biggest_size = cur_size
                continue
            
            if cur_size > biggest_size:
                biggest_size = cur_size
                biggest_k = k
        
        return sum_size, biggest_k

    def check(self):
        sum_need = 0
        for k in self.wait_map.keys():
            sum_need = sum_need + self.wait_map[k]['file_size']
            
        sum_need /= (1024 * 1024 * 1024)
        
        total = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        if total > 0 and total < sum_need:
            logger.warning('virtual_memory not enough, require {}, try `--poolsize {}`'.format(sum_need, math.floor(total)))


    def fetch(self, key: str):
        if key in self.active_map:
            return self.active_map[key]
        
        need = self.wait_map[key]['file_size']
        onnx = self.wait_map[key]['onnx']

        # check current memory use
        used_size, biggest_k = self.used()
        while biggest_k is not None and self.max_size - used_size < need:
            # if exceeded once, delete until `max(half_max, file_size)` left
            need = max(need, self.max_size / 2)
            if len(self.active_map) == 0:
                break

            del self.active_map[biggest_k]
            used_size, biggest_k = self.used()
        
        self.active_map[key] = OrtWrapper(onnx)
        return self.active_map[key]