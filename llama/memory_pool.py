from loguru import logger
from .utils import singleton
import onnxruntime as ort
import numpy as np
import os
import sys
import cupy as cp
import psutil
import math

def ortvalue_to_cupy(ort_value, dtype=np.float16):
    """
    Wrap the GPU memory of an OrtValue as a CuPy array without copying.
    """
    # Get pointer, shape, and compute total size in bytes.
    ptr = ort_value.data_ptr()         # pointer as an integer
    shape = tuple(ort_value.shape())    # e.g., (batch_size, 1, vocab_size)
    size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
    
    # Wrap the existing GPU memory. We pass the OrtValue as the owner so it stays alive.
    unowned_mem = cp.cuda.UnownedMemory(ptr, size_bytes, ort_value)
    memptr = cp.cuda.MemoryPointer(unowned_mem, 0)
    
    # Create a CuPy ndarray using the memory pointer.
    return cp.ndarray(shape, dtype=dtype, memptr=memptr)

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
        # Map common Cupy/NumPy dtypes to numpy dtypes expected by ORT.
        if dtype == cp.float32 or dtype == np.float32:
            return np.float32
        elif dtype == cp.uint8 or dtype == np.uint8:
            return np.uint8
        elif dtype == cp.int8 or dtype == np.int8:
            return np.int8
        elif dtype == cp.int64 or dtype == np.int64:
            return np.int64
        elif dtype == cp.float16 or dtype == np.float16:
            return np.float16

        else:
            raise ValueError("Unsupported dtype: {}".format(dtype))

    def forward(self, _inputs: dict):
        assert len(self.inputs) == len(_inputs)
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()

        for name, tensor in _inputs.items():
            self.io_binding.bind_input(
                name=name,
                device_type='cuda',
                device_id=0,
                element_type=self._get_ort_type(tensor.dtype),
                shape=tensor.shape,
                buffer_ptr=tensor.data.ptr
            )

        output_names = [output.name for output in self.sess.get_outputs()]
        for name in output_names:
            self.io_binding.bind_output(name, 'cuda')
        self.sess.run_with_iobinding(self.io_binding)

        # Retrieve outputs by wrapping the GPU buffers directly as CuPy arrays.
        outputs = {}
        ort_outputs = self.io_binding.get_outputs()
        for name, out in zip(self.output_names, ort_outputs):
            outputs[name] = ortvalue_to_cupy(out, dtype=np.float32)
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