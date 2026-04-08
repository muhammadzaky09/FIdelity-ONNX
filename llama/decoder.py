import onnxruntime as ort
import numpy as np
import os
from loguru import logger
from .memory_pool import MemoryPoolSimple

class Decoder:

    def __init__(self, pool: MemoryPoolSimple, onnxdir: str, nameformat: str, count: int = 32,
                 embed_file: str = 'embed.onnx',
                 norm_file: str = 'norm.onnx',
                 head_file: str = 'head.onnx',
                 embed_input: str = 'input',
                 embed_output: str = 'embed',
                 norm_input: str = 'input',
                 norm_output: str = 'output',
                 head_input: str = 'input',
                 head_output: str = 'output'):

        assert os.path.isdir(onnxdir)
        self._pool = pool

        for idx in range(count):
            filepath = os.path.join(onnxdir, nameformat.format(idx))
            self._pool.submit('decode{}'.format(idx), filepath)

        self._pool.submit('embed', os.path.join(onnxdir, embed_file))
        self._pool.submit('norm',  os.path.join(onnxdir, norm_file))
        self._pool.submit('head',  os.path.join(onnxdir, head_file))

        self._embed_input  = embed_input
        self._embed_output = embed_output
        self._norm_input   = norm_input
        self._norm_output  = norm_output
        self._head_input   = head_input
        self._head_output  = head_output

    def decode(self, _inputs: dict, idx: int):
        key = 'decode{}'.format(idx)
        handler = self._pool.fetch(key)
        outputs = handler.forward(_inputs)
        
        return outputs

    def embed(self, input_ids: np.array):
        handler = self._pool.fetch('embed')
        return handler.forward({self._embed_input: input_ids})[self._embed_output]

    def norm_head(self, hidden: np.array):
        handler = self._pool.fetch('norm')
        output = handler.forward({self._norm_input: hidden})[self._norm_output]

        handler = self._pool.fetch('head')
        output = handler.forward({self._head_input: output})[self._head_output]
        return output
