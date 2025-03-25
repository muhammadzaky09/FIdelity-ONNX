import numpy as np
from threading import Lock
import onnx
from onnx import helper

def singleton(cls):
    _instance = {}
    _instance_lock = Lock()

    def inner(*args, **kwargs):
        if cls not in _instance:
            with _instance_lock:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


def npsoftmax(x, axis):
    y = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(y) / np.sum(np.exp(y), axis=axis, keepdims=True)


def npmultinominal2D(x):
    assert len(x.shape) == 2

    ret = np.zeros((x.shape[0], 1), dtype=x.dtype)

    for row, pval in enumerate(x):
        ret[row] = np.random.multinomial(1, pval).argmax()

    return ret

def seeded_npmultinomial2D(x, seed=None):
    rng = np.random.RandomState(seed)
    assert len(x.shape) == 2
    ret = np.zeros((x.shape[0], 1), dtype=x.dtype)
    
    for row, pval in enumerate(x):
        try:
            # Attempt the normal multinomial sampling
            ret[row] = rng.multinomial(1, pval).argmax()
        except ValueError:
            # Minimal modification: only correct what is necessary.
            # Replace NaNs with 0, and clip negatives and values >1.
            pval_fixed = np.nan_to_num(pval, nan=0.0)
            pval_fixed = np.clip(pval_fixed, 0, 1)
            
            # Normalize: if sum is zero, fallback to uniform distribution.
            total = pval_fixed.sum()
            if total == 0:
                pval_fixed = np.ones_like(pval_fixed) / len(pval_fixed)
            else:
                pval_fixed = pval_fixed / total
                
            ret[row] = rng.multinomial(1, pval_fixed).argmax()
    return ret

def npgreedy2D(x):
    return np.argmax(x, axis=1).reshape(x.shape[0], 1)


if __name__ == '__main__':
    data = np.ones((12, 8))
    data1 = npsoftmax(data, -1)

    data2 = npmultinominal2D(data1)
    print(data2)