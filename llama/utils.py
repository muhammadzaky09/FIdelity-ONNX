
from threading import Lock
import cupy as cp

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

def cpsoftmax(x, axis):
    y = x - cp.max(x, axis=axis, keepdims=True)
    return cp.exp(y) / cp.sum(cp.exp(y), axis=axis, keepdims=True)


def cpmultinominal2D(x):
    ret = cp.zeros((x.shape[0], 1), dtype=x.dtype)
    for row, pval in enumerate(x):
        ret[row] = cp.random.multinomial(1, pval).argmax()
    return ret

def cpgreedy2D(x):
    return cp.argmax(x, axis=1).reshape(x.shape[0], 1)



# if __name__ == '__main__':
#     data = cp.ones((12, 8))
#     data1 = cpsoftmax(data, -1)

#     data2 = cpmultinominal2D(data1)
#     print(data2)