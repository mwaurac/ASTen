from . import _C
import numpy as np

__version__ = "0.1.0"

class Tensor:
    def __init__(self, data, dtype='float32', device='cpu', requires_grad=False):
        if isinstance(data, _C.Tensor):
            self._tensor = data
        elif isinstance(data, np.ndarray):
            self._tensor = _C.Tensor(data, device)
        elif isinstance(data, (list, tuple)):
            arr = np.array(data, dtype=self._numpy_dtype(dtype))
            self._tensor = _C.Tensor(arr, device)
        else:
            raise TypeError(f"Cannot create tensor from {type(data)}")
        
        if requires_grad:
            self._tensor.requires_grad = True
    
    @staticmethod
    def _numpy_dtype(dtype_str):
        dtype_map = {
            'float32': np.float32,
            'float64': np.float64,
            'int32': np.int32,
            'int64': np.int64,
        }
        return dtype_map.get(dtype_str, np.float32)
    
    @property
    def shape(self):
        return tuple(self._tensor.shape)
    
    @property
    def ndim(self):
        return self._tensor.ndim
    
    @property
    def dtype(self):
        return self._tensor.dtype
    
    @property
    def requires_grad(self):
        return self._tensor.requires_grad

    
    def numpy(self):
        return self._tensor.numpy()
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"

    # View ops
    def view(self, shape):
        return Tensor(self._tensor.view(list(shape)))
    def reshape(self, shape):
        return Tensor(self._tensor.reshape(list(shape)))

    def permute(self, *dims):
        return Tensor(self._tensor.permute(list(dims)))


def tensor(data, dtype='float32', device='cpu', requires_grad=False):
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


__all__ = [
    'Tensor',
    'zeros',
    'ones',
    'tensor',
]