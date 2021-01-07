from .tensor import Tensor
from .parameter import Parameter
from .module import Module
from .utils import *

def argmax(x:'Tensor', axis, keepdims=False):
    return Tensor(np.argmax(x.data, axis=axis, keepdims=keepdims))