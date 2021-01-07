from ..np import np
from ..tensor import Tensor, Dependency

def dropout(tensor: Tensor, dropout_ratio:int=0.5, training:bool=True) -> Tensor:
    """
    http://arxiv.org/abs/1207.0580
    """
    requires_grad = tensor.requires_grad
    mask = np.random.rand(*tensor.shape) > dropout_ratio
    if training:
        data = tensor.data * mask
    else:
        data = tensor.data * (1.0 - dropout_ratio)

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * mask
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)
    
def tanh(tensor: Tensor) -> Tensor:
    '''
    tanh = 
    '''
    data = np.tanh(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)

def sigmoid(tensor: Tensor) -> Tensor:
    data = 1 / (1 + np.exp(-tensor.data))
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (data * (1 - data))
        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)