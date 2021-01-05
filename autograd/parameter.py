from autograd.tensor import Tensor
from autograd.np import np

class Parameter(Tensor):
    def __init__(self, *shape) -> None:
        data = np.random.randn(*shape)
        super().__init__(data,requires_grad = True)