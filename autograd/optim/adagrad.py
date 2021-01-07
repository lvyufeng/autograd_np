from .optimizer import Optimizer
from ..tensor import Tensor
from ..np import np

class AdaGrad(Optimizer):
    def __init__(self, parameters, lr:float=0.01) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.h = None

    def step(self) -> None:
        if self.h is None:
            self.h = []
            for parameter in self.parameters:
                self.h.append(Tensor(np.zeros_like(parameter.data,dtype=parameter.data.dtype)))
        for idx, parameter in enumerate(self.parameters):
            self.h[idx] += parameter.grad * parameter.grad
            parameter -= self.lr * parameter.grad * (1 / (np.sqrt(self.h[idx].data) + 1e-7))