from ..np import np
from ..tensor import Tensor
from .optimizer import Optimizer
class Momentum(Optimizer):
    def __init__(self, parameters, lr:float=0.01, momentum=0.9) -> None:
        super(Momentum, self).__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self) -> None:
        if self.v is None:
            self.v = []
            for parameter in self.parameters:
                self.v.append(Tensor(np.zeros_like(parameter.data,dtype=np.float32)))
        for idx, parameter in enumerate(self.parameters):
            self.v[idx] = self.momentum * self.v[idx] - self.lr * parameter.grad
            parameter += self.v[idx]