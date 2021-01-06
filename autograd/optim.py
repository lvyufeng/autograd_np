"""
Optimizers go there
"""

from autograd.module import Module
from autograd.np import np

class Optimizer:
    def __init__(self, parameters):
        self.parameters = []
        self.add_param_group(parameters)
        
    def step(self):
        raise NotImplementedError

    def add_param_group(self, param_group):
        for group in param_group:
            self.parameters.append(group)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

class SGD(Optimizer):
    def __init__(self, parameters, lr:float = 0.01) -> None:
        super(SGD,self).__init__(parameters)
        self.lr = lr
    def step(self) -> None:
        for parameter in self.parameters:
            parameter -= parameter.grad * self.lr

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
                self.v.append(np.zeros_like(parameter))
        for idx, parameter in enumerate(self.parameters):
            self.v[idx] = self.momentum * self.v[idx] - self.lr * parameter.grad
            parameter += self.v[idx]



