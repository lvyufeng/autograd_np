"""
Optimizers go there
"""

from autograd.module import Module
from autograd.np import np
from autograd.tensor import Tensor
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
                self.v.append(Tensor(np.zeros_like(parameter.data,dtype=np.float32)))
        for idx, parameter in enumerate(self.parameters):
            self.v[idx] = self.momentum * self.v[idx] - self.lr * parameter.grad
            parameter += self.v[idx]

class AdaGrad(Optimizer):
    def __init__(self, parameters, lr:float=0.01) -> None:
        super().__init__(parameters)
        self.lr = lr
        self.h = None

    def step(self) -> None:
        if self.h is None:
            self.h = []
            for parameter in self.parameters:
                self.h.append(Tensor(np.zeros_like(parameter.data,dtype=np.float32)))
        for idx, parameter in enumerate(self.parameters):
            self.h[idx] += parameter.grad * parameter.grad
            parameter -= self.lr * parameter.grad * (1 / (np.sqrt(self.h[idx].data) + 1e-7))

class RMSprop(Optimizer):
    '''
    RMSprop
    '''
    def __init__(self, parameters, lr=0.01, decay_rate = 0.99):
        super().__init__(parameters)
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def step(self):
        if self.h is None:
            self.h = []
            for parameter in self.parameters:
                self.h.append(Tensor(np.zeros_like(parameter.data, dtype=np.float32)))

        for idx, parameter in enumerate(self.parameters):
            self.h[idx] *= self.decay_rate
            self.h[idx] += (1 - self.decay_rate) * (parameter.grad * parameter.grad)
            parameter -= self.lr * parameter.grad * (1 / (np.sqrt(self.h[idx].data) + 1e-7))

class Adam(Optimizer):
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def step(self):
        if self.m is None:
            self.m, self.v = [], []
            for parameter in self.parameters:
                self.m.append(Tensor(np.zeros_like(parameter.data, dtype=np.float32)))
                self.v.append(Tensor(np.zeros_like(parameter.data, dtype=np.float32)))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for idx, parameter in enumerate(self.parameters):
            self.m[idx] += (1 - self.beta1) * (parameter.grad - self.m[idx])
            self.v[idx] += (1 - self.beta2) * (parameter.grad * parameter.grad - self.v[idx])
            
            parameter -= Tensor(lr_t) * self.m[idx] * (1 / (np.sqrt(self.v[idx].data) + 1e-7))
