from .optimizer import Optimizer
from ..tensor import Tensor
from ..np import np

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
                self.m.append(Tensor(np.zeros_like(parameter.data, dtype=parameter.data.dtype)))
                self.v.append(Tensor(np.zeros_like(parameter.data, dtype=parameter.data.dtype)))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for idx, parameter in enumerate(self.parameters):
            self.m[idx] += (1 - self.beta1) * (parameter.grad - self.m[idx])
            self.v[idx] += (1 - self.beta2) * (parameter.grad * parameter.grad - self.v[idx])
            
            parameter -= Tensor(lr_t) * self.m[idx] * (1 / (np.sqrt(self.v[idx].data) + 1e-7))
