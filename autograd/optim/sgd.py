from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr:float = 0.01) -> None:
        super(SGD,self).__init__(parameters)
        self.lr = lr
    def step(self) -> None:
        for parameter in self.parameters:
            parameter -= parameter.grad * self.lr