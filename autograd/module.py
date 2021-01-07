from .tensor import Tensor
from .parameter import Parameter

from typing import Iterator

import inspect

class Module:
    training:bool
    def __init__(self):
        self.training = True
        
    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        result = self.forward(*input,**kwargs)
        return result

    def children(self) -> Iterator["Module"]:
        memo = set()
        for name, module in inspect.getmembers(self):
            if isinstance(module, Module) and module not in memo:
                memo.add(module)
                yield module

    def train(self, mode:bool=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
    
    def eval(self):
        self.train(False)