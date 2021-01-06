from autograd.module import Module
from autograd.parameter import Parameter
from autograd.tensor import Tensor

class Linear(Module):
    """
    y = x * W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        self.bias = bias
        self.w = Parameter(in_features, out_features)
        self.b = Parameter(out_features)
        
    def forward(self, inputs: Tensor) -> Tensor:
        if self.bias:
            y = inputs @ self.w + self.b
        else:
            y = inputs @ self.w
        return y