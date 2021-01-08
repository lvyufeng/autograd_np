from .module import Module
from autograd import Tensor, Parameter, np
from ..functional import dropout

class Dropout(Module):
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        
    def forward(self, inputs: Tensor) -> Tensor:
        return dropout(inputs, self.dropout_ratio, self.training)
