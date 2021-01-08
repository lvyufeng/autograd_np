from .module import Module
from .. import functional as F
from autograd import Tensor

class _Loss(Module):
    def __init__(self) -> None:
        super().__init__()

# class 