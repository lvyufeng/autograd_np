import unittest
import numpy as np
import cupy as cp
from autograd.tensor import Tensor

class TestTensorMul(unittest.TestCase):
    def test_simple_mul(self):
        t1 = Tensor([1,2,3],requires_grad=True)
        t2 = Tensor([4,5,6],requires_grad=True)

        t3 = t1 * t2

        assert t3.data.tolist() == [4,10,18]
        t3.backward(Tensor([-1,-2,-3]))

        assert t1.grad.data.tolist() == [-4, -10, -18]
        assert t2.grad.data.tolist() == [-1, -4 , -9 ]

        t1 *= 0.1
        assert t1.grad is None
        cp.testing.assert_array_almost_equal(t1.data, [0.1, 0.2, 0.3])
        # assert t1.data.tolist() == [0.1, 0.2, 0.3]

    def test_broadcast_mul(self):
        # broadcasting? A couple of things:
        # IF I do t1 + t2 and t1.shape == t2.shape, it's obvious what to do
        # but I'm alse allowed to add 1s to the beginning of either shape.
        # 
        # t1.shape == (10,5), t2.shape == (5,) => t1 + t2, t2 viewed as (1,5)
        # t2 = [1, 2, 3, 4, 5] => view t2 as [[1,2,3,4,5]]
        # 
        # The second thing I can do, is that if one tensor has a 1 in some dimension
        # I can expand it
        # t1 as (10,5) t2 as (1,5) is [[1,2,3,4,5]]
        t1 = Tensor([[1,2,3],[4,5,6]],requires_grad=True)
        t2 = Tensor([7,8,9],requires_grad=True)

        t3 = t1 * t2

        assert t3.data.tolist() == [[7,16,27],[28,40,54]]
        t3.backward(Tensor([[1,1,1],[1,1,1]]))

        assert t1.grad.data.tolist() == [[7,8,9],[7,8,9]]
        assert t2.grad.data.tolist() == [5, 7, 9]

    def test_broadcast_mul2(self):
        t1 = Tensor([[1,2,3],[4,5,6]],requires_grad=True) # (2,3)
        t2 = Tensor([[7,8,9]],requires_grad=True)         # (1,3)

        t3 = t1 * t2
        assert t3.data.tolist() == [[7,16,27],[28,40,54]]

        t3.backward(Tensor([[1,1,1],[1,1,1]]))

        assert t1.grad.data.tolist() == [[7,8,9],[7,8,9]]
        assert t2.grad.data.tolist() == [[5, 7 ,9]]
