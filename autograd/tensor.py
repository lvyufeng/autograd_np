from typing import List,NamedTuple,Callable,Optional,Union
from .np import np
from .utils import to_cpu, to_gpu

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray],np.ndarray]

Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable,dtype=np.float64)

Tensorable = Union['Tensor',float, np.ndarray]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)

class Tensor:
    def __init__(self,
                data: Arrayable,
                requires_grad: bool = False,
                depends_on: List[Dependency] = None
    ) -> None:
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []

        self.shape = self.data.shape
        self.dim = self.data.ndim
        self.size = self.data.size
        self.grad : Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data
        
    @data.setter
    def data(self, new_data:np.ndarray) -> None:
        self._data = new_data
        self.grad = None

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float32))

    def __repr__(self) -> str:
        return f"Tensor({self.data}.requires_grad={self.requires_grad})"

    def __add__(self, other) -> 'Tensor':
        """
        geys called if I do t + other
        """
        return _add(self, ensure_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        """ gets called if I do other + t """
        return _add(ensure_tensor(other),self)

    def __iadd__(self, other) -> 'Tensor':
        """
        when we do t += other
        """
        self.data += ensure_tensor(other).data
        return self

    def __isub__(self, other) -> 'Tensor':
        self.data -= ensure_tensor(other).data
        return self

    def __imul__(self, other) -> 'Tensor':
        self.data *= ensure_tensor(other).data
        return self

    def __mul__(self, other) -> 'Tensor':
        return _mul(self, ensure_tensor(other))

    def __rmul__(self,other) -> 'Tensor':
        return _mul(ensure_tensor(other),self)
    
    def __matmul__(self,other) -> 'Tensor':
        return _matmul(self,other)

    def __neg__(self) -> 'Tensor':
        return _neg(self)
    
    def __sub__(self,other) -> 'Tensor':
        return _sub(self,ensure_tensor(other))
        
    def __rsub__(self, other) -> 'Tensor':
        return _sub(ensure_tensor(other),self)

    def __getitem__(self,idxs) -> 'Tensor':
        return _slice(self,idxs)

    def __eq__(self, other) -> bool:
        return self.data == other.data
    
    def sum(self) -> 'Tensor':
        # raise NotImplementedError
        return tensor_sum(self)

    def mean(self) -> 'Tensor':
        return tensor_mean(self)

    def reshape(self, shape:tuple) -> 'Tensor':
        return tensor_reshape(self, shape)

    def backward(self, grad: 'Tensor' = None ) -> None:
        assert self.requires_grad,"called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must specified for non-0-tensor")

        self.grad.data += grad.data # type: ignore

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))
    def cpu(self):
        return to_cpu(self._data)

    def cuda(self):
        return to_gpu(self._data)

    def argmax(self, dim=None, keepdims=False) -> 'Tensor':
        return _argmax(self, dim, keepdims)

    def argmin(self, dim=None, keepdims=False) -> 'Tensor':
        return _argmin(self, dim, keepdims)

def tensor_sum(t:Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """
    data = t.data.sum()
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor, so each input element
            contributes that much
            """
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t,grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data,
                requires_grad,
                depends_on
        )

def tensor_mean(t:Tensor) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """
    data = t.data.mean()
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor, so each input element
            contributes that much
            """
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t,grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data,
                requires_grad,
                depends_on
        )

def tensor_reshape(t:Tensor, shape:tuple) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """
    data = t.data.reshape(shape)
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor, so each input element
            contributes that much
            """
            return grad.reshape(t.data.shape)

        depends_on = [Dependency(t,grad_fn)]
    else:
        depends_on = []
    
    return Tensor(data,
                requires_grad,
                depends_on
        )

def _add(t1: Tensor, t2:Tensor) -> Tensor:

    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # Idea: [1,2,3] + [4,5,6] => [5,7,9]
            # Handle the broadcasting properly
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis = 0)

            # Sum across broadcasted (but non-added dims)
            # (2,3) + (1,3) => (2,3) grad(2,3)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims = True)

            return grad
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis = 0)
             # Sum across broadcasted (but non-added dims)
            # (2,3) + (1,3) => (2,3) grad(2,3)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims = True)
                    
            
            return grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
        requires_grad,
        depends_on
    )

def _mul(t1: Tensor, t2:Tensor) -> Tensor:
    """
    y = (a + eps) * b = a * b + (eps * b * dL/dy)
    gradient_y = 5
    have dL/dy
    dL/da = dL/dy * dy/da(b)
    """
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data

            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis = 0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims = True)

            return grad
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis = 0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims = True)
                    
            return grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
        requires_grad,
        depends_on
    )

def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)

def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2

def _matmul(t1: Tensor, t2:Tensor) -> Tensor:
    """
    if t1 is (n1,m1) t2 is (m1,m2) then t1 @ t2 is (n1,m2)
    so grad3 is (n1,m2)

    if t3 = t1 @ t2 and grad3 is the gradient of some function wrt t3, then
        grad1 = grad @ t2.T
        grad2 = t1.T @ grad
    """
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
        requires_grad,
        depends_on
    )

def _slice(t: Tensor, *idx) -> Tensor:
    """
    t2 = t1[3:4,4:4]
    """ 
    data = t.data[idx]
    requires_grad = t.requires_grad
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idx] = grad
            return bigger_grad
        depends_on = Dependency(t,grad_fn)
    else:
        depends_on = []

    return Tensor(data,requires_grad,depends_on)

def _argmax(x:'Tensor', dim, keepdims=False) -> Tensor:
    return Tensor(np.argmax(x.data, axis=dim, keepdims=keepdims))

def _argmin(x:'Tensor', dim, keepdims=False) -> Tensor:
    return Tensor(np.argmin(x.data, axis=dim, keepdims=keepdims))