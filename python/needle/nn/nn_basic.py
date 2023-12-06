"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import pdb

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        import pdb
        #pdb.set_trace()
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device = device, dtype = dtype, requires_grad = True))
        self.bias = None
        if bias:
            self.bias = Parameter(ops.transpose(init.kaiming_uniform(out_features, 1, device = device, dtype = dtype, requires_grad = True)), requires_grad = True)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = X @ self.weight
        if self.bias is not None:
            # bias = ops.reshape(self.bias.data, (1, self.out_features))
            y = y + ops.broadcast_to(self.bias, y.shape)

        return y
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        flat_dim = 1
        for dim in X.shape[1:]:
            flat_dim *= dim

        return ops.reshape(X, (X.shape[0], flat_dim))
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x
        for module in self.modules:
            y = module(y)

        return y
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        #pdb.set_trace()
        zy = ops.summation(logits * init.one_hot(logits.shape[1], y, device = logits.device), axes = (1,))
        losses = ops.logsumexp(logits, axes = (1,)) - zy
        mean_loss = ops.summation(losses) / logits.shape[0]

        return mean_loss
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device = device, dtype = dtype, requires_grad = True))
        self.bias = Parameter(init.zeros(dim, device = device, dtype = dtype, requires_grad = True))
        self.running_mean = init.zeros(dim, device = device, dtype = dtype)
        self.running_var = init.ones(dim, device = device, dtype = dtype)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            E_x = ops.summation(x, (0,)) / x.shape[0]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * E_x.data
            E_x = ops.broadcast_to(ops.reshape(E_x, (1, self.dim)), x.shape)

            diff = x - E_x

            Var_x = ops.summation(ops.power_scalar(diff, 2), (0,)) / x.shape[0]
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * Var_x.data

        else:
            E_x = self.running_mean
            E_x = ops.broadcast_to(ops.reshape(E_x, (1, self.dim)), x.shape)
            diff = x - E_x
            Var_x = self.running_var

        Var_x = ops.broadcast_to(ops.reshape(Var_x, (1, self.dim)), x.shape)

        w = ops.broadcast_to(ops.reshape(self.weight, (1, x.shape[1])), x.shape)
        b = ops.broadcast_to(ops.reshape(self.bias, (1, x.shape[1])), x.shape)

        y = w * (diff / ops.power_scalar(Var_x + self.eps, 0.5)) + b

        return y
        # raise NotImplementedError()
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device = device, dtype = dtype, requires_grad = True))
        self.bias = Parameter(init.zeros(dim, device = device, dtype = dtype, requires_grad = True))
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        E_x = ops.summation(x, (1,)) / self.dim
        E_x = ops.broadcast_to(ops.reshape(E_x, (x.shape[0], 1)), x.shape)

        diff = x - E_x

        Var_x = ops.summation(ops.power_scalar(diff, 2), (1,)) / self.dim
        Var_x = ops.broadcast_to(ops.reshape(Var_x, (x.shape[0], 1)), x.shape)

        w = ops.broadcast_to(self.weight, x.shape)
        b = ops.broadcast_to(self.bias, x.shape)

        y = w * (diff / ops.power_scalar(Var_x + self.eps, 0.5)) + b

        return y
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x

        probs = init.randb(*x.shape, p = (1 - self.p))
        probs = probs / ((1.0 - self.p) + 1e-8)
        x = probs * x 

        return x
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        # raise NotImplementedError()
        ### END YOUR SOLUTION
