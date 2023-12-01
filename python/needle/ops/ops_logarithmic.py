from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(Z.shape)))

        elif type(self.axes) == int:
            self.axes = tuple([self.axes])

        max_z = Z.max(self.axes, keepdims = True)
        max_z_reduced = Z.max(self.axes)
        diff = Z - max_z
        exp_term = array_api.exp(diff)
        sum_term = array_api.sum(exp_term, self.axes)
        log_term = array_api.log(sum_term)
        result = log_term + max_z_reduced

        return result
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]

        out_grad_new_shape = [1] * len(a.shape)
        ctr = 0

        for ix in range(len(a.shape)):
            if ix not in self.axes:
                out_grad_new_shape[ix] = out_grad.shape[ctr]
                ctr += 1
        
        output = Tensor(node.realize_cached_data(), device = out_grad.device, dtype = out_grad.dtype)
        output = broadcast_to(reshape(output, out_grad_new_shape), a.shape)
        logsoftmax = a - output
        grad = broadcast_to(reshape(out_grad, out_grad_new_shape), a.shape) * exp(logsoftmax)

        return (grad, )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

