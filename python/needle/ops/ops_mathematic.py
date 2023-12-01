"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = out_grad * self.scalar * (a ** (self.scalar - 1))
        return (grad,)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * (b ** -1)
        grad_b = -out_grad * a * (b ** -2)
        return (grad_a, grad_b)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = out_grad / self.scalar
        return (grad,)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_axes = list(range(len(a.shape)))
        if self.axes is None:
            self.axes = (len(a.shape) - 1, len(a.shape) - 2)

        new_axes[self.axes[0]] = self.axes[1]
        new_axes[self.axes[1]] = self.axes[0]

        # return array_api.swapaxes(a, self.axes[0], self.axes[1])
        return a.permute(tuple(new_axes))
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = transpose(out_grad, self.axes)
        return (grad,)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = reshape(out_grad, a.shape)
        return(grad,)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        num_new_axes = len(self.shape) - len(a.shape)
        a_new_shape = [1] * num_new_axes + list(a.shape)
        b = reshape(a, tuple(a_new_shape))
        
        sum_axes = []

        for ix in range(len(self.shape)):
            if b.shape[ix] != self.shape[ix]:
                sum_axes.append(ix)

        grad = summation(out_grad, tuple(sum_axes))
        if grad.shape != a.shape:
            grad = reshape(grad, a.shape)

        return (grad,)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(a.shape)))

        elif type(self.axes) == int:
            self.axes = tuple([self.axes])

        elif not len(self.axes):
            return a
            
        return array_api.sum(a, self.axes)
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
        
        out_grad_reshaped = reshape(out_grad, out_grad_new_shape)
        grad = broadcast_to(out_grad_reshaped, a.shape)
        
        return (grad,)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad @ transpose(b)
        grad_b = transpose(a) @ out_grad

        if len(a.shape) < len(grad_a.shape):
            sum_axes = range(len(grad_a.shape) - len(a.shape))
            grad_a = summation(grad_a, tuple(sum_axes))

        if len(b.shape) < len(grad_b.shape):
            sum_axes = range(len(grad_b.shape) - len(b.shape))
            grad_b = summation(grad_b, tuple(sum_axes))

        return (grad_a, grad_b)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = -out_grad
        return (grad,)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = out_grad * (a ** -1)
        return (grad,)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = out_grad * exp(a)
        return (grad,)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        bin_array = node.realize_cached_data() > 0
        bin_tensor = Tensor(bin_array, device = out_grad.device, dtype = out_grad.dtype)

        grad = out_grad * bin_tensor
        return (grad,)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad = -(tanh(a) ** 2) + 1
        grad *= out_grad
        return (grad,)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(args) > 0, "Stack needs at least one array"
        
        shape = args[0].shape
        for arg in args:
            assert shape == arg.shape, "All arrays need to be of same size"

        out_shape = list(shape)
        out_shape.insert(self.axis, len(args))
        out_shape = tuple(out_shape)

        out = array_api.full(out_shape, 0, device = args[0].device)
        slices = [slice(0, dim) for dim in out.shape]
        
        for i, arg in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            out[tuple(slices)] = arg

        return out
        # return Tensor(out)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (split(out_grad, self.axis),)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        out = []
        slices = [slice(0, dim) for dim in A.shape]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        new_shape = tuple(new_shape)
        
        for ix in range(A.shape[self.axis]):
            slices[self.axis] = slice(ix, ix + 1)
            out.append(A[tuple(slices)].compact().reshape(new_shape))
            
        out = tuple(out)
        
        return out
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (stack(out_grad, self.axis), )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(a.shape)))

        elif type(self.axes) == int:
            self.axes = tuple([self.axes])

        return a.flip(self.axes)
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (flip(out_grad, self.axes), )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        out_shape = list(a.shape)
        for axis in self.axes:
            if axis >= len(a.shape):
              continue
              
            out_shape[axis] *= (self.dilation + 1)
            
        out_shape = tuple(out_shape)
        out = array_api.full(out_shape, 0, device = a.device)
        
        slices = [slice(0, dim) if axis not in self.axes else slice(0, dim, self.dilation + 1) for axis, dim in enumerate(out.shape)]        
        out[tuple(slices)] = a

        return out
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (undilate(out_grad, self.axes, self.dilation), )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = [slice(0, dim) if axis not in self.axes else slice(0, dim, self.dilation + 1) for axis, dim in enumerate(a.shape)]        
        out = a[tuple(slices)]

        return out 
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (dilate(out_grad, self.axes, self.dilation), )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        pads = [(0, 0) for i in range(len(A.shape))]
        pads[1] = pads[2] = (self.padding, self.padding)
        A = A.pad(tuple(pads))

        N, H, W, C_in = A.shape
        K1, K2, C_in, C_out = B.shape

        new_shape = (N, (H - K1) // self.stride + 1, (W - K2) // self.stride + 1, K1, K2, C_in)
        new_strides = (A.strides[0], self.stride * A.strides[1], self.stride * A.strides[2], A.strides[1], A.strides[2], A.strides[-1])
        im2col = A.as_strided(new_shape, new_strides).compact()
        im2col = im2col.reshape((N * new_shape[1] * new_shape[2], K1 * K2 * C_in))
        out = im2col @ B.compact().reshape((K1 * K2 * C_in, C_out))
        out = out.compact().reshape((N, new_shape[1], new_shape[2], C_out))

        return out
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs[0], node.inputs[1]
        if self.stride > 1:
          out_grad = dilate(out_grad, (1, 2), self.stride - 1)
  
        grad_A = conv(out_grad, transpose(flip(B, (0, 1)), (2, 3)), 1, B.shape[0] - self.padding - 1)
        
        grad_B = conv(transpose(A, (0, 3)), transpose(transpose(out_grad, (0, 2)), (0, 1)), 1, self.padding)
        grad_B = transpose(transpose(grad_B, (0, 2)), (0, 1))

        return (grad_A, grad_B)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
