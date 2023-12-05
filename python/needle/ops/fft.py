"""FFT Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

import numpy as np
import pdb
from ..init import *
PI = 3.142


class Concat(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along an existing dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size in the non-concatenated dimensions.
        """
        self.axis = axis
        self.lens = []

    def compute(self, args: TensorTuple) -> Tensor:
        out, self.lens = array_api.concat(args, self.axis)

        return out
      
    def gradient(self, out_grad, node):
        return (split_concat(out_grad, self.axis, self.lens),)


def concat(args, axis):
    return Concat(axis)(make_tuple(*args))


class SplitConcat(TensorTupleOp):
    def __init__(self, axis: int, lens: list):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Concat)
        Parameters:
        axis - dimension to split
        lens - sizes of each of the constituent arrays along the concatenated dimension
        """
        self.axis = axis
        self.lens = lens

    def compute(self, a):
        out = []
        slices = [slice(0, dim) for dim in a.shape]
        
        start = 0
        for ix in range(len(self.lens)):
            end = start + self.lens[ix]
            slices[self.axis] = slice(start, end)
            out.append(a[tuple(slices)].compact())
            start += self.lens[ix]
            
        out = tuple(out)
        
        return out

    def gradient(self, out_grad, node):
        return (concat(out_grad, self.axis),)


def split_concat(arg, axis, lens):
    return SplitConcat(axis, lens)(arg)


class FFT1D(TensorOp):
    def compute(self, a):
        return a.fft1d()

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        N = a.shape[0]

        base = NDArray(range(N), device = out_grad.device)
        jacob_init = base.reshape((1, N)).broadcast_to((N, N))
        mul = base.reshape((N, 1)).broadcast_to((N, N))
        jacob_init = jacob_init * mul
        
        jacob = array_api.full((N, N, 2), 0, out_grad.dtype, out_grad.device)
        jacob[:, :, 1] = NDArray(-2 * PI * jacob_init / N, device = out_grad.device)
        jacob = jacob.reshape((N * N, 2)).complex_exp().reshape((N, N, 2))
        jacob[:, :, 1] *= -1

        out_grad_arr = out_grad.realize_cached_data()
        
        rr = jacob[:, :, 0].reshape((N, N)) @ out_grad_arr[:, 0].reshape((N, 1))
        cc = jacob[:, :, 1].reshape((N, N)) @ out_grad_arr[:, 1].reshape((N, 1))
        grad = rr - cc 
        
        return (grad,)


def fft1d(a):
    return FFT1D()(a)


class IFFT1D(TensorOp):
    def __init__(self, only_real = False):
        self.only_real = only_real

    def compute(self, a):
        if self.only_real:
            a[:, 1] *= -1
            return a.ifft1d()[:, 0]

        return a.ifft1d()

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        N = a.shape[0]

        base = NDArray(range(N), device = out_grad.device)
        jacob_init = base.reshape((1, N)).broadcast_to((N, N))
        mul = base.reshape((N, 1)).broadcast_to((N, N))
        jacob_init = jacob_init * mul
        
        jacob = array_api.full((N, N, 2), 0, out_grad.dtype, out_grad.device)
        jacob[:, :, 1] = NDArray(2 * PI * jacob_init / N, device = out_grad.device)
        jacob = jacob.reshape((N * N, 2)).complex_exp().reshape((N, N, 2)) / N
        jacob[:, :, 1] *= -1

        out_grad_arr = out_grad.realize_cached_data()
        
        rr = jacob[:, :, 0].reshape((N, N)) @ out_grad_arr[:, 0].reshape((N, 1))
        cc = jacob[:, :, 1].reshape((N, N)) @ out_grad_arr[:, 1].reshape((N, 1))
        grad = rr - cc 
        
        return (grad,)


def ifft1d(a, only_real = False):
    return IFFT1D(only_real)(a)


class FFT2D(TensorOp):
    def compute(self, a):
        return a.fft2d()

    def gradient(self, out_grad, node):
        pass


def fft2d(a):
    return FFT2D()(a)


class IFFT2D(TensorOp):
    def compute(self, a):
      return a.ifft2d()

    def gradient(self, out_grad, node):
        pass


def ifft2d(a):
    return IFFT2D()(a)
