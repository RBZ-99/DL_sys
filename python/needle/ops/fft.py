"""FFT Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

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
        pass


def fft1d(a):
    return FFT1D()(a)


class IFFT1D(TensorOp):
    def compute(self, x):
        N = x.shape[0]

        if N == 1:
            return x

        else:
            X_even = ifft1d(x[::2])
            X_odd = ifft1d(x[1::2])
            factor = exp(2j * PI * NDArray(range(N)) / N)
            x = stack((X_even + factor[:int(N / 2)] * X_odd, X_even - factor[int(N / 2):] * X_odd), 0)

            return x / 2 

    def gradient(self, out_grad, node):
        pass


def ifft1d(a):
    return IFFT1D()(a)


class FFT2D(TensorOp):
    def compute(self, a):
        return a.fft2d()

    def gradient(self, out_grad, node):
        pass


def fft2d(a):
    return FFT2D()(a)


class IFFT2D(TensorOp):
    def compute(self, a):
        x = np.array([IFFT_forward(row) for row in x], dtype=np.complex128)
        x = np.array([IFFT_forward(col) for col in x.T], dtype=np.complex128).T

        return x

    def gradient(self, out_grad, node):
        pass


def ifft2d(a):
    return IFFT2D()(a)
