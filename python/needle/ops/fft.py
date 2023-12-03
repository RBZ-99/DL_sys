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
        assert len(args) > 0, "Concat needs at least one array"
        
        shape = args[0].shape
        ref_shape = list(shape)
        del ref_shape[self.axis]
        new_dim = 0

        for arg in args:
          arg_shape = list(arg.shape)
          new_dim += arg_shape[self.axis]
          del arg_shape[self.axis]
          assert ref_shape == arg_shape, "All arrays need to be of same size in all the non-concatenating axes"
          
        out_shape = list(shape)
        out_shape[self.axis] = new_dim
        out_shape = tuple(out_shape)

        out = array_api.full(out_shape, 0, device = args[0].device)
        slices = [slice(0, dim) for dim in out.shape]
        
        start = 0
        for i, arg in enumerate(args):
            end = start + arg.shape[self.axis]
            slices[self.axis] = slice(start, end)
            out[tuple(slices)] = arg
            start += arg.shape[self.axis]
            self.lens.append(arg.shape[self.axis])

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

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        out = []
        slices = [slice(0, dim) for dim in A.shape]
        
        start = 0
        for ix in range(len(self.lens)):
            end = start + self.lens[ix]
            slices[self.axis] = slice(start, end)
            out.append(A[tuple(slices)].compact())
            start += self.lens[ix]
            
        out = tuple(out)
        
        return out
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (concat(out_grad, self.axis), )
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def split_concat(arg, axis, lens):
    return SplitConcat(axis, lens)(arg)


class FFT1D(TensorOp):

    def compute(self, x):
        # N = x.shape[0]

        # if N == 1:
        #     arr = array_api.full((N, 2), 0, x.dtype, x.device)
        #     arr[:, 0] = x
        #     # pdb.set_trace()
        #     return arr
        #     # return Tensor(arr, device = x.device, dtype = x.dtype)

        # else:
        #     X_even = fft1d(Tensor(x[::2], device = x.device))
        #     X_odd = fft1d(Tensor(x[1::2], device = x.device))
            
        #     factor = array_api.full((N, 2), 0, x.dtype, x.device)
        #     factor[:, 1] = NDArray(range(N))
        #     factor1 = Tensor(factor[:N // 2, :], device = x.device, dtype = x.dtype)
        #     factor2 = Tensor(factor[N // 2:, :], device = x.device, dtype = x.dtype)
        #     # pdb.set_trace()
        #     # factor1 = factor[:N // 2, :]
        #     # factor2 = factor[N // 2:, :]
        #     factor1 = exp(-2 * PI * factor1 / N)
        #     factor2 = exp(-2 * PI * factor2 / N)
            
        #     # pdb.set_trace()
        #     # temp = factor1 * X_odd
        #     print("BEFORE")
        #     print(factor1.shape, factor2.shape, type(X_even), type(X_odd))
        #     print(X_even.shape, X_odd.shape)
        #     X = concat((X_even + factor1 * X_odd, X_even + factor2 * X_odd), 0)
        #     print("CHECK", factor1.shape, factor2.shape)
        #     print(type(X), X.shape)
        #     # X = stack((X_even + factor1 * X_odd, X_even + factor2 * X_odd), 0)
            
        #     return X

        return x.fft1d()

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
    def compute(self, x):
        x = NDArray([fft1d(row) for row in x], dtype=np.complex128)
        x = NDArray([fft1d(col) for col in transpose(x)], dtype=np.complex128)
        x = transpose(x)

        return x

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
