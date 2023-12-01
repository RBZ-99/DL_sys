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
PI = 3.142


class FFT1D(TensorOp):

    def compute(self, x):
        N = x.shape[0]
        import pdb

        if N == 1:
            pdb.set_trace()
            return x

        else:
            X_even = fft1d(x[::2])
            X_odd = fft1d(x[1::2])
            factor = exp(-2j * PI * NDArray(range(N)) / N)
            X = stack((X_even + factor[:int(N / 2)] * X_odd, X_even + factor[int(N / 2):] * X_odd), 0)
            pdb.set_trace()
            return X
    
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
