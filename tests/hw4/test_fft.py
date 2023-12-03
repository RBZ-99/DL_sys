import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import mugrade
import torch

import needle as ndl
from needle import backend_ndarray as nd


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

DIMS = [
        (1, 4),
        (1, 8),
        (1, 16),
        (1, 32),
        (4, 4),
        (4, 8),
        (4, 16),
        (8, 8),
        (8, 16),
        (16, 16)
        # (1, 10),
        # (10, 10),
        # (100, 100),
        # (4, 10),
        # (4, 100),
        # (10, 100),
        # (10, 4),
        # (100, 4),
        # (100, 10)
        ]

@pytest.mark.parametrize("m,n", DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft1d_forward(m, n, device):
    _A = np.random.randn(n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    own_res = ndl.fft1d(A).numpy()
    np_res = np.fft.fft(_A)

    np.testing.assert_allclose(np_res.real, own_res[:, 0], atol=1e-2)
    np.testing.assert_allclose(np_res.imag, own_res[:, 1], atol=1e-2)

@pytest.mark.parametrize("m,n", DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft2d_forward(m, n, device):
    _A = np.random.randn(m, n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    own_res = ndl.fft2d(A).numpy()
    np_res = np.fft.fft2(_A)
    
    np.testing.assert_allclose(np_res.real, own_res[:, :, 0], atol=1e-1)
    np.testing.assert_allclose(np_res.imag, own_res[:, :, 1], atol=1e-1)

@pytest.mark.parametrize("m,n", DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ifft1d_forward(m, n, device):
    _A = np.random.randn(m, n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.fft.ifft(np.fft.fft(_A)), ndl.ifft1d(ndl.fft1d(A)).numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(A.numpy(), ndl.ifft1d(ndl.fft1d(A)).numpy(), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("m,n", DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ifft1d_forward(m, n, device):
    _A = np.random.randn(m, n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.fft.ifft2(np.fft.fft2(_A)), ndl.ifft2d(ndl.fft2d(A)).numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(A.numpy(), ndl.iff21d(ndl.fft2d(A)).numpy(), atol=1e-5, rtol=1e-5)


def backward_check(f, *args, **kwargs):
    eps = 1e-5
    out = f(*args, **kwargs)
    c = np.random.randn(*out.shape)
    numerical_grad = [np.zeros(a.shape) for a in args]
    num_args = len(args)
    for i in range(num_args):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = (f(*args, **kwargs).numpy() * c).sum()
            args[i].realize_cached_data().flat[j] += eps
            numerical_grad[i].flat[j] = (f1 - f2) / (2 * eps)
    backward_grad = out.op.gradient_as_tuple(ndl.Tensor(c, device=args[0].device), out)
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 4.2e-1
    return [g.numpy() for g in backward_grad]


@pytest.mark.parametrize("m,n", DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft1d_backward(m, n, device):
    _A = np.random.randn(m, n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    backward_check(ndl.fft1d, A)

@pytest.mark.parametrize("m,n", DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft2d_backward(m, n, device):
    _A = np.random.randn(m, n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    backward_check(ndl.fft2d, A)
