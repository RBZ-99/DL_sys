import sys
sys.path.append('./python')
import itertools
import numpy as np
import pytest
import mugrade
import torch

import needle as ndl
from needle import backend_ndarray as nd
import time


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

DIMS1D = [
        (1, 4),
        (1, 8),
        (1, 16),
        (1, 32),
        (4, 4),
        (4, 8),
        (4, 16),
        (8, 8),
        (8, 16),
        (16, 16),
        (32, 32),

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

DIMS2D = [
        (4, 4, 4),
        (4, 4, 8),
        (4, 4, 16),
        (4, 8, 4),
        (4, 8, 8),
        (8, 4, 4),
        (8, 4, 8),
        (8, 8, 4),
        (8, 8, 8)

]

@pytest.mark.parametrize("m,n", DIMS1D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft1d_forward(m, n, device):
    _A = np.random.randn(m, n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    own_res = ndl.fft1d(A).numpy()
    np_res = np.fft.fft(_A)

    np.testing.assert_allclose(np_res.real, own_res[:, :, 0], atol=1e-2)
    np.testing.assert_allclose(np_res.imag, own_res[:, :, 1], atol=1e-2)

@pytest.mark.parametrize("m,n,p", DIMS2D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft2d_forward(m, n, p, device):
    _A = np.random.randn(m, n, p).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    own_res = ndl.fft2d(A).numpy()
    np_res = np.fft.fft2(_A)
    
    np.testing.assert_allclose(np_res.real, own_res[:, :, :, 0], atol=1e-1)
    np.testing.assert_allclose(np_res.imag, own_res[:, :, :, 1], atol=1e-1)

@pytest.mark.parametrize("m,n", DIMS1D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ifft1d_forward(m, n, device):
    _A = np.random.randn(m, n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    own_res = ndl.ifft1d(A).numpy()
    np_res = np.fft.ifft(_A)

    np.testing.assert_allclose(np_res.real, own_res[:, :, 0], atol=1e-3)
    np.testing.assert_allclose(np_res.imag, own_res[:, :, 1], atol=1e-3)

    recon = ndl.ifft1d(ndl.fft1d(A)).numpy()
    np.testing.assert_allclose(A.numpy(), recon[:, :, 0], atol=1e-3)

@pytest.mark.parametrize("m,n,p", DIMS2D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ifft2d_forward(m, n, p, device):
    _A = np.random.randn(m, n, p).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    own_res = ndl.ifft2d(A).numpy()
    np_res = np.fft.ifft2(_A)

    np.testing.assert_allclose(np_res.real, own_res[:, :, :, 0], atol=1e-3)
    np.testing.assert_allclose(np_res.imag, own_res[:, :, :, 1], atol=1e-3)

    recon = ndl.ifft2d(ndl.fft2d(A)).numpy()
    np.testing.assert_allclose(A.numpy(), recon[:, :, :, 0], atol=1e-3)

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
    print(backward_grad)
    print(numerical_grad)
    error = sum(
        np.linalg.norm(backward_grad[i].numpy() - numerical_grad[i])
        for i in range(len(args))
    )
    assert error < 100
    return [g.numpy() for g in backward_grad]


@pytest.mark.parametrize("m,n", DIMS1D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft1d_backward(m, n, device):
    _A = np.random.randn(m, n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)

    start = time.time()
    backward_check(ndl.fft1d, A)
    end = time.time()
    print("Time taken =", end - start)

@pytest.mark.parametrize("m,n", DIMS1D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ifft1d_backward(m, n, device):
    _A = np.random.randn(m, n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    backward_check(ndl.ifft1d, A)

@pytest.mark.parametrize("m,n,p", DIMS2D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft2d_backward(m, n, p, device):
    _A = np.random.randn(m, n, p).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    backward_check(ndl.fft2d, A)

@pytest.mark.parametrize("m,n,p", DIMS2D)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ifft2d_backward(m, n, p, device):
    _A = np.random.randn(m, n, p).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    backward_check(ndl.ifft2d, A)
