import sys

sys.path.append("./python")
sys.path.append("./apps")
from simple_ml import *
import numdifftools as nd

import numpy as np
import mugrade
import needle as ndl


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]

DIMS = [(1, 4),
        (1, 10),
        (1, 100),
        (4, 4),
        (10, 10),
        (100, 100),
        (4, 10),
        (4, 100),
        (10, 100),
        (10, 4),
        (100, 4),
        (100, 10)]

@pytest.mark.parametrize("m,n", DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft1d_forward(m, n, device):
    _A = np.random.randn(m, n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.fft.fft(_A), ndl.fft1d(A), atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("m,n", DIMS)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_fft2d_forward(m, n, device):
    _A = np.random.randn(m, n).astype(np.float32)
    A = ndl.Tensor(nd.array(_A), device=device)
    np.testing.assert_allclose(np.fft.fft2(_A), ndl.fft2d(A), atol=1e-5, rtol=1e-5)


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
