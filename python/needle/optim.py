"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for ix, param in enumerate(self.params):
            gt_1 = param.grad.data + self.weight_decay * param.data

            ut = ndl.init.zeros(*gt_1.shape, device = gt_1.device, dtype = gt_1.dtype)
            if ix in self.u:
                ut = self.u[ix]

            ut_1 = self.momentum * ut + (1 - self.momentum) * gt_1

            param.data = param.data - self.lr * ut_1

            self.u[ix] = ut_1
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1

        for ix, param in enumerate(self.params):
            gt_1 = param.grad.data + self.weight_decay * param.data

            mt = ndl.init.zeros(*gt_1.shape, device = gt_1.device, dtype = gt_1.dtype)
            if ix in self.m:
                mt = self.m[ix]

            vt = ndl.init.zeros(*gt_1.shape, device = gt_1.device, dtype = gt_1.dtype)
            if ix in self.v:
                vt = self.v[ix]

            mt_1 = self.beta1 * mt + (1 - self.beta1) * gt_1
            mt_1_corr = mt_1 / (1 - (self.beta1 ** self.t) + self.eps)
            vt_1 = self.beta2 * vt + (1 - self.beta2) * (gt_1 ** 2)
            vt_1_corr = vt_1 / (1 - (self.beta2 ** self.t) + self.eps)

            param.data = param.data - self.lr * mt_1_corr / ((vt_1_corr ** 0.5) + self.eps)

            self.m[ix] = mt_1
            self.v[ix] = vt_1
            
        # raise NotImplementedError()
        ### END YOUR SOLUTION
