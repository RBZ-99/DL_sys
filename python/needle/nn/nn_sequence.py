"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** -1
        # raise NotImplementedError()
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        bound = (1 / hidden_size) ** 0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low = -bound, high = bound,
        device = device, dtype = dtype, requires_grad = True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low = -bound, high = bound,
        device = device, dtype = dtype, requires_grad = True))
        
        self.bias = bias
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low = -bound, high = bound,
        device = device, dtype = dtype, requires_grad = True))
            self.bias_hh = Parameter(init.rand(hidden_size, low = -bound, high = bound,
        device = device, dtype = dtype, requires_grad = True))

        self.non_linearity = nonlinearity
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        out = X @ self.W_ih
        
        if h is not None:
            out = out + h @ self.W_hh

        if self.bias:
            bias_ih = ops.broadcast_to(ops.reshape(self.bias_ih, (1, self.hidden_size)), (X.shape[0], self.hidden_size))
            bias_hh = ops.broadcast_to(ops.reshape(self.bias_hh, (1, self.hidden_size)), (X.shape[0], self.hidden_size))
            out += bias_ih + bias_hh
            
        if self.non_linearity == "tanh":
            out = ops.tanh(out)

        elif self.non_linearity == "relu":
            out = ops.relu(out)

        return out
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        
        self.rnn_cells = []
        for i in range(num_layers):
            if not i:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, 
                bias, nonlinearity, device, dtype))        
            
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, 
                bias, nonlinearity, device, dtype))
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0 is None:
            h0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, 
            device = self.device, dtype = self.dtype)
        
        out = []
        h_n = []
        inputs = list(ops.split(X, 0))
        h0 = list(ops.split(h0, 0))

        for layer_idx in range(self.num_layers):
            prev_h = h0[layer_idx]
            out = []

            for input_idx, inp in enumerate(inputs):
                prev_h = self.rnn_cells[layer_idx](inp, prev_h)
                inputs[input_idx] = prev_h
                out.append(prev_h)

            h_n.append(prev_h)
        
        return ops.stack(out, 0), ops.stack(h_n, 0)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        bound = (1 / hidden_size) ** 0.5
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low = -bound, high = bound,
        device = device, dtype = dtype, requires_grad = True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low = -bound, high = bound,
        device = device, dtype = dtype, requires_grad = True))
        
        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low = -bound, high = bound,
        device = device, dtype = dtype, requires_grad = True))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low = -bound, high = bound,
        device = device, dtype = dtype, requires_grad = True))

        # raise NotImplementedError()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        gate_outs = X @ self.W_ih
        
        if h is not None:
            gate_outs = gate_outs + h[0] @ self.W_hh

        if self.bias:
            bias_ih = ops.broadcast_to(ops.reshape(self.bias_ih, (1, 4 * self.hidden_size)), (X.shape[0], 4 * self.hidden_size))
            bias_hh = ops.broadcast_to(ops.reshape(self.bias_hh, (1, 4 * self.hidden_size)), (X.shape[0], 4 * self.hidden_size))
            gate_outs += bias_ih + bias_hh
            
        gate_outs = tuple(ops.split(gate_outs, 1))
        gates = []
        for i in range(4):
            gate_out = ops.stack(gate_outs[i * self.hidden_size : (i + 1) * self.hidden_size], 1)
            
            if i == 2:
                gate_out = ops.tanh(gate_out)
            
            else:
                gate_out = Sigmoid()(gate_out)

            gates.append(gate_out)

        out_c = gates[0] * gates[2]
        if h is not None:
            out_c += h[1] * gates[1]

        out_h = gates[3] * ops.tanh(out_c)

        return (out_h, out_c)
        # raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        
        self.lstm_cells = []
        for i in range(num_layers):
            if not i:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size, 
                bias, device, dtype))        
            
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, 
                bias, device, dtype))
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        h0, c0 = None, None

        if h is None:
            h0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, 
            device = self.device, dtype = self.dtype)
            c0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, 
            device = self.device, dtype = self.dtype)
            h = (h0, c0)
            
        else:
            h0, c0 = h

        out = []
        h_n = []
        c_n = []
        inputs = list(ops.split(X, 0))
        h0 = list(ops.split(h0, 0))
        c0 = list(ops.split(c0, 0))

        for layer_idx in range(self.num_layers):
            prev_h = h0[layer_idx]
            prev_c = c0[layer_idx]
            out = []

            for input_idx, inp in enumerate(inputs):
                prev_h, prev_c = self.lstm_cells[layer_idx](inp, (prev_h, prev_c))
                inputs[input_idx] = prev_h
                out.append(prev_h)

            h_n.append(prev_h)
            c_n.append(prev_c)
        
        return ops.stack(out, 0), (ops.stack(h_n, 0), ops.stack(c_n, 0))
        # raise NotImplementedError()
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(init.randn(num_embeddings, embedding_dim,
        device = device, dtype = dtype, requires_grad = True))
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        one_hot_vecs = init.one_hot(self.num_embeddings, x,
        device = self.device, dtype = self.dtype, requires_grad = True)
        out = one_hot_vecs.reshape((seq_len * bs, self.num_embeddings)) @ self.weight
        out = out.reshape((seq_len, bs, self.embedding_dim))

        return out
        # raise NotImplementedError()
        ### END YOUR SOLUTION