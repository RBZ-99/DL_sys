import numpy as np
from numpy.core import as_array

def FFT_forward(x):
    N = len(x)
    if N == 1:
        return x
    else:
        X_even = FFT_forward(x[::2])
        X_odd = FFT_forward(x[1::2])
        factor = np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate([X_even+factor[:int(N/2)]*X_odd,X_even+factor[int(N/2):]*X_odd])
        return X
    

def IFFT_forward(X):
    N = len(X)
    if N == 1:
        return X
    else:
        X_even = IFFT_forward(X[::2])
        X_odd = IFFT_forward(X[1::2])
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        x = np.concatenate([X_even + factor[:int(N/2)] * X_odd,X_even - factor[int(N/2):] * X_odd])
        return x / 2 


def FFT2D_forward(X):

    x = as_array(X)

    ans = []

    x = np.array([FFT_forward(row) for row in x], dtype=np.complex128)
    
    x = np.array([FFT_forward(col) for col in x.T], dtype=np.complex128).T

    return x

def IFFT2D_forward(X):

    x = as_array(X)

    ans = []

    x = np.array([IFFT_forward(row) for row in x], dtype=np.complex128)
    
    x = np.array([IFFT_forward(col) for col in x.T], dtype=np.complex128).T

    return x





def IFFT2D_backward(X):
    return X



x = np.array([1, 2, 3, 4])
X = IFFT(x)

print("Original Signal:", x)
print("IFFT Result:", X)