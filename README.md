# Deep Learning Systems Project
Public repository for the project of 10-714 (Deep Learning Systems @ CMU)
=======
# **Fast Fourier Transform based Convolution Neural Networks**
Hello, this is Rushikesh, Shaurya and Sheenu's project for Deep learning systems. In this project, we add autodiff support for 1D and 2D Fast Fourier Transform and Inverse Fast Fourier Transform. We test our implementation with custom test cases and further show the application of these operations in a CNN.

## Applications 
The applications of FFT are many-fold but a few of those are:

1) In image filtering, it is commonly used for convolution and correlation. It allows efficient computation of the convolution of an image with a filter kernel in the frequency domain, which can be faster than performing the operation directly in the spatial domain. This is the application that we will be focusing on in this project.

2) In image compression, transformation of the image from the spatial to the frequency domain using FFT allows for the removal of high-frequency components. This in turn reduces the amount of data needed to represent the image that too without major loss of signal/quality.

3) In audio processing, FFT is used for tasks such as equalization, in which the frequency response of an audio signal is tuned. Shazam in audio processing is a prime example of using FFT for a mainstream product.

# Benefits
Frequency Domain Representation: FFT transforms signals from the time domain to the frequency domain, providing CNNs with a more comprehensive view of the input data.

Reduced Complexity: By leveraging FFT, CNNs can reduce computational complexity, making them more efficient in processing large datasets and real-time applications.

Robustness to Noise: Frequency domain representations often enhance the model's ability to handle noisy data, contributing to improved robustness in various environments.

# Discrete Fourier Transforms**

The DFT takes as input a signal x[n] of N samples, and produces a sequence x[m] (m = 0,1,2, ...N-1) of N complex numbers representing amplitude and phase for each analysis frequency. It can be defined as:

$$
\begin{equation}
X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-j \frac{2\pi}{N}kn}, \quad k = 0, 1, \ldots, N-1
\end{equation}
$$
