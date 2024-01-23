# Deep Learning Systems Project
Public repository for the project of 10-714 (Deep Learning Systems @ CMU)
=======
# Needle Framework 
Needle is a deep learning framework created to implement fundamental and even complicated classes and functions for the implementation of a complete deep learning pipeline. 

Needle can also be known as analogous to a basic implementation of PyTorch framework. A few crucial functions which have been implemented in Needle are -

1) Automatic Differentiation 
2) nn.Module Library Implementation
3) Image Convolution Operations
4) Sequence Modeling Functions
5) Transformer Implementation

# Fast Fourier Transform based Convolution Neural Networks
Hello, this is Rushikesh, Shaurya and Sheenu's project for Deep learning systems. In this project, we add autodiff support for 1D and 2D Fast Fourier Transform and Inverse Fast Fourier Transform. We test our implementation with custom test cases and further show the application of these operations in a CNN.

## Applications 
The applications of FFT are many-fold but a few of those are:

1) In image filtering, it is commonly used for convolution and correlation. It allows efficient computation of the convolution of an image with a filter kernel in the frequency domain, which can be faster than performing the operation directly in the spatial domain. This is the application that we will be focusing on in this project.

2) In image compression, transformation of the image from the spatial to the frequency domain using FFT allows for the removal of high-frequency components. This in turn reduces the amount of data needed to represent the image that too without major loss of signal/quality.

3) In audio processing, FFT is used for tasks such as equalization, in which the frequency response of an audio signal is tuned. Shazam in audio processing is a prime example of using FFT for a mainstream product.

## Benefits
Frequency Domain Representation: FFT transforms signals from the time domain to the frequency domain, providing CNNs with a more comprehensive view of the input data.

Reduced Complexity: By leveraging FFT, CNNs can reduce computational complexity, making them more efficient in processing large datasets and real-time applications.

Robustness to Noise: Frequency domain representations often enhance the model's ability to handle noisy data, contributing to improved robustness in various environments.

## Discrete Fourier Transforms

The DFT takes as input a signal x[n] of N samples, and produces a sequence x[m] (m = 0,1,2, ...N-1) of N complex numbers representing amplitude and phase for each analysis frequency. It can be defined as:

$$
\begin{equation}
X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-j \frac{2\pi}{N}kn}, \quad k = 0, 1, \ldots, N-1
\end{equation}
$$

# Fast Fourier Transform (FFT)

The Fourier domain representation of any real signal satisfies the Hermitian property: X[i, j] = conj(X[-i, -j]). This function always returns all positive and negative frequency terms even though, for real inputs, half of these values are redundant.

## Advantages of FFT

The DFT calculation for N samples requires approximately N·N complex calculation operations and is a time intensive and a memory intensive calculation. If the length of the sequence is a power of 2,

$$
\begin{equation}
N = 2^m, \hspace{0.5cm} m = 0,1,2 ...
\end{equation}
$$

the DFT would take about $\theta$($N^2$) operations whereas it can be calculated with approximately $\theta$(N·log2(N)) operations using fast Fourier transform (FFT).

The advantages of the FFT include speed and memory efficiency. The DFT can process sequences of any size efficiently but is slower than the FFT and requires more memory, because it saves intermediate results while processing.

## Forward 1DFFT
We implemented both the recursive and the iterative methods for the Cooley–Tukey FFT 1D algorithm. We found that for our given case, the recursive method performs faster than the iterative method, hence we proceeded with the recursive approach for our further calculations.

## Backward 1DFFT

## Forward 2DFFT

FFT2D is computed using FFT1D such that for a given 2 dimensional matrix, with r rows and c columns, first the FFT1D is applied to each of the rows. The resulting output is then traversed column wise and again FFT1D is applied to each of those columns.

# Helper Functions

## Trignometric Functions and Complex Arithmetic Implementations

We have implemented the sin and cos functions to further implement the complex variables. Apart from this we have implemented Concatenate and Split as Tensor Operations in Python.

For the low level NDArray implementations, the functionality for using Complex Exponential and Complex Multiplication has been implemented.

## Numerical Gradient Checker Functions

We have used two ways to test our implementations.

One of the way is by visualizing the implementation of our 2DFFT forward function on the image and compare the output with numpy's implementation of 2DFFT forward.

The second way of testing used is by implementing the numerical gradient checker to test our backward implementations of 1D and 2D FFT functions.

# AIM

So our aim now is to implement the FFT2D forward function in a CNN pipeline. And after doing that we train that model, and using the backward 2DFFT functions, the loss should propagate in an appropriate manner and thus converge the loss of the model accordingly. To do so we built a CNN model slightly modified with FFT2D. The network and the further training details are mentioned in the sections below.

# Final CNN Network

Our model consists of Convolutional layers, BatchNormalization, ReLU for non-linearity, a linear layer along with the FFT2D and Inverse FFT2D (IFFT2D). The architecture diagram is shown below:

# Experiments 

We ran our method on the dataset by rescaling the images to have dimensions equal to the nearest exponent of 2, so that FFT can be used on them easily. The experiment follow the common pipeline of a simple CNN based classifier, only difference being the use of a FFT2D and IFFT2D layer.

### Dataset

We used MNIST dataset to train and test our implementation of ConvFFT based model and our implementation of simple CNN based model and to see their performance. MNIST is a  database of handwritten digits that is commonly used for training various image processing systems. MNIST database contains 60,000 training images and 10,000 testing images, all of which are grayscale and of 28x28 pixels

### Results

Our method performs as well as the baseline method on MNIST Dataset, where both of the, achieve 99% accuracy on the test set.

# References

1) https://dsp.stackexchange.com/questions/70113/dft-derivative-property

2)Lu Chi, Borui Jiang, Yadong Mu Fast Fourier Convolution. In NIPS, 2020.

3)https://pytorch.org/docs/stable/notes/autograd.html


