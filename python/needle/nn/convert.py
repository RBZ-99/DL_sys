import sys
sys.path.append('/home/rushikesh/Projects/dlsys/DL_sys/python')
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
import pdb
# import opencv
import cv2
import needle as ndl
import matplotlib.pyplot as plt


'''
A test code to find the FFT of an image and also find its inverse FFT.
'''
img = cv2.imread("./sample_image_mnist.webp")
img = cv2.resize(img, (2048, 2048), interpolation = cv2.INTER_LINEAR)

inp = Tensor(img[:, :, 0])

inp = ndl.reshape(inp, (1, inp.shape[0], inp.shape[1]))
op = ndl.fft2d(inp)
fft_op = op.realize_cached_data()[0, :, :, 0].numpy().reshape((2048, 2048))
fft_op = np.log(abs(fft_op))
color = np.stack((fft_op,) * 3, axis = -1)


cv2.imwrite("fft.png", color)

inp2 = Tensor(op)
op2 = ndl.ifft2d(inp2)
fft_op = op2.realize_cached_data()[0, :, :, 0].numpy().reshape((2048, 2048))
color = np.stack((fft_op,) * 3, axis = -1)


cv2.imwrite("ifft.png", color)

# color = np.stack((img,) * 3, axis = -1)
cv2.imwrite("input.png", img)

np_fft = np.fft.fft2(img[:, :, 0])
# import pdb; pdb.set_trace()
color = np.stack((np_fft,) * 3, axis = -1)
cv2.imwrite("np_fft.png", color.real)

np_ifft = np.fft.ifft2(np_fft)
color = np.stack((np_ifft,) * 3, axis = -1)
cv2.imwrite("np_ifft.png", color.real)
