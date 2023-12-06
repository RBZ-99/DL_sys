import sys
sys.path.append('/Users/rushikeshzawar/Downloads/Personal/CMU_COURSES/dlsys/project/repo/DL_sys/python')
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
import pdb

#from fft import *
import needle as ndl
from needle.nn.nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential,
    Residual,
    BatchNorm2d
)
from needle.nn.nn_conv import Conv


class Fin_FFC(Module):

    def __init__(self):
        super(Fin_FFC, self).__init__()
#         self.ratio = ratio
        
#         in_cg = int(in_channels * ratio)
#         in_cl = channels - in_cg
#         r = 16

        # self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.conv1 = Conv(1, 16, kernel_size=3,stride=1) #,padding=0)
        self.relu = ReLU()
        self.conv2 = Conv(32,64, kernel_size=3,stride=1) #,padding=0)
        self.conv3 = Conv(64,96,kernel_size=3,stride=1)#,padding=0)
        self.conv4 = Conv(96,128,kernel_size=3,stride=1)#,padding=0)
        self.conv5 = Conv(128,256,kernel_size=3,stride=1)#,padding=0)
        self.linear = Linear(786432,10) #check the size to initialize this layer
#         self.conv_a2l = None if in_cl == 0 else nn.Conv2d(channels // r, in_cl, kernel_size=1, bias=True)
#         self.conv_a2g = None if in_cg == 0 else nn.Conv2d(channels // r, in_cg, kernel_size=1, bias=True)
        #self.sigmoid = Sigmoid()

    def forward(self, x):
#         batch, c, h , w = x.shape
#         channels_local = int(self.ratio_g * c)
#         channels_global = c - channels_local
        pdb.set_trace()
        
        x = self.conv1(x)
        x = self.relu(x)
        a,b,c,d = x.shape
        x = ops.reshape(x,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        
        x = ops.fft2d(x) #torch.fft.rfft2(x) #assuming it returns two separate channels
        
        
        b_f,h_f,w_f,c_f = x.shape #256, 32, 32, 2
        #x = 

        #x = ops.reshape(x,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        x = ops.reshape(x,(a,b*2,h_f,w_f))
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        #x = self.avgpool(x)
    
        
        b_fl, c_fl, h_fl, w_fl = x.shape
        x = ops.reshape(x,(int(b_fl*c_fl/2),h_fl,w_fl,2))
        
        x = ops.ifft2d(x,only_real=True) #torch.fft.ifft2(x) #assuming it takes (b*c/2,h,w,2)

        new_shape = x.shape[0]*x.shape[1]*x.shape[2] #*x.shape[3]

        
        x = ops.reshape(x,(new_shape,1)) #.flatten(x)

        x = self.linear(ops.transpose(x))
#         x = self.linear2(x)
        
        
        return x

Ffc_layer = Fin_FFC()
arr1 = Tensor(np.random.rand(16,1,32,32))
out = Ffc_layer(arr1)


