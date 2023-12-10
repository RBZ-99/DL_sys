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


class Fin_base(Module):

    def __init__(self):
        super(Fin_base, self).__init__()

        self.conv1 = Conv(1, 16, kernel_size=5,stride=2,padding=4,device=ndl.cuda())
        self.bn1 = BatchNorm2d(16, device=ndl.cuda())

        self.relu = ReLU()
        self.conv2 = Conv(16,64, kernel_size=3,stride=1,device=ndl.cuda())#,padding=2)
        self.bn2 = BatchNorm2d(64, device=ndl.cuda())
        self.conv3 = Conv(64,96,kernel_size=3,stride=1,device=ndl.cuda())#,padding=2)
        self.bn3 = BatchNorm2d(96, device=ndl.cuda())
        self.conv4 = Conv(96,128,kernel_size=3,stride=1,device=ndl.cuda())#,padding=2)
        self.bn4 = BatchNorm2d(128, device=ndl.cuda())
        self.linear = Linear(32768,10,device=ndl.cuda()) #check the size to initialize this layer


    def forward(self, x):
        #pdb.set_trace()
        batch_size,b = x.shape
        x = ops.reshape(x,(batch_size,1,28,28))
        x = self.conv1(x)
        x = self.bn1(x)

        a,b,c,d = x.shape
        #x = ops.reshape(x,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        #x = ops.fft2d(x) #torch.fft.rfft2(x) #assuming it returns two separate channels
        #b_f,h_f,w_f,c_f = x.shape #256, 32, 32, 2
        #x = ops.reshape(x,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        #x = ops.reshape(x,(a,b*2,h_f,w_f))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        b_fl, c_fl, h_fl, w_fl = x.shape
        #x = ops.reshape(x,(int(b_fl*c_fl/2),h_fl,w_fl,2))
        #pdb.set_trace()
        #x = ops.ifft2d(x,only_real=True) #torch.fft.ifft2(x) #assuming it takes (b*c/2,h,w,2)
        a1,b1,c1,d1 = x.shape
        #x = ops.reshape(x,(batch_size,int(c_fl/2),b1,c1))
        #new_shape = int((c_fl/2)*c1*b1) #x.shape[0]*x.shape[1]*x.shape[2] #*x.shape[3]
        new_shape = x.shape[1]*x.shape[2]*x.shape[3]
        x = ops.reshape(x,(batch_size,new_shape)) #.flatten(x)
        #pdb.set_trace()

        x = self.linear(x)
        
        
        return x



class Fin_FFC(Module):

    def __init__(self):
        super(Fin_FFC, self).__init__()
#         self.ratio = ratio
        
#         in_cg = int(in_channels * ratio)
#         in_cl = channels - in_cg
#         r = 16

        # self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.conv1 = Conv(1, 16, kernel_size=5,stride=2,padding=4,device=ndl.cuda())
        self.bn1 = BatchNorm2d(16, device=ndl.cuda())

        self.relu = ReLU()
        self.conv2 = Conv(32,64, kernel_size=3,stride=1,device=ndl.cuda())#,padding=2)
        self.bn2 = BatchNorm2d(64, device=ndl.cuda())
        self.conv3 = Conv(64,96,kernel_size=3,stride=1,device=ndl.cuda())#,padding=2)
        self.bn3 = BatchNorm2d(96, device=ndl.cuda())
        
        self.conv4 = Conv(48,96,kernel_size=3,stride=1,device=ndl.cuda())#,padding=2)
        self.bn4 = BatchNorm2d(96, device=ndl.cuda())

        self.linear = Linear(24576,10,device=ndl.cuda()) #check the size to initialize this layer


    def forward(self, x):
#         batch, c, h , w = x.shape
#         channels_local = int(self.ratio_g * c)
#         channels_global = c - channels_local
        #pdb.set_trace()
        batch_size,b = x.shape
        x = ops.reshape(x,(batch_size,1,28,28))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        a,b,c,d = x.shape
        x = ops.reshape(x,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        
        x = ops.fft2d(x) #torch.fft.rfft2(x) #assuming it returns two separate channels
        b_f,h_f,w_f,c_f = x.shape #256, 32, 32, 2
        #x = ops.reshape(x,(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        x = ops.reshape(x,(a,b*2,h_f,w_f))
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        #x = self.avgpool(x)
    
        b_fl, c_fl, h_fl, w_fl = x.shape
        x = ops.reshape(x,(int(b_fl*c_fl/2),h_fl,w_fl,2))
        #pdb.set_trace()
        
        x = ops.ifft2d(x,only_real=True) #torch.fft.ifft2(x) #assuming it takes (b*c/2,h,w,2)
        b_4,h_4,w_4,c_4 = x.shape
        
        x = ops.reshape(x,(b_4,h_4,w_4))
        x = ops.reshape(x,(batch_size,int(b_4/batch_size),h_4,w_4))
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        a1,b1,c1,d1 = x.shape

        #x = ops.reshape(x,(batch_size,int(c_fl/2),b1,c1))
        #new_shape = int((c_fl/2)*c1*b1) #x.shape[0]*x.shape[1]*x.shape[2] #*x.shape[3]
        new_shape = int(b1*c1*d1)
        
        x = ops.reshape(x,(batch_size,new_shape)) #.flatten(x)

        x = self.linear(x)
#         x = self.linear2(x)
        
        
        return x

# Ffc_layer = Fin_FFC()
# arr1 = Tensor(np.random.rand(16,1,32,32))
# out = Ffc_layer(arr1)


