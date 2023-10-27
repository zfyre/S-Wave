import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import logging
handler=logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger(__name__)


"""
That is correct. The forward method uses the same filter coefficients h0 and h1 for all levels of decomposition. This means that the DWT is **orthogonal**, meaning that the subbands are uncorrelated and have equal energy. Orthogonal DWTs have some advantages, such as perfect reconstruction, sparsity, and computational efficiency. However, they also have some limitations, such as lack of shift-invariance, aliasing, and poor directional selectivity.

If you want to use different filters for different decomposition levels, you may need to use a **biorthogonal** or **non-orthogonal** DWT. These types of DWTs allow more flexibility and control over the filter design, but they also introduce some trade-offs, such as redundancy, complexity, and distortion. You can find some examples of biorthogonal and non-orthogonal DWTs in this [repository].

"""

"""    
    x (tensor): 
        Input of shape: (N{Batch size}, in_C{input_channel}, in_L{input_Length of signal}) 

"""


"""
Change the Model to Incorporate the involvement of channels along with the signals.


"""




# Reference:
# https://notebook.community/QuantScientist/Deep-Learning-Boot-Camp/day02-PyTORCH-and-PyCUDA/PyTorch/31-PyTorch-using-CONV1D-on-one-dimensional-data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensions

N_FEATURES = 32
N_MULT_FACTOR = 4

# Size of first linear layer
N_HIDDEN=N_FEATURES * N_MULT_FACTOR

# CNN kernel size
N_CNN_KERNEL = 3 # Pre-defined
MAX_POOL_KERNEL = 4 # Pre-defined


N_OUTPUT_FILTER_SIZE = 6 # output filter size
N_CHANNELS = 1 # number of input signal channels

DEBUG_ON=True

def debug(x):
    if DEBUG_ON:
        print ('(x.size():' + str (x.size()))

class FilterConv(nn.Module):
    def __init__(self, n_feature, n_hidden=N_HIDDEN, n_output=N_OUTPUT_FILTER_SIZE, n_cnn_kernel=N_CNN_KERNEL, n_mult_factor=N_MULT_FACTOR, n_channels = N_CHANNELS):
        """
        Params
        ------
        * n_feature: int
            The input signal length, Currently for 1D only.
        * n_hidden: int
            The number of hidden layer neurons.
        * n_output: int
            The desired low_pass filter size.
        * n_cnn_kernel: int
            The length of CNN kernel.
        * n_mult_factor: int
            The n_hidden/n_features.
        """
        super(FilterConv, self).__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_output =  n_output 
        self.n_cnn_kernel = n_cnn_kernel
        self.n_mult_factor = n_mult_factor
        self.n_l2_hidden=self.n_hidden * (self.n_mult_factor - self.n_cnn_kernel + 3)
                        
        self.linear = nn.Sequential(
            torch.nn.Linear(self.n_feature, self.n_hidden),
            torch.nn.Dropout(p=1 -.85),            
            torch.nn.LeakyReLU (0.1),            
            torch.nn.BatchNorm1d(self.n_hidden, eps=1e-05, momentum=0.1, affine=True)            
        )                
        self.conv = nn.Sequential(     
            # Block 1       
            torch.nn.Conv1d(in_channels = self.n_feature,out_channels = self.n_hidden, 
                            kernel_size=(self.n_cnn_kernel,), stride=(1,), padding=(1,)),
            torch.nn.Dropout(p=1 -.75),            
            torch.nn.LeakyReLU (0.1),
            torch.nn.BatchNorm1d(self.n_hidden, eps=1e-05, momentum=0.1, affine=True),      
        )                       
        self.out = nn.Sequential(
            torch.nn.Linear(self.n_l2_hidden,
                            self.n_output),  
        )                

        
    def forward(self, x):

        """ Forward pass of the FilterConv.

        Args:
            x (tensor): Input of shape: (N{Batch size}, in_C{input_channel}, in_L{input_Length of signal})

        Returns:
            h0 (tensor)
            the low pass filter for wavelet Transform.
        """

        batch_size = x.data.shape[0] # must be calculated here in forward() since its is a dynamic size        
        x = self.linear(x)  
        debug(x)              
        # for CNN        
        x = x.view(batch_size,self.n_feature,self.n_mult_factor)
        debug(x)              
        x = self.conv(x)
        debug(x)              
        # for Linear layer
        # NOTE: self.n_l2_hidden is equal to x.shape[1]*x.shape[2]
        x = x.view(batch_size, x.shape[1]*x.shape[2]) # Modifies the shape for self.out -> basically flattening the tensor             
        debug(x)              
        x=self.out(x)   
        debug(x)
        return x


net = FilterConv(n_feature=N_FEATURES)   # define the network    
net.to(device=device)

summary(net,input_size=(N_FEATURES,))

x = torch.rand([2,N_FEATURES])
debug(x)

y = net(x)





















