import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import logging
from icecream import ic
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

N_INFEATURES = 1024 # Signal Length
N_INCHANNELS = 1 # number of input signal channels
N_OUTPUT_FILTER_SIZE = 6 # output filter size


DEBUG_ON=False

def debug(x):
    if DEBUG_ON:
        print ('(x.size():' + str (x.size()))

class FilterConv(nn.Module):
    def __init__(self, in_channels = N_INCHANNELS, out_channels=N_OUTPUT_FILTER_SIZE):
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
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels,8,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm1d(8),
            nn.Dropout1d(1-.85),
            nn.LeakyReLU(0.1),

            nn.Conv1d(8, 16,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Dropout1d(1-.85),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),

            nn.Conv1d(16, 32,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Dropout1d(1-.85),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            # nn.Conv1d(32, 64,kernel_size=4,stride=2,padding=1,bias=False),
            # nn.Dropout1d(1-.85),
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(0.1),

            # nn.Conv1d(64, 128,kernel_size=4,stride=2,padding=1,bias=False),
            # nn.Dropout1d(1-.85),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(0.1),

            nn.Flatten(),
        )
        self.out = nn.Linear(4096, out_features=out_channels)
        # summary(self,N_INCHANNELS)        
        
    def forward(self, x):

        """ Forward pass of the FilterConv.

        Args:
            x (tensor): Input of shape: (N{Batch size}, in_C{input_channel}, in_L{input_Length of signal})

        Returns:
            h0 (tensor)
            the low pass filter for wavelet Transform.
        """
        x = self.conv(x)
        debug(x)
        x = self.out(x)
        debug(x)
        return x


# net = FilterConv()   # define the network    
# net.to(device=device)


# x = torch.rand([1024,1,N_INFEATURES])
# ic(x.shape)

# y = net(x)
# ic(y.shape)





















