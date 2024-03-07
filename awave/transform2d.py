import torch
import torch.nn as nn
import numpy as np
from awave.utils import lowlevel
from awave.transform import AbstractWT
from awave.utils.misc import init_filter, low_to_high, low_to_high_parallel

from icecream import ic
from visualization import * 
import matplotlib.pyplot as plt


def _get_h0(filter_model, x, useExistingFilter = False, wave='db3', init_factor=1, noise_factor=0, const_factor=0):
    if useExistingFilter:
        h0, _ = lowlevel.load_wavelet(wave)
        h0 = init_filter(h0, init_factor, noise_factor, const_factor)
        batch_size = x.shape[0]
        h0_list = [h0 for i in range(batch_size)]
        h0 = torch.stack(h0_list) 
        # parameterize
        h0 = nn.Parameter(h0, requires_grad=True)
    else :    
        low_pass = filter_model(x)
        h0 = torch.reshape(low_pass, [low_pass.size(0), 1, 1, low_pass.size(1)])

    return h0

class DWT2d(AbstractWT):
    '''Class of 2d wavelet transform 
    Params
    ------
    J: int
        number of levels of decomposition
    wave: str
         which wavelet to use.
         can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
    mode: str
        'zero', 'symmetric', 'reflect' or 'periodization'. The padding scheme

    filter_model: nn.Module;
        A Predefine nn.Module object for determnining filters as a model
    '''

    def __init__(self, wave='db3', filter_model = None, mode='periodization', J=5, init_factor=1, noise_factor=0, const_factor=0, device='cpu', useExistingFilter = False, random_level = False):
        super().__init__()
        h0, _ = lowlevel.load_wavelet(wave)

        # Get the filter model.
        self.filter_model = filter_model
        self.useExistingFilter = useExistingFilter

        self.J = J
        self.random_level = random_level
        self.mode = mode
        self.wt_type = 'DWT2d'
        self.device = device

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = ()
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Initialize the filter
        if self.filter_model is not None:
            self.h0 = _get_h0(self.filter_model, x, self.useExistingFilter, wave='db3', init_factor=1, noise_factor=0, const_factor=0)
            self.h0 = self.h0.to(self.device)
        h1 = low_to_high_parallel(self.h0) 

        # h1 = low_to_high(self.h0)
        # ic(h1.shape)
        # ic(self.h0.shape)

        batch = self.h0.size(0)
        
        h0_col = self.h0.reshape((batch, 1, 1, -1, 1))
        h1_col = h1.reshape((batch, 1, 1, -1, 1))
        h0_row = self.h0.reshape((batch, 1,  1, 1, -1))
        h1_row = h1.reshape((batch, 1, 1, 1, -1))



        # Do a multilevel transform
        J = self.J
        if self.random_level:
            J = np.random.randint(1,self.J)
        for j in range(J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.forward(
                ll, h0_col, h1_col, h0_row, h1_row, mode)
            yh += (high,)

        return (ll,) + yh

    def inverse(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        coeffs = list(coeffs)
        yl = coeffs.pop(0)
        yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        h1 = low_to_high_parallel(self.h0)
        # h1 = low_to_high(self.h0)
        
        # TODO: reshape the following filters correctly.
        batch = self.h0.size(0)

        g0_col = self.h0.reshape((batch, 1, 1, -1, 1))
        g1_col = h1.reshape((batch, 1, 1, -1, 1))
        g0_row = self.h0.reshape((batch, 1, 1, 1, -1))
        g1_row = h1.reshape((batch, 1, 1, 1, -1))

        # Do a multilevel inverse transform
        idx = 0
        for h in yh[::-1]:
            # print(f"Reconstructing layers {idx}-----------------------------------")
            if h is None:
                # TODO: Not modified following line because never using this line
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)   
            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]

            ll = lowlevel.SFB2D.forward(
                ll, h, g0_col, g1_col, g0_row, g1_row, mode)
            idx = idx + 1

        return ll
