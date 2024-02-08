import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from icecream import ic

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv1d, nn.BatchNorm1d, nn.Linear)):
            nn.init.normal_(m.weight.data,0.0,0.005 ) # play with these parms

# great for tanh or sigmoid-type activations;
def initialize_weights_xavier(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.BatchNorm1d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

# great for relu-type activations;
def initialize_weights_he(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.BatchNorm1d, nn.Linear)):
            # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1) # 'a' - negative slope used in leaky-relu
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)    
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

# def plot_grad_flow(named_parameters):
#     ave_grads = []
#     layers = []
#     for n, p in named_parameters:
#         if(p.requires_grad) and ("bias" not in n):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean())
#     plt.plot(ave_grads, alpha=0.3, color="b")
#     plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
#     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
#     plt.xlim(xmin=0, xmax=len(ave_grads))
#     plt.xlabel("Layers")
#     plt.ylabel("average gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.show()


def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


def tuple_dim(x):
    tot_dim = 0
    for i in range(len(x)):
        shape = torch.tensor(x[i].shape)
        tot_dim += torch.prod(shape).item()
    return tot_dim


def tuple_to_tensor(x):
    batch_size = x[0].size(0)
    J = len(x)
    y = torch.tensor([]).to(x[0].device)
    list_of_size = [0]
    for j in range(J):
        a = x[j].reshape(batch_size, -1)
        y = torch.cat((y, a), dim=1)
        list_of_size.append(list_of_size[-1] + a.shape[1])
    return (y, list_of_size)


def tensor_to_tuple(y, d, list_of_size):
    x = []
    J = len(list_of_size) - 1
    for j in range(J):
        n0 = list_of_size[j]
        n1 = list_of_size[j + 1]
        x.append(y[:, n0:n1].reshape(d[j].shape))
    return tuple(x)


def init_filter(x, init_factor, noise_factor, const_factor):

    '''add random noise to tensor
    Params
    ------
    * x: torch.tensor;
        input
    * init_factor: float;

    * noise_factor: float;;
        amount of noise added to original filter
        
    * const_factor: float;
        amount of constant added to original filter
    '''

    shape = x.shape
    x = init_factor * x + noise_factor * torch.randn(shape) + const_factor * torch.ones(shape)
    return x


def pad_within(x, stride=2, start_row=0, start_col=0):
    w = x.new_zeros(stride, stride)
    if start_row == 0 and start_col == 0:
        w[0, 0] = 1
    elif start_row == 0 and start_col == 1:
        w[0, 1] = 1
    elif start_row == 1 and start_col == 0:
        w[1, 0] = 1
    else:
        w[1, 1] = 1
    if len(x.shape) == 2:
        x = x[None, None]
    return F.conv_transpose2d(x, w.expand(x.size(1), 1, stride, stride), stride=stride, groups=x.size(1)).squeeze()


def low_to_high(x):
    """Converts lowpass filter to highpass filter. Input must be of shape (1,1,n) where n is length of filter via quadrature mirror filters
    """
    n = x.size(2)
    seq = (-1) ** torch.arange(n, device=x.device)
    y = torch.flip(x, (0, 2)) * seq
    return y

def low_to_high_parallel(x):
    """Converts lowpass filter`s to highpass filter`s. Input must be of shape (B, 1,1,n) where n is length of filter via quadrature mirror filters & B is batch size
    """
    B, _, _, n = x.size()
    seq = (-1) ** torch.arange(n, device=x.device)
    seq = seq.view(1, 1, 1, n).expand(B, 1, 1, n)
    y = torch.flip(x, (3,)) * seq
    return y

def conv2d_parallel(x, h, p, s, C):
    x_reshape = x.reshape(-1, x.shape[-2], x.shape[-1])
    h_reshape = h.reshape(-1, 1, h.shape[-2], h.shape[-1])
    # print(x_reshape.shape, h_reshape.shape)
    out = F.conv2d(x_reshape, h_reshape, padding=p, stride=s, groups=C*x.shape[0])
    # print(out.shape)
    return out.reshape(-1, h.shape[1], out.shape[-2], out.shape[-1])

def conv_transpose2d_parallel(lo, hi, g0, g1, pad, s, C=None):

    lo_reshape = lo.reshape(-1, lo.shape[-2], lo.shape[-1])
    hi_reshape = hi.reshape(-1, hi.shape[-2], hi.shape[-1])

    g0_reshape = g0.reshape(-1, 1, g0.shape[-2], g0.shape[-1])
    g1_reshape = g1.reshape(-1, 1, g1.shape[-2], g1.shape[-1])
    # print(lo.shape, hi.shape, lo_reshape.shape, hi_reshape.shape)
    # print(g0.shape, g1.shape, g0_reshape.shape, g1_reshape.shape)

    out = F.conv_transpose2d(lo_reshape, g0_reshape, stride=s, padding=pad, groups=C*lo.shape[0]) + \
    F.conv_transpose2d(hi_reshape, g1_reshape, stride=s, padding=pad, groups=C*lo.shape[0])
    return out.reshape(-1, C, out.shape[-2], out.shape[-1])
    
def conv1d_parallel(input, filters, padding, stride, groups=None):
    # v = F.conv1d(h0, h0, stride=2, padding=n)
    B, C, H, W = input.size()
    _, out_channels, kC, kW = filters.size()

    # Reshape input to (B, C, H*W)
    input = input.view(B, H * W)
    # Reshape filters to (in_channels * out_channels, 1, kH, kW)
    filters = filters.view(B * out_channels, kC, kW)
    # Apply transposed 2D convolution with groups=in_channels*out_channels
    if(type(padding) == tuple):
        padding = tuple(int(x) for x in padding)
    # ic(input.shape, filters.shape)
    output = F.conv1d(input, filters, stride=stride, padding=padding, groups=B)
    # Reshape output to (batch, out_channels, C, H, W)
    output = output.view(B, out_channels,1, output.size(1))

    return output

def get_wavefun(w_transform, level=5):
    '''Get wavelet function from wavelet object.
    Params
    ------
    w_transform: obj
        DWT1d or DWT2d object
    '''
    h0 = w_transform.h0
    h1 = low_to_high(h0)

    h0 = list(h0.squeeze().detach().cpu().numpy())[::-1]
    h1 = list(h1.squeeze().detach().cpu().numpy())[::-1]

    my_filter_bank = (h0, h1, h0[::-1], h1[::-1])
    my_wavelet = pywt.Wavelet('My Wavelet', filter_bank=my_filter_bank)
    wave = my_wavelet.wavefun(level=level)
    (phi, psi, x) = wave[0], wave[1], wave[4]

    return phi, psi, x


def dist(wt1, wt2):
    """function to compute distance between two wavelets 
    """
    _, psi1, _ = get_wavefun(wt1)
    _, psi2, _ = get_wavefun(wt2)

    if len(psi1) > len(psi2):
        psi2 = np.pad(psi2, (0, len(psi1) - len(psi2)), mode='constant', constant_values=(0,))
    if len(psi1) < len(psi2):
        psi1 = np.pad(psi1, (0, len(psi2) - len(psi1)), mode='constant', constant_values=(0,))

    distance = []
    # circular shift 
    for i in range(len(psi1)):
        psi1_r = np.roll(psi1, i)
        d = np.linalg.norm(psi1_r - psi2)
        distance.append(d.item())
    # flip filter
    psi1_f = psi1[::-1]
    for i in range(len(psi1)):
        psi1_r = np.roll(psi1_f, i)
        d = np.linalg.norm(psi1_r - psi2)
        distance.append(d.item())

    return min(distance)


def get_1dfilts(w_transform):
    '''Get 1d filters from DWT1d object.
    Params
    ------
    w_transform: obj
        DWT1d object
    '''
    if w_transform.wt_type == 'DWT1d':
        h0 = w_transform.h0.squeeze().detach().cpu()
        h1 = low_to_high(w_transform.h0)
        h1 = h1.squeeze().detach().cpu()
        h0 = F.pad(h0, pad=(0, 0), mode='constant', value=0)
        h1 = F.pad(h1, pad=(0, 0), mode='constant', value=0)
        return (h0, h1)
    else:
        raise ValueError('no such type of wavelet transform is supported')


def get_2dfilts(w_transform):
    '''Get 2d filters from DWT2d object.
    Params
    ------
    w_transform: obj
        DWT2d object
    '''
    if w_transform.wt_type == 'DTCWT2d':
        h0o = w_transform.xfm.h0o.data
        h1o = w_transform.xfm.h1o.data
        h0a = w_transform.xfm.h0a.data
        h1a = w_transform.xfm.h1a.data
        h0b = w_transform.xfm.h0b.data
        h1b = w_transform.xfm.h1b.data

        # compute first level wavelet filters
        h0_r = F.pad(h0o.squeeze().detach().cpu(), pad=(0, 1), mode='constant', value=0)
        h0_i = F.pad(h0o.squeeze().detach().cpu(), pad=(1, 0), mode='constant', value=0)
        h1_r = F.pad(h1o.squeeze().detach().cpu(), pad=(0, 1), mode='constant', value=0)
        h1_i = F.pad(h1o.squeeze().detach().cpu(), pad=(1, 0), mode='constant', value=0)

        lh_filt_r1 = h0_r.unsqueeze(0) * h1_r.unsqueeze(1) / np.sqrt(2)
        lh_filt_r2 = h0_i.unsqueeze(0) * h1_i.unsqueeze(1) / np.sqrt(2)
        lh_filt_i1 = h0_i.unsqueeze(0) * h1_r.unsqueeze(1) / np.sqrt(2)
        lh_filt_i2 = h0_r.unsqueeze(0) * h1_i.unsqueeze(1) / np.sqrt(2)
        filt_15r = lh_filt_r1 - lh_filt_r2
        filt_165r = lh_filt_r1 + lh_filt_r2
        filt_15i = lh_filt_i1 + lh_filt_i2
        filt_165i = lh_filt_i1 - lh_filt_i2

        hh_filt_r1 = h1_r.unsqueeze(0) * h1_r.unsqueeze(1) / np.sqrt(2)
        hh_filt_r2 = h1_i.unsqueeze(0) * h1_i.unsqueeze(1) / np.sqrt(2)
        hh_filt_i1 = h1_i.unsqueeze(0) * h1_r.unsqueeze(1) / np.sqrt(2)
        hh_filt_i2 = h1_r.unsqueeze(0) * h1_i.unsqueeze(1) / np.sqrt(2)
        filt_45r = hh_filt_r1 - hh_filt_r2
        filt_135r = hh_filt_r1 + hh_filt_r2
        filt_45i = hh_filt_i1 + hh_filt_i2
        filt_135i = hh_filt_i1 - hh_filt_i2

        hl_filt_r1 = h1_r.unsqueeze(0) * h0_r.unsqueeze(1) / np.sqrt(2)
        hl_filt_r2 = h1_i.unsqueeze(0) * h0_i.unsqueeze(1) / np.sqrt(2)
        hl_filt_i1 = h1_i.unsqueeze(0) * h0_r.unsqueeze(1) / np.sqrt(2)
        hl_filt_i2 = h1_r.unsqueeze(0) * h0_i.unsqueeze(1) / np.sqrt(2)
        filt_75r = hl_filt_r1 - hl_filt_r2
        filt_105r = hl_filt_r1 + hl_filt_r2
        filt_75i = hl_filt_i1 + hl_filt_i2
        filt_105i = hl_filt_i1 - hl_filt_i2

        fl_filt_reals = [filt_15r, filt_45r, filt_75r, filt_105r, filt_135r, filt_165r]
        fl_filt_imags = [filt_15i, filt_45i, filt_75i, filt_105i, filt_135i, filt_165i]

        # compute second level wavelet filters
        h0_a = h0a.squeeze().detach().cpu()
        h0_b = h0b.squeeze().detach().cpu()
        h1_a = h1a.squeeze().detach().cpu()
        h1_b = h1b.squeeze().detach().cpu()

        lh_filt_r1 = pad_within(h0_b.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=0) / np.sqrt(2)
        lh_filt_r2 = pad_within(h0_a.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=1) / np.sqrt(2)
        lh_filt_i1 = pad_within(h0_a.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=1) / np.sqrt(2)
        lh_filt_i2 = pad_within(h0_b.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=0) / np.sqrt(2)
        filt_15r = lh_filt_r1 - lh_filt_r2
        filt_165r = lh_filt_r1 + lh_filt_r2
        filt_15i = lh_filt_i1 + lh_filt_i2
        filt_165i = lh_filt_i1 - lh_filt_i2

        hh_filt_r1 = pad_within(h1_a.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=1) / np.sqrt(2)
        hh_filt_r2 = pad_within(h1_b.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=0) / np.sqrt(2)
        hh_filt_i1 = pad_within(h1_b.unsqueeze(0) * h1_a.unsqueeze(1), start_row=1, start_col=0) / np.sqrt(2)
        hh_filt_i2 = pad_within(h1_a.unsqueeze(0) * h1_b.unsqueeze(1), start_row=0, start_col=1) / np.sqrt(2)
        filt_45r = hh_filt_r1 - hh_filt_r2
        filt_135r = hh_filt_r1 + hh_filt_r2
        filt_45i = hh_filt_i1 + hh_filt_i2
        filt_135i = hh_filt_i1 - hh_filt_i2

        hl_filt_r1 = pad_within(h1_a.unsqueeze(0) * h0_b.unsqueeze(1), start_row=0, start_col=1) / np.sqrt(2)
        hl_filt_r2 = pad_within(h1_b.unsqueeze(0) * h0_a.unsqueeze(1), start_row=1, start_col=0) / np.sqrt(2)
        hl_filt_i1 = pad_within(h1_b.unsqueeze(0) * h0_b.unsqueeze(1), start_row=0, start_col=0) / np.sqrt(2)
        hl_filt_i2 = pad_within(h1_a.unsqueeze(0) * h0_a.unsqueeze(1), start_row=1, start_col=1) / np.sqrt(2)
        filt_75r = hl_filt_r1 - hl_filt_r2
        filt_105r = hl_filt_r1 + hl_filt_r2
        filt_75i = hl_filt_i1 + hl_filt_i2
        filt_105i = hl_filt_i1 - hl_filt_i2

        sl_filt_reals = [filt_15r, filt_45r, filt_75r, filt_105r, filt_135r, filt_165r]
        sl_filt_imags = [filt_15i, filt_45i, filt_75i, filt_105i, filt_135i, filt_165i]

        return (fl_filt_reals, fl_filt_imags), (sl_filt_reals, sl_filt_imags)

    elif w_transform.wt_type == 'DWT2d':
        h0 = w_transform.h0.squeeze().detach().cpu()
        h1 = low_to_high(w_transform.h0)
        h1 = h1.squeeze().detach().cpu()
        h0 = F.pad(h0, pad=(0, 0), mode='constant', value=0)
        h1 = F.pad(h1, pad=(0, 0), mode='constant', value=0)

        filt_ll = h0.unsqueeze(0) * h0.unsqueeze(1)
        filt_lh = h0.unsqueeze(0) * h1.unsqueeze(1)
        filt_hl = h1.unsqueeze(0) * h0.unsqueeze(1)
        filt_hh = h1.unsqueeze(0) * h1.unsqueeze(1)

        return (h0, h1), (filt_ll, filt_lh, filt_hl, filt_hh)

    else:
        raise ValueError('no such type of wavelet transform is supported')
