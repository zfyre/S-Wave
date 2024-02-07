import torch

h0 = torch.rand([32, 1, 1, 8])
h1 = torch.rand([32, 1, 1, 8])

g0_col = h0.reshape((h0.size(0), 1, -1, 1))
g1_col = h1.reshape((h1.size(0), 1, -1, 1))
g0_row = h0.reshape((h0.size(0), 1, 1, -1))
g1_row = h1.reshape((h1.size(0), 1, 1, -1))

print(g0_col.shape)
print(g1_col.shape)
print(g0_row.shape)
print(g1_row.shape)

print(h0.numel())

x = torch.rand([3,3,6,24732])
# lh, hl, hh = torch.unbind(x, dim=2)





# import torch
# import numpy as np
# from awave.filtermodel import FilterConv
# from visualization import *
# from config import *
# from icecream import ic


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# awt = torch.load("models/wavelet-colab-256-lr=1e-4.pth", map_location='cpu')
# model = awt.filter_model
# model.to(device=device)

# # print(awt.train_losses)
# # plot_loss(awt.train_losses)

# data = torch.load(DATA_PATH)
# x = torch.split(data, min(BATCH_SIZE*5, data.size(0)), 0)

# ic(data.shape, x[3].shape)

# ic(model)

# y = model(x[3])

# ic(len(x[3]))
# print("Out shape:", y.shape)

# for id in range(100, 120):
#     plt.close()
#     h0 = y[id]
#     sig = x[3   ][id]
#     # plot_waveform(sig,4100)
#     # ic(filter)
#     high = torch.reshape(low_to_high(torch.reshape(h0, [1, 1, h0.size(0)])),[h0.size(0)])
#     low = h0

#     plot_wavelet(low, high, sig, 4100)

#     # plot_filter_banks(low, high)
#     # plotdiag(low, high, sig, 4100)
#     # break



# t = torch.ones([4,1,3])
# print(t)
# print(t.sum((1,2)))

# s = 0.5*(t.sum((1,2)) - 1)**2
# print(s)
# r = s.sum()
# print(r)

# Correct Code Form forward paralleliztion!!
""" 
import torch
import torch.nn.functional as F

# Input shape: (B, C, H, W), B = batch size
input = torch.rand([32, 1, 1, 1024])  # Small batch size and length for illustration
filters = torch.rand([32, 2, 1, 1, 10])
print("Input:")
print(input.shape)
print("\nFilters:")
print(filters.shape)

lohi = F.conv2d(input, filters[0], padding=(0, 8), stride=(1, 2), groups=1)

# Reshape input to (B, C, H*W)
input = input.view(32, 1, 1024)

# Reshape filters to (out_channels, in_channels, kH, kW)
filters = filters.view(32 * 2, 1, 1, 10)


# Apply 2D convolution with groups=2
output = F.conv2d(input, filters, stride=(1, 2), padding=(0, 8), groups=32)

# Reshape output to (batch, out_channels, C, H, W)
output = output.view(32, 2, output.size(1), output.size(2))

# Print input, filters, and output for illustration
print("\nOutput:")
print(output.shape)
print(lohi.shape)

max_abs_diff = torch.max(torch.abs(lohi[0] - output[0]))

# Print the maximum absolute difference
print("\nMaximum Absolute Difference:", max_abs_diff.item())

assert torch.allclose(lohi[0], output[0], atol=1e-6)
"""

"""
import torch
import torch.nn.functional as F

# Input shape: (B, C, H, W), B = batch size
input = torch.rand([32, 1, 1, 40])  # Small batch size and length for illustration
filters = torch.rand([32, 1, 1, 1, 10])
print("Input:")
print(input.shape)
print("\nFilters:")
print(filters.shape)

lohi = F.conv_transpose2d(input, filters[0], stride=(1, 2), padding=(0, 8), groups=1)

# Reshape input to (B, C, H*W)
input = input.view(32, 1,  40)
# Reshape filters to (in_channels * out_channels, 1, kH, kW)
filters = filters.view(32*1, 1, 1, 10)
# Apply transposed 2D convolution with groups=in_channels*out_channels
output = F.conv_transpose2d(input, filters, stride=(1, 2), padding=(0, 8), groups=32)
# Reshape output to (batch, out_channels, C, H, W)
output = output.view(32, 1, output.size(1), output.size(2))

# Print input, filters, and output for illustration
print("\nOutput:")
print(output.shape)
print(lohi.shape)

max_abs_diff = torch.max(torch.abs(lohi[0] - output[0]))

# Print the maximum absolute difference
print("\nMaximum Absolute Difference:", max_abs_diff.item())

assert torch.allclose(lohi[0], output[0], atol=1e-6)
"""

# mod = torch.rand([1,1,10])
# print(mod)
# print(mod[:,:,:5])
# print(mod[:,:,5:])


# awt = torch.load('models/awave.filtermodel__BATCH-32__EPOCH-2__DATA-ADCF__FILTER-6__TIME-1704713867.891409.pth')
# print(awt.filter_model.parameters())


