# import torch
# from awave.filtermodel import FilterConv
# from visualization import *
# from config import *
# from icecream import ic

# model = torch.load("models/awave.filtermodel__BATCH-32__EPOCH-5__DATA-ADCF__FILTER-10__TIME-1703848653.323739.pth")
# model.to(DEVICE)

# data = torch.load(DATA_PATH)
# # ic(data.shape, x[0].shape)
# x = torch.split(data, min(BATCH_SIZE*500, data.size(0)), 0)


# y = model(x[0])
# ic(len(x[0]))
# ic(y.shape)

# for id in range(100,112):
#     h0 = y[id]
#     sig = x[0][id]
#     plot_waveform(sig,4100)
#     # ic(filter)
#     high = torch.reshape(low_to_high(torch.reshape(h0, [1, 1, h0.size(0)])),[h0.size(0)])
#     low = h0
#     plot_filter_banks(low, high)

#     # break

# import torch
# import torch.nn.functional as F

# # Input shape: (B, C, H, W), B = batch size

# input = torch.rand([32,1,1,1024])
# filters = torch.rand([32,2,1,1,10])
# print(input.shape)
# print(filters.shape)

# input = input.view(32, 1, 1 * 1 * 1024)  # Reshape to (B, 1, C*H*W)
# filters = filters.view(32, 1 * 1 * 10, 2)  # Reshape to (B, C*H*W, out_channels)
# print(input.shape)
# print(filters.shape)

# output = F.conv2d(input, filters, stride=(2,), groups=32, padding=0)  # Apply convolution with groups=B
# print(output.shape)
# # output = output.view(32, 1, 1, )  
'''
import torch
import torch.nn.functional as F


# Input shape: (B, C, H, W), B = batch size

input = torch.rand([32, 1, 1, 1024])
filters = torch.rand([32, 2, 1, 1, 10])
print(input.shape)
print(filters.shape)

# Reshape input to (B, C, H*W)
input = input.view(32, 1, 1024)

# Reshape filters to (out_channels, in_channels, kH, kW)
filters = filters.view(32 * 2, 1, 1, 10)

# Apply 2D convolution with groups=32
output = F.conv2d(input, filters, stride=(1, 2), padding=(0, 8), groups=32)
print(output.shape)
output = output.view(32, 2, 1, output.size(1), output.size(2))
print(output.shape)
'''

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



'''

# import torch
# import torch.nn.functional as F

# # Small example for debugging
# input = torch.rand([2, 1, 8])  # Smaller batch size and input size
# filters = torch.rand([2, 2, 1, 2])  # Smaller batch size and input size

# # Print initial shapes
# print("Input Shape:", input.shape)
# print("Filters Shape:", filters.shape)

# # Manually compute convolution with the first filter
# lohi = F.conv1d(input, filters[0], padding=2, stride=2)
# print("lohi Shape: ", lohi.shape)
# # Reshape input to (B, C, H*W)
# input = input.view(2, 8)

# # Reshape filters to (out_channels, in_channels, kernel_size)
# filters = filters.view(2 * 2, 1, 2)


# # Apply 1D convolution with groups=2
# output = F.conv1d(input, filters, stride=2, padding=2, groups=2)
# # Reshape output to (batch, out_channels, C, H, W)
# output = output.view(2, 2, output.size(1))
# print("Output Shape: ", output.shape)

# # Print intermediate values
# print("\nIntermediate Values:")
# print("Input after reshape:", input)
# print("Filters after reshape:", filters)
# print("Manual Convolution (lohi):", lohi)
# print("Actual Output:", output)

# # Check the maximum absolute difference
# max_abs_diff = torch.max(torch.abs(lohi[0] - output[0]))

# # Print the maximum absolute difference
# print("\nMaximum Absolute Difference:", max_abs_diff.item())
'''