import torch
import matplotlib.pyplot as plt
from icecream import ic
# -------------------------------------------
"""Quadrature/Conjugate filters generation."""
# x = torch.rand([1,1,20])
# n = x.size(2)
# seq = (-1) ** torch.arange(n)
# print(seq)
# y = torch.flip(x, (0, 2)) * seq
# print(x)
# print(y)
# -------------------------------------------
"""Quadrature/Conjugate filters Condition."""
# h0 = torch.rand([1,1,4])
# n = h0.size(2)
# ic(h0)   
# h_f = torch.fft.fft(h0)
# mod = abs(h_f) ** 2 
# ic(h_f)
# ic(mod)
# cmf_identity = mod[0, 0, :n // 2] + mod[0, 0, n // 2:]
# ic(cmf_identity)
# print(h_f)
# -------------------------------------------
"""Visualizing trained 2D model on CIFAR-10"""
# def denormalize(img):
#     mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
#     std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
#     img = img * std + mean  # Apply the reverse formula
#     return img

# # Make sure to clip the values to [0, 1] if you're planning to visualize the images
# # img_denormalized = denormalize(img)
# # img_denormalized.clamp_(0, 1)

# awt = torch.load('models/awave.transform2d__BATCH-512__EPOCH-1__DATA-all-losses__FILTER-20__TIME-1708893136.8584223.pth')

# data = torch.load('data/cifar10_test.pth')
# # plt.imshow(denormalize(data[1]).permute(1, 2, 0))
# # plt.title("Original Image")
# # plt.show()

# s = data[1].shape
# img = data[1].reshape(1, s[0], s[1], s[2])
# coeffs = awt.forward(img)
# recon_x = awt.inverse(coeffs)
# recon_x = recon_x.squeeze().detach()

# plt.imshow(denormalize(recon_x).permute(1, 2, 0))
# plt.title("Approximation Image")
# plt.show()

# -------------------------------------------
"""Visualizing the trained 2D model's filters and reconstruction and images."""
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image
from awave.utils.misc import low_to_high


def denormalize(img):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    img = img * std + mean  # Apply the reverse formula
    return img

def plot2D(model):
    data = torch.load('data/cifar10_test.pth')
    image = data[np.random.randint(0,10000)].to(model.device)

    s = image.shape
    image = image.reshape(1, s[0], s[1], s[2])
    coeffs = model.forward(image)
    # print(coeffs)
    recon_x = model.inverse(coeffs)

    recon_x = recon_x.squeeze().detach().cpu()
    image = image.squeeze().detach().cpu()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original Image
    axes[0].imshow(denormalize(image).permute(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Approximation Image
    axes[1].imshow(denormalize(recon_x).permute(1, 2, 0))
    axes[1].set_title("Approximation Image")
    axes[1].axis('off')

    plt.show()

def plot2DD(model, img):
    image = img.to(device)
    coeffs = model.forward(image)

    recon_x = model.inverse(coeffs)

    recon_x = recon_x.squeeze().detach().cpu()
    image = image.squeeze().detach().cpu()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original Image
    axes[0].imshow(denormalize(image).permute(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Approximation Image
    axes[1].imshow(denormalize(recon_x).permute(1, 2, 0))
    axes[1].set_title("Approximation Image")
    axes[1].axis('off')

    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

awt = torch.load('models/awave.transform2d/filtersize_16-batchsize_64-epochs_50-LR_0.001-J16.pth')
data = torch.load('data/cifar10_test.pth')
image = data[np.random.randint(0,10000)].to(device)
s = image.shape
image = image.reshape([1, s[0], s[1], s[2]])
low_pass_filter = awt.filter_model(image).squeeze()

# # Normalizing to make sum = 0
# low = low_pass_filter -  torch.mean(low_pass_filter)
# high = low_to_high(low.reshape(1,1,-1)).squeeze()

# fig, ax = plt.subplots(1,1,figsize=(10,5))
# low = low.detach().numpy()
# high = high.detach().numpy()

# phi = np.convolve(low, low)
# psi = np.convolve(high, low)


# # Plot the scaling and wavelet functions
# ax.plot(phi, label='Scaling Function (phi)')
# ax.plot(psi, label='Wavelet Function (psi)')
# plt.legend(loc='best')
# plt.show()

"""For plotting the original and approximate"""
# plot2D(awt)



""" Elon Musk Transform"""
img = Image.open("material/licensed-image.jpg")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img = transform(img)
print(img.shape)
s1 = img.shape
img = img.reshape([1, s1[0], s1[1], s1[2]])
plot2DD(awt, img)

# -------------------------------------------
# from visualization import plot2D, plot2DPrime

# data = torch.load('data/cifar10_test.pth')
# plt.imshow(denormalize(data[69]).permute(1, 2, 0))
# plt.title("Original Image")
# plt.show()
# img = denormalize(data[69])
# plot2DPrime(img.detach().numpy())

# -------------------------------------------
# h0 = torch.rand([32, 1, 1, 8])
# h1 = torch.rand([32, 1, 1, 8])

# g0_col = h0.reshape((h0.size(0), 1, -1, 1))
# g1_col = h1.reshape((h1.size(0), 1, -1, 1))
# g0_row = h0.reshape((h0.size(0), 1, 1, -1))
# g1_row = h1.reshape((h1.size(0), 1, 1, -1))

# print(g0_col.shape)
# print(g1_col.shape)
# print(g0_row.shape)
# print(g1_row.shape)

# print(h0.numel())

# x = torch.rand([3,3,6,24732])
# lh, hl, hh = torch.unbind(x, dim=2)
# -------------------------------------------






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


