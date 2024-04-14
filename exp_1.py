"""
Does training on Image dataset and testing on the transformation of the Image changes the outcome?
or vice versa ...
"""
import pywt
import torch
import numpy as np
import matplotlib.pyplot as plt

from awave.transform2d import DWT2d
from PIL import Image
from torchvision import transforms
from visualization import plot2DimageWithModel, plot_filter_banks
from awave.utils.misc import coeffs_to_array2D, compression
from awave.utils.misc import get_wavefun

# np.random.seed()

def low_to_high(x):
    """Converts lowpass filter to highpass filter. Input must be of shape (1,1,n) where n is length of filter via quadrature mirror filters
    """
    if x.shape != [1, 1, x.shape[-1]]:
        x = x.reshape(1, 1, -1)
    n = x.size(2)
    seq = (-1) ** torch.arange(n, device=x.device)
    y = torch.flip(x, (0, 2)) * seq
    return y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Model
awt2 = torch.load('models/awave.transform2d/2D_STL10WaveletTransformJ16.pth', map_location=device).to(device)
model = awt2.filter_model.eval()
awt = DWT2d(filter_model = model, J = 3, device=device, useExistingFilter=False, wave='db3').to(device=device)

# Load the Image
data = torch.load('data/stl10_test.pth')
image = data[np.random.randint(0,len(data))].to(device)

# UnComment to load elon!!
# image = Image.open("material/licensed-image.jpg")
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# image = transform(image)

# Compute the 2D Fourier transform
fourier_image = torch.fft.fftshift(torch.fft.fft2(image.to(torch.cfloat)))

s = fourier_image.shape
print(s)

image_input_to_model = torch.log(torch.abs(fourier_image))

image_input_to_model = image_input_to_model.reshape([1, s[0], s[1], s[2]])
low_pass_filter = awt.filter_model(image_input_to_model)
high_pass_filter = low_to_high(low_pass_filter)

# Plotting the wavelet and Scaling functions
# plot_filter_banks(low_pass_filter, level = 2)


# Getting the coefficients of the Wavelet Transform
coeffs = awt(image_input_to_model)
print(coeffs[0].shape)

plot2DimageWithModel(awt, image_input_to_model, coeffs)
    



""" Well it does recreates!! But i dont know if this persists if Model trained on Fourier basis"""