import torch
import numpy as np
import matplotlib.pyplot as plt
import pywt
from awave.transform2d import DWT2d
from PIL import Image
from torchvision import transforms
from visualization import plot2DimageWithModel, plot_filter_banks
from awave.utils.misc import coeffs_to_array2D, compression
from awave.utils.misc import get_wavefun

np.random.seed(654)

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
# awt2 = torch.load('models/awave.transform2d/filtersize_16-batchsize_64-epochs_50-LR_0.001-J16.pth')
model = awt2.filter_model
model.eval()
awt = DWT2d(filter_model = model, J = 3, device=device, useExistingFilter=False, wave='db3').to(device=device)

# awt = torch.load('models/awave.transform2d/filtersize_16-batchsize_64-epochs_5-LR_0.001.pth')
# awt = DWT2d(filter_model = awt_2.filter_model, J=2, device=awt_2.device, useExistingFilter=False, wave='db3',mode='periodization').to(device=awt_2.device)

# Load the Image
data = torch.load('data/stl10_test.pth')
image = data[np.random.randint(0,len(data))].to(device)

# Uncomment to load cameraman!!
# image = pywt.data.camera().astype(np.float32)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5))
# ])
# image = transform(image)

# UnComment to load elon!!
# image = Image.open("material/licensed-image.jpg")
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# image = transform(image)


# Convert the image to gray-scale
# image = transforms.Grayscale(num_output_channels=1)(image)

s = image.shape
print(s)
image = image.reshape([1, s[0], s[1], s[2]])

image_input_to_model = image
# image_input_to_model = torch.cat([image]*3, dim=1)
# print(image_input_to_model.shape)
low_pass_filter = awt.filter_model(image_input_to_model)
high_pass_filter = low_to_high(low_pass_filter)

# Plotting the wavelet and Scaling functions
plot_filter_banks(low_pass_filter, level = awt.J)


# Getting the coefficients of the Wavelet Transform
coeffs = awt(image_input_to_model)
print(coeffs[0].shape)

plot2DimageWithModel(awt, image_input_to_model, coeffs)
    
with torch.no_grad():
    # Assuming temp_coeffs is a tuple
    temp_coeffs = tuple(coeffs[i]/torch.abs(coeffs[i]).max() for i in range(len(coeffs)))
arr = coeffs_to_array2D(temp_coeffs)
# plt.imshow(arr.permute(1, 2, 0))
plt.imshow(torch.abs(arr[0]), cmap='gray')
plt.colorbar()
plt.show()


# Finding the new reconstructions with the new coefficients.
new_coeffs = compression(coeffs, 0.10)
new_arr = coeffs_to_array2D(new_coeffs)
# plt.imshow(torch.abs(new_arr[0]), cmap='gray')
plt.colorbar()
# plot2DimageWithModel(awt, image_input_to_model, new_coeffs)
