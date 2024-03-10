import torch
import numpy as np
import matplotlib.pyplot as plt
import pywt
from awave.transform2d import DWT2d
from PIL import Image
from torchvision import transforms
from visualization import plot2DimageWithModel, plot_filter_banks
from awave.utils.misc import coeffs_to_array2D, compression

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
# awt2 = torch.load('models/awave.transform2d/filtersize_16-batchsize_64-epochs_50-LR_0.001-J16.pth')
model = awt2.filter_model
model.eval()
awt = DWT2d(filter_model = model, J=2, device=device, useExistingFilter=False, wave='db3').to(device=device)

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

## UnComment to load elon!!
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
# plot_filter_banks(low_pass_filter, high_pass_filter)


# Getting the coefficients of the Wavelet Transform
coeffs = awt(image_input_to_model)


def sparse_vector_encoding(wavelet_coefficients):

    # Coarse coefficients
    coarse_coefficients = wavelet_coefficients[0]
    # Exclude the coarsest level
    fine_coefficients = wavelet_coefficients[1:]

    # Calculate the number of levels and the dimensions
    levels = len(fine_coefficients)
    batches, channels, _, _ = coarse_coefficients.shape
    P, Q = fine_coefficients[0].shape[-2], fine_coefficients[0].shape[-1]
    num_vectors = 3 * (P // (2 ** levels)) * (Q // (2 ** levels))
    vector_dim = 3 * sum([4 ** i for i in range(1, levels + 1)])

    # Initialize the encoded vectors
    encoded_sparse_vectors = torch.zeros(batches, channels, 3, (P // (2 ** levels)), (Q // (2 ** levels)) , vector_dim)
    print(encoded_sparse_vectors.shape)

    # Fill the encoded vectors using the parent-offspring relationship
    for level, coeffs in enumerate(fine_coefficients, start=1):
        LH_band, HL_band, HH_band = coeffs[..., 0, :, :], coeffs[..., 1, :, :], coeffs[..., 2, :, :]
        print(LH_band.shape, HL_band.shape, HH_band.shape)
        # Insert LH, HL, and HH band coefficients
        for itr, band in enumerate([LH_band, HL_band, HH_band]):
            for i in range(P // (2 ** level)):
                for j in range(Q // (2 ** level)):
                    encoded_sparse_vectors[..., itr, i // 2 ** (levels - level), j // 2 ** (levels - level)] = band[..., i, j]

    encoded_sparse_vectors = encoded_sparse_vectors.flatten(start_dim=2, end_dim=4)
    return coarse_coefficients, encoded_sparse_vectors

coarse_coefficients, encoded_vectors = sparse_vector_encoding(coeffs)
print(encoded_vectors.shape)  # Print the shape of encoded vectors


concatenated_vector = encoded_vectors.flatten(start_dim=2, end_dim=3)
print(concatenated_vector.shape)

plt.plot(concatenated_vector[0][0].detach().numpy())
plt.show()


""" Applying compressed Sensing on the Wavelet coefficients"""




