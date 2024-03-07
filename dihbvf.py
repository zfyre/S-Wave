# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import pywt

# def denormalize(img):
#     mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
#     std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
#     img = img * std + mean  # Apply the reverse formula
#     return img

# def plot2DTestData(dataset = 'cifar10'):
#     data = torch.load(f'data/{dataset}_test.pth')
#     length = len(data)
#     image = data[np.random.randint(0,length)]
#     print(image.shape)
#     coeffs = pywt.wavedec2(image, 'db2', mode='periodization', level=1)
#     recon_x = torch.Tensor(pywt.waverec2(coeffs, 'db2', mode='periodization'))

#     print(recon_x.shape)
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))

#     # Original Image
#     axes[0].imshow(denormalize(image).permute(1, 2, 0))
#     axes[0].set_title("Original Image")
#     axes[0].axis('off')

#     # Approximation Image
#     axes[1].imshow(denormalize(recon_x).permute(1, 2, 0))
#     axes[1].set_title("Approximation Image")
#     axes[1].axis('off')

#     plt.show()

# plot2DTestData()

import pywt

def daubechies_wavelet(k):
    wavelet_name = 'db' + str(k)
    wavelet = pywt.Wavelet(wavelet_name)
    return wavelet

# Example usage:
k = 4  # Specify the length of the Daubechies wavelet
wavelet = daubechies_wavelet(k)
print("Daubechies wavelet of length", k, ":", wavelet.dec_lo)
