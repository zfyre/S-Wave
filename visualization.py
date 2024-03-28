import pywt
import torch
from time import time
import numpy as np
import matplotlib.pyplot as plt
from awave.utils.misc import low_to_high, get_wavefun

from icecream import ic
countfb = 0
countsg = 0

def plot_filter_banks(low_pass_filter, level = 1):
    
    fig, ax = plt.subplots(1,1,figsize=(10,5)) 
    phi, psi, x = get_wavefun(low_pass_filter, level=level)
    # Plot the scaling and wavelet functions
    ax.plot(phi, label='Scaling Function (phi)')
    ax.plot(psi, label='Wavelet Function (psi)')
    plt.legend(loc='best')
    plt.plot()
    # plt.show()
    plt.savefig(f'res/{time()}.png')


def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    global countsg
    plt.savefig(f'res/waveform/waveform{countsg}.png')
    countsg +=1
    # plt.show(block=False)

def plotdiag(low, high, waveform, sample_rate):

    plt.close('all')
    # Assuming low, high, and waveform are already defined
    low = low.detach().numpy()
    high = high.detach().numpy()
    [dec_lo, dec_hi, rec_lo, rec_hi] = [low, high, low, high]

    axs1 = plt.subplot(3, 2 ,1)
    axs2 = plt.subplot(3, 2, 2)
    axs3 = plt.subplot(3, 2, 3)
    axs4 = plt.subplot(3, 2, 4)

    # Plot the first set of plots
    axs1.stem(dec_lo)
    axs1.set_title('Decomposition low-pass filter')
    axs2.stem(dec_hi)
    axs2.set_title('Decomposition high-pass filter')
    axs3.stem(rec_lo)
    axs3.set_title('Reconstruction low-pass filter')
    axs4.stem(rec_hi)
    axs4.set_title('Reconstruction high-pass filter')

    # Plot the waveform covering the whole bottom row
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    # Create a single subplot for the waveform
    ax_waveform = plt.subplot(3, 1, 3)
    for c in range(num_channels):
        ax_waveform.plot(time_axis, waveform[c], linewidth=1)
        ax_waveform.grid(True)
        if num_channels > 1:
            ax_waveform.set_ylabel(f"Channel {c+1}")
    ax_waveform.set_xlabel('Time (s)')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    #Show the plot
    # plt.show()
    # Save the plot
    global countsg
    plt.savefig(f'res/plots{countsg}.png')
    countsg +=1

def plot_wavelet(low, high, waveform, sample_rate):

    fig, ax = plt.subplots(2,1,figsize=(10,5))

    low = low.detach().numpy()
    high = high.detach().numpy()

    phi = np.convolve(low, low)
    psi = np.convolve(high, low)

    # Plot the scaling and wavelet functions
    ax[0].plot(phi, label='Scaling Function (phi)')
    ax[0].plot(psi, label='Wavelet Function (psi)')

    #-----------------------------------------------------

    # Plot the waveform covering the whole bottom row
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    # Create a single subplot for the waveform
    for c in range(num_channels):
        ax[1].plot(time_axis, waveform[c], linewidth=1)
        ax[1].grid(True)
        if num_channels > 1:
            ax[1].set_ylabel(f"Channel {c+1}")
    ax[1].set_xlabel('Time (s)')

    plt.legend()
    plt.show(block=False)
    plt.pause(4)
    plt.close()

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()
    sample_rate = sample_rate.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show(block=False)

def plot_loss(train_losses : list):

    epochs = range(1, len(train_losses) + 1)

    # Plotting losses on a logarithmic scale
    plt.semilogy(epochs, train_losses, label='Loss', marker='x',color='black', base = 10, linestyle = '-')

    # Adding title and labels
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Logarithmic Loss')

    # Adding legend
    plt.legend()

    # Displaying the plot
    plt.show()

def denormalize(img):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    img = img * std + mean  # Apply the reverse formula
    return img

def plot2DTestData(model, dataset = 'cifar10'):
    data = torch.load(f'data/{dataset}_test.pth')
    length = len(data)
    image = data[np.random.randint(0,length)].to(model.device)

    s = image.shape
    image = image.reshape(1, s[0], s[1], s[2])
    coeffs = model.w_transform(image)

    recon_x = model.wt_inverse(coeffs)

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



def plot2DimageWithModel(model, image, coeffs = None):
    s = image.shape
    if s != [1, s[-3], s[-2], s[-1]]:
        image = image.reshape(1, s[-3], s[-2], s[-1]).to(model.device)
    if coeffs == None:
        coeffs = model(image)
    recon_x = model.inverse(coeffs)

    recon_x = recon_x.squeeze().detach().cpu()
    image = image.squeeze().detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original Image
    axes[0].imshow(denormalize(image).permute(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Approximation Image
    axes[1].imshow(denormalize(recon_x).permute(1, 2, 0), cmap='gray')
    axes[1].set_title("Approximation Image")
    axes[1].axis('off')

    # plt.show()
