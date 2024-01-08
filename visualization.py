import matplotlib.pyplot as plt
import pywt
from awave.utils.misc import low_to_high
import torch
from icecream import ic

countfb = 0
countsg = 0

def plot_filter_banks(low, high):
    
    low = low.detach().numpy()
    high = high.detach().numpy()
    [dec_lo, dec_hi, rec_lo, rec_hi] = [low, high, low, high]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,7))
    ax1.stem(dec_lo) 
    ax1.set_title('Decomposition low-pass filter')
    ax2.stem(dec_hi)
    ax2.set_title('Decomposition high-pass filter')
    ax3.stem(rec_lo)
    ax3.set_title('Reconstruction low-pass filter')
    ax4.stem(rec_hi)
    ax4.set_title('Reconstruction high-pass filter')

    global countfb
    plt.savefig(f'res/filter_banks/filter_bank{countfb}.png')
    # plt.show()
    countfb +=1

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

    # Assuming low, high, and waveform are already defined
    low = low.detach().numpy()
    high = high.detach().numpy()
    [dec_lo, dec_hi, rec_lo, rec_hi] = [low, high, low, high]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot the first set of plots
    axs[0, 0].stem(dec_lo)
    axs[0, 0].set_title('Decomposition low-pass filter')
    axs[0, 1].stem(dec_hi)
    axs[0, 1].set_title('Decomposition high-pass filter')

    axs[1, 0].stem(rec_lo)
    axs[1, 0].set_title('Reconstruction low-pass filter')
    axs[1, 1].stem(rec_hi)
    axs[1, 1].set_title('Reconstruction high-pass filter')

    # Plot the waveform covering the whole bottom row
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    # Create a single subplot for the waveform
    ax_waveform = plt.subplot(2, 1, 2)
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
