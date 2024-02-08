import torch
import os
"""
def make_dataset(signal, signal_len, channels):

    # Calculating the final size of the dataset
    size = len(signal)
    d0 = 0
    print(size)
    for i in range(size):
        d0 += len(signal[i][0])//signal_len
    # Not using the last cut for each signal
    print(d0)

    data = torch.zeros([d0, channels, signal_len])
    print(data.shape)
    curridx = 0      
    for i in range(size):
        print(signal[i].shape)
        y = torch.split(signal[i],signal_len,1)

        # Removing the last Tensor
        if len(y)%signal_len != 0:
            y = y[:len(y)-1]

        for j, x in enumerate(y):   
            data[curridx] = x
            curridx+=1
        

    return data

x = torch.load('data/audio_data_tensor.pth')
print(x.shape)
d = make_dataset(x, 1024, 1)
print(d.shape)
torch.save(d, 'data/audio_data_correct_format.pth')

"""
# x = torch.rand((3,4))
# print(x)
# y = torch.split(x, 2,1)
# print(y, len(y))
# _y = y[:len(y)-1]
# print(_y)
# z = torch.rand((2,1,5))
# print(z, z.shape, z[0])


# """ sample rate = 4100"""

# x = torch.load('data/audio_data.pth')
# mx = 0
# for i in range(len(x)):
#     mx = max(x[i]['waveform'].shape[1], mx)
# y = torch.zeros([len(x), 1, mx])
# cnt = 0
# for idx,sig in enumerate(x):
#     y[idx] = sig['waveform']

# print(y.shape)
# print(y)
# torch.save(y,"data/audio_data_tensor.pth")

def make_cifar_dataset():
    import torch
    import torchvision
    import torchvision.transforms as transforms

    rootdir = './data'
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)

    # Step 1: Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root=rootdir, train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root=rootdir, train=False,
                                        download=True, transform=transform)

    # Step 2: Extract image data from the datasets
    train_data = torch.stack([img for img, _ in trainset], dim=0)
    test_data = torch.stack([img for img, _ in testset], dim=0)

    # Step 3: Save training and testing data into separate .pth files
    torch.save(train_data, f'{rootdir}/ifar10_train_data.pth')
    torch.save(test_data, f'{rootdir}/cifar10_test_data.pth')

    print("Data tensors saved successfully!")



if __name__ == "__main__":
    make_cifar_dataset()