import torch
import torch.nn as nn
from awave.transform1d import DWT1d
from awave.transform2d import DWT2d
from awave.filtermodel import FilterConv
from config import *
import time
from icecream import ic
import pywt

from torchvision import datasets, transforms, models



def train1d(data, filter_model, device):

    # Initializing
    awt = DWT1d(filter_model = filter_model, device=device).to(device=device)

    # Training
    awt.fit(X=data,batch_size = BATCH_SIZE, num_epochs = NUM_EPOCHS, lr= LR)

    name = f"models/{awt.__module__}__BATCH-{BATCH_SIZE}__EPOCH-{NUM_EPOCHS}__DATA-{DATA_NAME}__FILTER-{OUT_CHANNELS}__TIME-{time.time()}.pth"
    torch.save(awt, name)

def train2d(data, filter_model, device):
    
    # Initializing
    awt = DWT2d(filter_model = filter_model, J=8, device=device, useExistingFilter=False, wave='coif1').to(device=device)
    # Test Data
    data_test = torch.load('data/cifar10_test.pth')
    # Training
    awt.fit(X=data, X_test=None, batch_size = BATCH_SIZE, num_epochs = NUM_EPOCHS, lr= LR)

    name = f"models/{awt.__module__}__BATCH-{BATCH_SIZE}__EPOCH-{NUM_EPOCHS}__DATA-{DATA_NAME}__FILTER-{OUT_CHANNELS}__TIME-{time.time()}.pth"
    torch.save(awt, name)

if __name__ == "__main__":
    
    """Set the device , 'cpu' by default.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """Provide the filter prediction model. 
    """
    # model = FilterConv(in_channels = IN_CHANNELS, out_channels = OUT_CHANNELS).to(device)
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, OUT_CHANNELS)
    )
    model.to(device)
    print(model)
    
    """Load the data. 
    """

    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5), (0.5))
    ])

    original = transform(pywt.data.camera()).squeeze()
    original = torch.stack([original, original, original])
    data = [original for i in range(100)]
    data = torch.stack(data)

    # data = torch.load(DATA_PATH).to(device)
    # # # ic(data.shape, x[0].shape)
    # x = torch.split(data, min(BATCH_SIZE*500, data.size(0)), 0)
    # data = torch.rand([1000, 3, 32, 32])
    # ic(len(x1))
    # ic(x1[0].shape)
    # Dry run an example on model
    # ic(model(x1[0]).shape)

    """ Following line for CIFAR10 dataset"""
    # data = torch.load(DATA_PATH).to(device)
    ic(data.shape)

    """Train the model. 
    """
    train2d(data, model, device)
    # train1d(data, model, device)