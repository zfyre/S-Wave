import torch
from awave.transform1d import DWT1d
from awave.filtermodel import FilterConv
from config import *
import time
from icecream import ic

"""
START:

STEP1: Dataset Import karooo!! -> done

STEP2*: Dataset ko Desired shape me laao: -> done
    raw data: then in [For 1-d signals this should be 3-dimensional, (num_examples, num_curves/channels_per_example, length_of_curve)] numpy/tensor form,
    NOTE: The fit method will automatically make the dataloader
    NOTE: Keep in mind the inputs to the provided FilterModel as well make it adjust to the input of AbstractWT.

STEP3: Create a FilterModel and initialize it..

STEP4: Create an DWT1d Object and pass the required parameters through it.

STEP5: call the fit function to fit the Wavelet to the Graph.

END

"""

model = FilterConv(in_channels = IN_CHANNELS, out_channels = OUT_CHANNELS)
model.to(device = DEVICE)

data = torch.load(DATA_PATH)
# ic(data.shape, x[0].shape)
x = torch.split(data, min(BATCH_SIZE*500, data.size(0)), 0)

# ic(len(x1))
# ic(x1[0].shape)

# Dry run an example on model
# ic(model(x1[0]).shape)

# Initializing
awt = DWT1d(filter_model = model)
# awt = DWT1d()
ic(awt.h0)

# Training
awt.fit(X=x[0],batch_size = BATCH_SIZE, num_epochs = NUM_EPOCHS)

name = f"models/{model.__module__}__BATCH-{BATCH_SIZE}__EPOCH-{NUM_EPOCHS}__DATA-{DATA_NAME}__FILTER-{OUT_CHANNELS}__TIME-{time.time()}.pth"
# print(name)

torch.save(model, name)
