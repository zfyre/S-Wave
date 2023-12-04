import torch
from awave.transform1d import DWT1d
from awave.filtermodel import FilterConv

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FilterConv(in_channels=1,out_channels=6)
model.to(device=device)

x = torch.load('data/audio_data_correct_format.pth')
ic(x.shape)
ic(x[0].shape)
x1 = torch.split(x, 32*100, 0)
ic(len(x1))
ic(x1[0].shape)


# Dry run an example on model
# ic(model(x1[0]).shape)


awt = DWT1d(filter_model=model)
ic(awt.h0)
awt.fit(X=x1[0],batch_size=32,num_epochs=32)
ic(awt.h0)
