import torch
from awave.transform1d import DWT1d



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

x = torch.load('data/audio_data_correct_format.pth')
awt = DWT1d()
awt.fit(x)
