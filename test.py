import torch
from awave.filtermodel import FilterConv
from visualization import *
from config import *
from icecream import ic

model = torch.load("models/awave.filtermodel__BATCH-32__EPOCH-5__DATA-ADCF__FILTER-10__TIME-1703848653.323739.pth")
model.to(DEVICE)

data = torch.load(DATA_PATH)
# ic(data.shape, x[0].shape)
x = torch.split(data, min(BATCH_SIZE*500, data.size(0)), 0)


y = model(x[0])
ic(len(x[0]))
ic(y.shape)

for id in range(100,112):
    h0 = y[id]
    sig = x[0][id]
    plot_waveform(sig,4100)
    # ic(filter)
    high = torch.reshape(low_to_high(torch.reshape(h0, [1, 1, h0.size(0)])),[h0.size(0)])
    low = h0
    plot_filter_banks(low, high)

    # break

