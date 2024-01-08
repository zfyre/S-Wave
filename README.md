# Learning Sparse Wavelet Representation

- Using a External Model to Predict the low-pass filters and minimizing the loss function like an Autoencoder setup.
- Download the [temporary dataset](https://drive.google.com/file/d/15TvzIC-kLzpb88xCt9dRFSTcPGJGo3F5/view?usp=drive_link) using wget and unzip to the `./data` directory.
- Use the [visualization.py](./visualization.py) to visualize the produced filters etc..

![Image](materials/plots0.png)

```python
# Initialization of Filter Prediction Model:
model = FilterConv(in_channels = IN_CHANNELS, out_channels = OUT_CHANNELS)
model.to(device = DEVICE)

# Initialization of Autoencoder Model:
data = torch.load(DATA_PATH)
awt = DWT1d(filter_model = model)

# Training:
awt.fit(X = data, batch_size = BATCH_SIZE, num_epochs = NUM_EPOCHS)

name = f"{name_of_your_model}.pth"
torch.save(awt, name)
```
currently implemented for 1D, using [transform1d.py](./awave/transform1d.py)

---
## OPEN FOR CONTRIBUTIONS AND MORE IDEAS!!
---
- Idea based on Paper [Recoskie & Mann, 2018](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjGy8SR5_WCAxXEa2wGHdlgCm8QFnoECAkQAw&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1802.02961%23%3A~%3Atext%3DThe%2520learned%2520wavelets%2520are%2520shown%2Clearn%2520from%2520raw%2520audio%2520data.&usg=AOvVaw0TjVoVVJS3c4JWTkyR4SW4&opi=89978449)
- Implementation based on [Yu-Group/adaptive-wavelets](https://github.com/Yu-Group/adaptive-wavelets)
```yaml
@article{ha2021adaptive,
  title={Adaptive wavelet distillation from neural networks through interpretations},
  author={Ha, Wooseok and Singh, Chandan and Lanusse, Francois and Upadhyayula, Srigokul and Yu, Bin},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
