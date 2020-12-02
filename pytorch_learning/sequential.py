import torch
from torch import nn

d_in = 3
d_hidden = 4
d_out = 1
model = torch.nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(),
                            nn.Linear(d_hidden, d_out), nn.Sigmoid())
example_tensor = torch.tensor([[1., 2, 3], [4, 5, 6]])
transformed = model(example_tensor)
print(transformed.shape)
params = model.parameters()
for param in params:
    print(param)
