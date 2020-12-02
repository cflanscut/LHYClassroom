import torch

d_in = 3
d_out = 4
linear_module = torch.nn.Linear(d_in, d_out)
example_tensor = torch.tensor([[1., 2, 3], [4, 5, 6]])
out_tensor = torch.tensor([[1., 2, 3, 4], [2, 3, 4, 5]])
transformed = linear_module(example_tensor)

print('example_tensor', example_tensor.shape)
print('transformed', transformed.shape)
print('w:', linear_module.weight)
print('b:', linear_module.bias)
