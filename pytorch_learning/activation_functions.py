import torch
activation_fn = torch.nn.ReLU()
example_tensor = torch.tensor([-1, 1., 0])
activated = activation_fn(example_tensor)
print(activated)
