import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

model = nn.Linear(1, 1, bias=False)
X_simple = torch.tensor([[1.], [2]])
Y_simple = torch.tensor([[-2.], [-4]])

optim = torch.optim.SGD(model.parameters(), lr=0.1)
mse_loss_fn = nn.MSELoss()

for i in range(20):
    y_hat = model(X_simple)
    loss = mse_loss_fn(y_hat, Y_simple)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(model.weight)


class fakedataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        docstring
        """
        return self.x[idx], self.y[idx]


x = np.random.rand(100, 10)
y = np.random.rand(100)
Dataset = fakedataset(x, y)
DataLoader = DataLoader(Dataset, batch_size=4, shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(DataLoader):
    print(i_batch, sample_batched)
