import torch
import numpy as np
x_torch = torch.tensor([0.1, 0.2, 0.3])
x_numpy = np.array([0.1, 0.2, 0.3])
print(x_torch)
print(x_numpy)
print(torch.from_numpy(x_numpy), x_torch.numpy())
# 求正则化
print(torch.norm(x_torch))
# 求平均
print(torch.mean(x_torch, dim=0))
# reshape功能
N, C, W, H = 10000, 3, 25, 25
X = torch.randn((N, C, W, H))
print(X.view(-1, C, 625).shape)
# 相加减的时候，维度不同，如果是空或者是1则会自动broadcast，取大的维度

# gradien descent
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
c = a + b
d = b + 1
e = c + d
print(c)
print(d)
print(e)

# CPU创建tensor，GPU运算
""" cpu = torch.device("cpu")
gpu = torch.device("cuda:1")
x = torch.rand(10)
print(x)
x = x.to(gpu)
print(x)
x = x.to(cpu)
print(x) """


def f(x):
    return (x - 2)**2


x = torch.tensor([1.0], requires_grad=True)
y = f(x)
y.backward()
print(x.grad)

d = 2
n = 50
X = torch.randn(n, d)
true_w = torch.tensor([[-1.0], [2.0]])
y = X @ true_w + torch.randn(n, 1) * 0.1
print(y.shape)


def model(X, w):
    return X @ w


def rss(y, y_hat):
    return torch.norm(y - y_hat)**2 / n


w = torch.tensor([[1.], [0]], requires_grad=True)
step_size = 0.1
for i in range(200):
    y_hat = model(X, w)
    loss = rss(y, y_hat)
    loss.backward()
    w.data = w.data - step_size * w.grad
    w.grad.detach()
    w.grad.zero_()

print(w.view(2).detach().numpy())
