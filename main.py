import torch
from typing import List
import numpy as np
import pandas as pd
import sklearn.preprocessing

def approx_simplex_projection(x: torch.tensor, dim: int, num_iters: int) -> torch.tensor:
    mask = torch.ones(list(x.shape), dtype=x.dtype, device=x.device)
    with torch.no_grad():
        for i in range(num_iters - 1):
            n_act = torch.sum(mask, dim=dim)
            x_sum = torch.sum(x * mask, dim=dim)
            t = (x_sum - 1.0) / n_act
            x1 = x - t.unsqueeze(dim=dim)
            mask = (x1 >= 0).to(x.dtype)
        n_act = torch.sum(mask, dim=dim)
    x_sum = torch.sum(x * mask, dim=dim)
    t = (x_sum - 1.0) / n_act
    x1 = x - t.unsqueeze(dim=dim)
    return x1


class ManifoldModule(torch.nn.Module):
    pass


class Simplex(torch.nn.Module):
    def __init__(self, shape: List[int], dim: int, dtype=torch.float32, device=None, num_iters=6):
        super().__init__()
        self.shape = shape
        self.dim = dim
        self.num_iters = num_iters
        data = torch.rand(shape, dtype=dtype, device=device)
        data = torch.log(data + 1e-15)
        data = data / torch.sum(data, dim=dim).unsqueeze(dim)
        self.param = torch.nn.Parameter(data)

    def project(self):
        self.param.data = approx_simplex_projection(self.param.data, dim=self.dim, num_iters=self.num_iters)


class L1Linear(torch.nn.Module):
    def __init__(self, input_width, output_width, dtype=torch.float32, device=None, num_iters=6):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.weights_pos_neg = Simplex([output_width, input_width * 2], dim=1, dtype=dtype, device=device, num_iters=num_iters)
        self.bias = torch.nn.Parameter(torch.zeros([output_width], dtype=dtype, device=device))

    def forward(self, X):
        weights = self.weights_pos_neg.param[:, :self.input_width] - self.weights_pos_neg.param[:, self.input_width:]
        assert X.shape[0] == self.input_width
        return torch.matmul(weights, X) + self.bias.view(-1, 1)


class Maxout(torch.nn.Module):
    def __init__(self, width, arity):
        super().__init__()
        self.width = width
        self.arity = arity

    def forward(self, X):
        X1 = X.view(self.width, self.arity, X.shape[1])
        return torch.max(X1, dim=1)[0]


class MinMaxout(torch.nn.Module):
    def __init__(self, width, min_arity, max_arity):
        super().__init__()
        self.width = width
        self.min_arity = min_arity
        self.max_arity = max_arity

    def forward(self, X):
        X1 = X.view(self.width, self.min_arity, self.max_arity, X.shape[1])
        return torch.max(torch.min(X1, dim=1)[0], dim=1)[0]



class Network(torch.nn.Module):
    def __init__(self,
                 widths: List[int],
                 min_arity: int,
                 max_arity: int,
                 init_scale: float,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.widths = widths
        self.depth = len(widths) - 1
        self.lin_layers = torch.nn.ModuleList([])
        self.act_layers = torch.nn.ModuleList([])
        self.scale = torch.nn.Parameter(torch.full([], init_scale, dtype=dtype, device=device))
        for i in range(self.depth):
            self.lin_layers.append(L1Linear(widths[i], min_arity * max_arity * widths[i + 1], dtype=dtype, device=device))
            self.act_layers.append(MinMaxout(widths[i + 1], min_arity, max_arity))

    def forward(self, X):
        for i in range(self.depth):
            X = X * self.scale
            X = self.lin_layers[i](X)
            X = self.act_layers[i](X)
        return X


def compute_loss(P, Y):
    return torch.nn.functional.cross_entropy(P.t(), Y)

def compute_accuracy(P, Y):
    P_max = torch.argmax(P, dim=0)
    return torch.mean((P_max == Y).to(torch.float32))

def load_data(filename):
    df = pd.read_csv(filename, header=None)
    X = torch.stack(list(torch.from_numpy(df[i].values) for i in range(10))).to(torch.float32)
    Y = torch.from_numpy(df[10].values).to(torch.long)
    return X, Y

raw_train_X, train_Y = load_data('~/nn/datasets/poker/poker-hand-training-true.data')
raw_test_X, test_Y = load_data('~/nn/datasets/poker/poker-hand-testing.data')

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(raw_train_X.T)
train_X = torch.from_numpy(scaler.transform(raw_train_X.T).T).to(torch.float32)
test_X = torch.from_numpy(scaler.transform(raw_test_X.T).T).to(torch.float32)

net = Network(widths=[10] + 4 * [20] + [10],
              min_arity=2,
              max_arity=1,
              init_scale=2.0,
              dtype=torch.float32,
              device=torch.device('cpu'))

optimizer = torch.optim.Adam(net.parameters(), lr=0.015, betas=(0.5, 0.5))
print(optimizer)
print("init_scale: {}".format(net.scale))

average_params = [torch.zeros_like(p) for p in net.parameters()]
average_param_beta = 0.995
average_param_weight = 0.0
epoch = 1

with torch.no_grad():
    for mod in net.modules():
        if isinstance(mod, ManifoldModule):
            mod.project()
for _ in range(1, 50001):
    optimizer.zero_grad()
    P = net(train_X)
    train_loss = compute_loss(P, train_Y)
    train_loss.backward()
    optimizer.step()
    with torch.no_grad():
        for mod in net.modules():
            if isinstance(mod, ManifoldModule):
                mod.project()
        for i, p in enumerate(net.parameters()):
            average_params[i] = average_param_beta * average_params[i] + (1 - average_param_beta) * p
            average_param_weight = average_param_beta * average_param_weight + (1 - average_param_beta)

    if epoch % 100 == 0:
        with torch.no_grad():
            saved_params = [p.data.clone() for p in net.parameters()]
            for i, p in enumerate(net.parameters()):
                p.data = average_params[i] / average_param_weight

            test_P = net(test_X)
            test_loss = compute_loss(test_P, test_Y)
            test_acc = compute_accuracy(test_P, test_Y)
            print("{}: scale={:.6f}, train={:.6f}, test={:.6f}, acc={:.6f}, lr={:.6f}".format(
                epoch, float(net.scale), float(train_loss), float(test_loss), float(test_acc),
              optimizer.param_groups[0]['lr']))

            for i, p in enumerate(net.parameters()):
                p.data = saved_params[i]
    epoch += 1
