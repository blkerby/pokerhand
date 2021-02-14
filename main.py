import torch
from typing import List
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.metrics
import logging
import moe_pytorch

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

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
    x1 = torch.clamp(x - t.unsqueeze(dim=dim), min=0.0)
    # logging.info(torch.mean(torch.sum(x1, dim=1)))
    return x1  #/ torch.sum(torch.abs(x1), dim=dim).unsqueeze(dim=dim)


class ManifoldModule(torch.nn.Module):
    pass


class Simplex(ManifoldModule):
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
        self.weights_pos_neg = Simplex([output_width, input_width * 2], dim=1, dtype=dtype, device=device,
                                       num_iters=num_iters)
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


class SoftMaxout(torch.nn.Module):
    def __init__(self, width, arity):
        super().__init__()
        self.width = width
        self.arity = arity

    def forward(self, X):
        X1 = X.view(self.width, self.arity, X.shape[1])
        Xm = torch.softmax(X1, dim=1)
        return torch.sum(X1 * Xm, dim=1)
        # return torch.max(torch.min(X1, dim=1)[0], dim=1)[0]


class Multiplier(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, X):
        X1 = X[:self.width, :]
        X2 = X[self.width:, :]
        return X1 * torch.sigmoid(X2)


class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.clamp(X, min=0)

class PReLU(torch.nn.Module):
    def __init__(self, width, dtype=torch.float32, device=None):
        super().__init__()
        self.width = width
        self.left_slope = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        self.right_slope = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        # self.position = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        self.position = torch.nn.Parameter(torch.zeros([width], dtype=dtype, device=device))
        # self.bias = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        self.bias = torch.nn.Parameter(torch.zeros([width], dtype=dtype, device=device))

    def forward(self, X):
        delta = X - self.position.view(1, -1)
        return self.bias.view(1, -1) + torch.where(delta >= 0,
                                                   delta * self.right_slope.view(1, -1),
                                                   delta * self.left_slope.view(1, -1))


class Network(torch.nn.Module):
    def __init__(self,
                 widths: List[int],
                 num_experts: int,
                 selection_size: int,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.widths = widths
        self.depth = len(widths) - 1
        # self.lin_layers = torch.nn.ModuleList([])
        self.gate_layers = torch.nn.ModuleList([])
        self.expert_layers = torch.nn.ModuleList([])
        # self.act_layers = torch.nn.ModuleList([])
        for i in range(self.depth):
            lin = torch.nn.Linear(widths[i], num_experts)
            lin.weight.data.zero_()
            lin.bias.data.zero_()
            self.gate_layers.append(lin)
            experts = torch.nn.ModuleList([])
            for _ in range(num_experts):
                experts.append(torch.nn.Sequential(
                    torch.nn.Linear(widths[i], widths[i + 1]),
                    PReLU(widths[i + 1], dtype=dtype, device=device)
                ))
            self.expert_layers.append(moe_pytorch.MixtureOfExperts(experts, selection_size))
            # self.act_layers.append(PReLU(widths[i + 1], dtype=dtype, device=device))
            # self.act_layers.append(ReLU(widths[i + 1], dtype=dtype, device=device))

    def forward(self, X):
        for i in range(self.depth):
            G = self.gate_layers[i](X)
            X = self.expert_layers[i](X, G)
            # X = self.lin_layers[i](X)
            # X = self.act_layers[i](X)
        return X


def compute_loss(P, Y):
    return torch.nn.functional.cross_entropy(P, Y)


def compute_accuracy(P, Y):
    P_max = torch.argmax(P, dim=1)
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
train_X = torch.from_numpy(scaler.transform(raw_train_X.T)).to(torch.float32)
test_X = torch.from_numpy(scaler.transform(raw_test_X.T)).to(torch.float32)

ensemble_size = 4
networks = [Network(widths=[10] + [8, 8, 8] + [10],
                    num_experts=4,
                    selection_size=2,
                    dtype=torch.float32,
                    device=torch.device('cpu'))
            for _ in range(ensemble_size)]

optimizers = [torch.optim.Adam(networks[i].parameters(), lr=0.01, betas=(0.99, 0.99))
              for i in range(ensemble_size)]

# optimizers = [torch.optim.Adam(networks[i].parameters(), lr=0.003, betas=(0.5, 0.5))
#               for i in range(ensemble_size)]

logging.info(optimizers[0])

average_params = [[torch.zeros_like(p) for p in net.parameters()] for net in networks]
average_param_beta = 0.99  # 0.98
average_param_weight = 0.0
epoch = 1

# with torch.no_grad():
#     for net in networks:
#         for mod in net.modules():
#             if isinstance(mod, ManifoldModule):
#                 mod.project()
for _ in range(1, 50001):
    total_loss = 0.0
    for j, net in enumerate(networks):
        optimizers[j].zero_grad()
        P = net(train_X)
        train_loss = compute_loss(P, train_Y)
        train_loss.backward()
        total_loss += float(train_loss)
        # optimizers[j].param_groups[0]['lr'] = 0.01
        # optimizers[j].param_groups[0]['betas'] = (0.995, 0.995)
        optimizers[j].step()
        with torch.no_grad():
            # for mod in net.modules():
            #     if isinstance(mod, ManifoldModule):
            #         mod.project()
            for i, p in enumerate(net.parameters()):
                average_params[j][i] = average_param_beta * average_params[j][i] + (1 - average_param_beta) * p
                average_param_weight = average_param_beta * average_param_weight + (1 - average_param_beta)

    if epoch % 100 == 0:
        with torch.no_grad():
            saved_params = [[p.data.clone() for p in net.parameters()] for net in networks]
            for j, net in enumerate(networks):
                for i, p in enumerate(net.parameters()):
                    p.data = average_params[j][i] / average_param_weight

            test_P = sum(net(test_X) for net in networks) / ensemble_size
            test_loss = compute_loss(test_P, test_Y)
            test_acc = compute_accuracy(test_P, test_Y)

            logging.info("{}: train={:.6f}, test={:.6f}, acc={:.6f}".format(
                epoch, total_loss / ensemble_size, float(test_loss), float(test_acc)))
            # logging.info("{}: scale={:.6f}, train={:.6f}, test={:.6f}, acc={:.6f}".format(
            #     epoch, float(sum(net.scale * net.std_scale for net in networks)) / ensemble_size,
            #            total_loss / ensemble_size, float(test_loss), float(test_acc)))

            for j, net in enumerate(networks):
                for i, p in enumerate(net.parameters()):
                    p.data = saved_params[j][i]
    epoch += 1

torch.set_printoptions(linewidth=120)
print(sklearn.metrics.confusion_matrix(test_Y, torch.argmax(test_P, dim=0)))