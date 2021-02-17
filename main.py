import torch
from typing import List, Optional
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.metrics
import logging
# import moe_pytorch
import sparselin_pytorch
import math


logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("pokerhand.log"),
                              logging.StreamHandler()])

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
    def __init__(self, shape: List[int], dim: int, dtype=torch.float32, device=None, num_iters=8):
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
        self.weights_pos_neg = Simplex([(input_width + 1) * 2, output_width], dim=0, dtype=dtype, device=device,
                                       num_iters=num_iters)
        # self.scale_factor = scale_factor
        # init_scale = math.sqrt(input_width)
        # self.scale = torch.nn.Parameter(torch.full([output_width], init_scale / scale_factor, dtype=dtype, device=device))
        # self.bias = torch.nn.Parameter(torch.zeros([output_width], dtype=dtype, device=device))

    def forward(self, X):
        raw_weights = self.weights_pos_neg.param[:(self.input_width + 1), :] - self.weights_pos_neg.param[(self.input_width + 1):, :]
        weights = raw_weights[1:, :]
        bias = raw_weights[0, :]
        assert X.shape[1] == self.input_width
        # return torch.matmul(X, weights) * (self.scale.view(1, -1) * self.scale_factor) + self.bias.view(1, -1)
        # return torch.matmul(X, weights) #+ self.bias.view(1, -1)
        return torch.matmul(X, weights) + bias

    def penalty(self, pen_exp):
        p = self.weights_pos_neg.param
        return torch.mean(1 - (1 - p) ** pen_exp - p)

class Maxout(torch.nn.Module):
    def __init__(self, width, arity):
        super().__init__()
        self.width = width
        self.arity = arity

    def forward(self, X):
        X1 = X.view(self.width, self.arity, X.shape[1])
        return torch.max(X1, dim=1)[0]


# class MinMaxout(torch.nn.Module):
#     def __init__(self, width, min_arity, max_arity):
#         super().__init__()
#         self.width = width
#         self.min_arity = min_arity
#         self.max_arity = max_arity
#
#     def forward(self, X):
#         X1 = X.view(X.shape[0], self.width, self.min_arity, self.max_arity)
#         return torch.max(torch.min(X1, dim=2)[0], dim=2)[0]

class Minout(torch.nn.Module):
    def __init__(self, width, min_arity):
        super().__init__()
        self.width = width
        self.min_arity = min_arity

    def forward(self, X):
        X1 = X.view(X.shape[0], self.width, self.min_arity)
        return torch.min(X1, dim=2)[0]


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


class ReSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # out = torch.asinh(torch.clamp(X, min=0))
        # out = torch.sqrt(torch.clamp(X, min=0) + 0.25) - 0.5
        # out = 1 - 1 / (1 + torch.clamp(X, min=0))
        out = torch.clamp(X, min=0.0, max=1.0)
        self.frac_nonzero = float(torch.sum(out != 0.0)) / float(X.shape[0] * X.shape[1])
        self.mean_act = torch.mean(out)
        return out

class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        out = torch.clamp(X, min=0)
        self.frac_nonzero = float(torch.sum(out != 0.0)) / float(X.shape[0] * X.shape[1])
        self.mean_act = torch.mean(out)
        self.mean_sat = torch.mean(out - X)
        return out

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
                 pen_act_coef: float = 0.0,
                 pen_sat_coef: float = 0.0,
                 pen_lin_coef: float = 0.0,
                 pen_lin_exp: float = 2.0,
                 init_scale: float = 1.0,
                 scale_factor: float = 1.0,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.widths = widths
        self.depth = len(widths) - 1
        self.pen_act_coef = pen_act_coef
        self.pen_sat_coef = pen_sat_coef
        self.pen_lin_coef = pen_lin_coef
        self.pen_lin_exp = pen_lin_exp
        self.scale_factor = scale_factor
        self.scale = torch.nn.Parameter(torch.full([], init_scale / scale_factor, dtype=dtype, device=device))
        self.lin_layers = torch.nn.ModuleList([])
        # self.min_layers = torch.nn.ModuleList([])
        self.relu_layers = torch.nn.ModuleList([])
        # self.scale = torch.nn.Parameter(torch.zeros([widths[-1]], dtype=dtype, device=device))
        # self.bias = torch.nn.Parameter(torch.zeros([widths[-1]], dtype=dtype, device=device))
        for i in range(self.depth):
            # self.lin_layers.append(L1Linear(widths[i], widths[i + 1], dtype=dtype, device=device))
            if i == self.depth - 1:
                self.lin_layers.append(L1Linear(widths[i], widths[i + 1], dtype=dtype, device=device))
                # self.lin_layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
            else:
                self.lin_layers.append(L1Linear(widths[i], widths[i + 1], dtype=dtype, device=device))
                # self.lin_layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
                # self.min_layers.append(Minout(widths[i + 1], min_arity=min_arity))
                # self.min_layers.append(MinMaxout(widths[i + 1], min_arity=min_arity, max_arity=1))
                self.relu_layers.append(ReLU())
                # self.relu_layers.append(ReSigmoid())
                # self.act_layers.append(PReLU(widths[i + 1], dtype=dtype, device=device))

    def forward(self, X):
        for i in range(self.depth):
            if i == self.depth - 1:
                X = self.lin_layers[i](X)
            else:
                X = self.lin_layers[i](X)
                # X = self.min_layers[i](X)
                X = self.relu_layers[i](X)
            X = X * (self.scale * self.scale_factor)
        # X = X * (self.scale.view(1, -1) * self.scale_factor) + self.bias.view(1, -1)
        return X

    def penalty(self):
        return sum(layer.mean_act for layer in self.relu_layers) * self.scale * self.scale_factor * self.pen_act_coef + \
               sum(layer.mean_sat for layer in self.relu_layers) * self.scale * self.scale_factor * self.pen_sat_coef + \
               sum(layer.penalty(self.pen_lin_exp) for layer in self.lin_layers) * self.pen_lin_coef
            #+ \
               # torch.sum(self.scale ** 2) * self.pen_scale_coef

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

ensemble_size = 1
networks = [Network(widths=[10] + [32, 32, 32] + [10],
                    pen_act_coef=0.0,
                    pen_sat_coef=0.0,
                    pen_lin_coef=0.0,
                    pen_lin_exp=2.0,
                    scale_factor=2.0,
                    init_scale=1.0,
                    dtype=torch.float32,
                    device=torch.device('cpu'))
            for _ in range(ensemble_size)]

_, Y_cnt = torch.unique(train_Y, return_counts=True)
Y_p = Y_cnt.to(torch.float32) / torch.sum(Y_cnt).to(torch.float32)
Y_log_p = torch.log(Y_p)
# for net in networks:
#     net.lin_layers[-1].bias.data[:] = Y_log_p

# optimizers = [torch.optim.Adam(networks[i].parameters(), lr=0.001, betas=(0.995, 0.995))
#               for i in range(ensemble_size)]
lr0 = 0.003
lr1 = lr0
optimizers = [torch.optim.Adam(networks[i].parameters(), lr=lr0, betas=(0.95, 0.95), eps=1e-15)
              for i in range(ensemble_size)]
# optimizers = [torch.optim.SGD(networks[i].parameters(), lr=0.2, momentum=0.95)
#               for i in range(ensemble_size)]
# optimizers = [torch.optim.Adam(networks[i].parameters(), lr=0.003, betas=(0.5, 0.5))
#               for i in range(ensemble_size)]

logging.info(optimizers[0])

average_params = [[torch.zeros_like(p) for p in net.parameters()] for net in networks]
average_param_beta = 0.98
average_param_weight = 0.0
epoch = 1

with torch.no_grad():
    for net in networks:
        for mod in net.modules():
            if isinstance(mod, ManifoldModule):
                mod.project()
for _ in range(1, 50001):
    frac = min(epoch / 2000, 1.0)
    for j, net in enumerate(networks):
        net.pen_act_coef = frac * 0.02
        net.pen_sat_coef = net.pen_act_coef * 0.0
        net.pen_lin_coef = frac * 0.02
        # optimizers[j].param_groups[0]['lr'] = (1 - frac) * lr0 + frac * lr1

    total_loss = 0.0
    total_obj = 0.0
    for j, net in enumerate(networks):
        optimizers[j].zero_grad()
        P = net(train_X)
        train_loss = compute_loss(P, train_Y)
        obj = train_loss + net.penalty()
        obj.backward()

        # for layer in networks[0].lin_layers:
        #     print(torch.norm(layer.weights_pos_neg.param.grad))
            # print(torch.max(torch.abs(layer.weights_pos_neg.param.grad)), torch.mean(torch.abs(layer.weights_pos_neg.param.grad)))
        # torch.nn.utils.clip_grad_value_(net.parameters(), 0.002)
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
        # for param in net.parameters():
        #     if param.grad is not None:
        #         param.grad[:] = torch.sign(param.grad)

        total_loss += float(train_loss)
        total_obj += float(obj)
        # optimizers[j].param_groups[0]['lr'] = lr0 / (1 + epoch / lr_scale)
        # optimizers[j].param_groups[0]['momentum'] = 0.95
        # optimizers[j].param_groups[0]['betas'] = (0.95, 0.95)
        optimizers[j].step()
        with torch.no_grad():
            for mod in net.modules():
                if isinstance(mod, ManifoldModule):
                    mod.project()
            # for mod in net.modules():
                # if isinstance(mod, L1Linear):
                #     mod.bias.data -= 3e-4
                # if isinstance(mod, torch.nn.Linear):
                #     mod.bias.data -= 0.01
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

            for j, net in enumerate(networks):
                for i, p in enumerate(net.parameters()):
                    p.data = saved_params[j][i]

            act_nonzero = [layer.frac_nonzero for layer in networks[0].relu_layers]
            wt_nonzero = [float(torch.sum(layer.weights_pos_neg.param != 0) / (layer.weights_pos_neg.param.view(-1).shape[0]))
                          for layer in networks[0].lin_layers]
            scale = float(networks[0].scale * networks[0].scale_factor)
            # bias = [float(torch.mean(layer.bias)) for layer in networks[0].lin_layers]
            mean_act = [(networks[0].scale * networks[0].scale_factor * layer.mean_act).tolist() for layer in networks[0].relu_layers]
            mean_sat = [(networks[0].scale * networks[0].scale_factor * layer.mean_sat).tolist() for layer in networks[0].relu_layers]
            logging.info("{}: lr={:.6f}, train={:.6f}, obj={:.6f}, test={:.6f}, acc={:.6f}, scale={:.3f}, act_nz={}, wt_nz={}, act={}, sat={}".format(
                epoch, optimizers[0].param_groups[0]['lr'], total_loss / ensemble_size, total_obj / ensemble_size, float(test_loss), float(test_acc),
                scale, act_nonzero, wt_nonzero, mean_act, mean_sat))
            # scale = torch.mean(abs(networks[0].scale)) * networks[0].scale_factor
            # logging.info("{}: train={:.6f}, obj={:.6f}, test={:.6f}, acc={:.6f}, scale={:.3f}, act_nz={}, wt_nz={}".format(
            #     epoch, total_loss / ensemble_size, total_obj / ensemble_size, float(test_loss), float(test_acc),
            #     scale,
            #     act_nonzero, wt_nonzero))
    epoch += 1

torch.set_printoptions(linewidth=120)
print(sklearn.metrics.confusion_matrix(test_Y, torch.argmax(test_P, dim=1)))

Y1 = networks[0].lin_layers[0](train_X)
Y2 = networks[0].relu_layers[0](Y1)
Y3 = networks[0].lin_layers[1](Y2)
Y4 = networks[0].relu_layers[1](Y3)
Y5 = networks[0].lin_layers[2](Y4)
Y6 = networks[0].relu_layers[2](Y5)
print(torch.sum(Y2 != 0, dim=0))
print(torch.sum(Y4 != 0, dim=0))
print(torch.sum(Y6 != 0, dim=0))