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
    return x1  # / torch.sum(torch.abs(x1), dim=dim).unsqueeze(dim=dim)


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
    def __init__(self, input_width, output_width, pen_coef=0.0, pen_exp=2.0, dtype=torch.float32, device=None,
                 num_iters=6):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.pen_coef = pen_coef
        self.pen_exp = pen_exp
        self.weights_pos_neg = Simplex([input_width * 2, output_width], dim=0, dtype=dtype, device=device,
                                       num_iters=num_iters)

    def forward(self, X):
        weights = self.weights_pos_neg.param[:self.input_width, :] - self.weights_pos_neg.param[self.input_width:, :]
        assert X.shape[1] == self.input_width
        return torch.matmul(X, weights)

    def penalty(self):
        p = self.weights_pos_neg.param
        return self.pen_coef * torch.mean(1 - (1 - p) ** self.pen_exp - p)


class HardSigmoid(torch.nn.Module):
    def __init__(self, width: int,
                 init_scale: float,
                 init_bias: float,
                 pen_act_coef: float,
                 pen_act_target_frac: float,
                 pen_sat_coef: float,
                 pen_sat_target_frac: float,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.pen_act_coef = pen_act_coef
        self.pen_act_target_frac = pen_act_target_frac
        self.pen_sat_coef = pen_sat_coef
        self.pen_sat_target_frac = pen_sat_target_frac
        self.bias = torch.nn.Parameter(torch.full([width], init_bias, dtype=dtype, device=device))
        self.scale = torch.nn.Parameter(torch.full([width], init_scale, dtype=dtype, device=device))

    def forward(self, X):
        self.X_scaled = (X + self.bias.view(1, -1)) * self.scale.view(1, -1)
        out = torch.clamp(self.X_scaled, min=0.0, max=1.0)
        self.cnt_rows = X.shape[0]
        self.cnt_act = torch.sum(self.X_scaled >= 0.0, dim=0)
        self.cnt_sat = torch.sum(self.X_scaled >= 1.0, dim=0)
        return out

    def penalty(self):
        bias_coef = self.pen_act_coef * (
                self.cnt_act * (1 - self.pen_act_target_frac) - (self.cnt_rows - self.cnt_act) * self.pen_act_target_frac)
        pen_bias = torch.sum(self.bias * bias_coef)
        scale_coef = self.pen_sat_coef * (
                self.cnt_sat * (1 - self.pen_sat_target_frac) - (self.cnt_act - self.cnt_sat) * self.pen_sat_target_frac)
        pen_scale = torch.sum(self.scale * scale_coef)
        return pen_bias + pen_scale
        # pen_act = self.pen_act_coef * (torch.clamp(self.X_scaled, min=0.0) * (1 - self.pen_act_target_frac) - torch.clamp(self.X_scaled, max=0.0) * self.pen_act_target_frac)
        # pen_sat = self.pen_sat_coef * (torch.clamp(self.X_scaled, min=1.0) * (1 - self.pen_sat_target_frac) - torch.clamp(self.X_scaled, min=0.0, max=1.0) * self.pen_sat_target_frac)
        # return torch.sum(pen_act) + torch.sum(pen_sat)

class Network(torch.nn.Module):
    def __init__(self,
                 widths: List[int],
                 init_scale: float = 1.0,
                 init_bias: float = 0.0,
                 pen_act_coef: float = 0.0,
                 pen_act_target_frac: float = 0.5,
                 pen_sat_coef: float = 0.0,
                 pen_sat_target_frac: float = 0.5,
                 pen_lin_coef: float = 0.0,
                 pen_lin_exp: float = 2.0,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.widths = widths
        self.depth = len(widths) - 1
        self.lin_layers = torch.nn.ModuleList([])
        self.act_layers = torch.nn.ModuleList([])
        self.output_bias = torch.nn.Parameter(torch.zeros([self.widths[-1]], dtype=dtype, device=device))
        self.output_scale = torch.nn.Parameter(torch.ones([self.widths[-1]], dtype=dtype, device=device))
        for i in range(self.depth):
            self.lin_layers.append(L1Linear(widths[i], widths[i + 1],
                                            pen_coef=pen_lin_coef, pen_exp=pen_lin_exp,
                                            dtype=dtype, device=device))
            if i != self.depth - 1:
                self.act_layers.append(HardSigmoid(widths[i + 1],
                                                    init_scale=init_scale,
                                                    init_bias=init_bias,
                                                    pen_act_coef=pen_act_coef,
                                                    pen_act_target_frac=pen_act_target_frac,
                                                    pen_sat_coef=pen_sat_coef,
                                                    pen_sat_target_frac=pen_sat_target_frac,
                                                    dtype=dtype, device=device))

    def forward(self, X):
        for i in range(self.depth):
            X = self.lin_layers[i](X)
            if i != self.depth - 1:
                X = self.act_layers[i](X)
        X = X * self.output_scale.view(1, -1) + self.output_bias.view(1, -1)
        return X

    def penalty(self):
        return sum(layer.penalty() for layer in self.act_layers) + \
               sum(layer.penalty() for layer in self.lin_layers)


def compute_loss(P, Y):
    return torch.nn.functional.cross_entropy(P, Y, reduction='sum')


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
                    # networks = [Network(widths=[10] + 5 * [32] + [10],
                    pen_act_coef=0.0001,
                    pen_act_target_frac=0.1,
                    pen_sat_coef=0.0,
                    pen_sat_target_frac=0.5,
                    pen_lin_coef=0.0,
                    pen_lin_exp=5.0,
                    init_scale=2.0,
                    init_bias=0.0,
                    dtype=torch.float32,
                    device=torch.device('cpu'))
            for _ in range(ensemble_size)]

_, Y_cnt = torch.unique(train_Y, return_counts=True)
Y_p = Y_cnt.to(torch.float32) / torch.sum(Y_cnt).to(torch.float32)
Y_log_p = torch.log(Y_p)
for net in networks:
    net.output_bias.data[:] = Y_log_p

# optimizers = [torch.optim.Adam(networks[i].parameters(), lr=0.001, betas=(0.995, 0.995))
#               for i in range(ensemble_size)]
lr0 = 0.005
lr1 = 0.005
# grad_max = 1e-6
optimizers = [torch.optim.Adam(networks[i].parameters(), lr=lr0, betas=(0.95, 0.95), eps=1e-15)
              for i in range(ensemble_size)]
# optimizers = [torch.optim.SGD(networks[i].parameters(), lr=lr0, momentum=0.95)
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
    frac = min(epoch / 3000, 1.0)

    total_loss = 0.0
    total_obj = 0.0
    for j, net in enumerate(networks):
        net.train()
        optimizers[j].zero_grad()
        P = net(train_X)
        train_loss = compute_loss(P, train_Y)
        obj = train_loss + net.penalty()
        obj.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1e-5)

        total_loss += float(train_loss)
        total_obj += float(obj)
        optimizers[j].param_groups[0]['lr'] = lr0 * (1 - frac) + lr1 * frac
        optimizers[j].step()
        with torch.no_grad():
            for mod in net.modules():
                if isinstance(mod, ManifoldModule):
                    mod.project()
            for i, p in enumerate(net.parameters()):
                average_params[j][i] = average_param_beta * average_params[j][i] + (1 - average_param_beta) * p
        average_param_weight = average_param_beta * average_param_weight + (1 - average_param_beta)

    if epoch % 100 == 0:
        with torch.no_grad():
            saved_params = [[p.data.clone() for p in net.parameters()] for net in networks]
            for j, net in enumerate(networks):
                for i, p in enumerate(net.parameters()):
                    p.data = average_params[j][i] / average_param_weight

            for net in networks:
                net.eval()
            test_P = sum(net(test_X) for net in networks) / ensemble_size
            test_loss = compute_loss(test_P, test_Y)
            test_acc = compute_accuracy(test_P, test_Y)

            for j, net in enumerate(networks):
                for i, p in enumerate(net.parameters()):
                    p.data = saved_params[j][i]

            act_fracs = []
            for layer in networks[0].act_layers:
                act_fracs.append(torch.mean(layer.cnt_act.to(torch.float32)) / layer.cnt_rows)
            act_fracs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_fracs))

            sat_fracs = []
            for layer in networks[0].act_layers:
                sat_fracs.append(torch.sum(layer.cnt_sat.to(torch.float32)) / torch.sum(layer.cnt_act))
            sat_fracs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in sat_fracs))

            wt_fracs = []
            for layer in networks[0].lin_layers:
                wt_fracs.append(torch.mean((layer.weights_pos_neg.param > 0).to(torch.float32)))
            wt_fracs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in wt_fracs))

            logging.info(
                "{}: lr={:.6f}, train={:.6f}, obj={:.6f}, test={:.6f}, acc={:.6f}, act={}, sat={}, wt={}".format(
                    epoch, optimizers[0].param_groups[0]['lr'], total_loss / ensemble_size / train_X.shape[0],
                    total_obj / ensemble_size / train_X.shape[0],
                    float(test_loss / test_X.shape[0]), float(test_acc),
                act_fracs_fmt, sat_fracs_fmt, wt_fracs_fmt))
    epoch += 1

torch.set_printoptions(linewidth=120)
print(sklearn.metrics.confusion_matrix(test_Y, torch.argmax(test_P, dim=1)))

Y1 = networks[0].lin_layers[0](train_X)
Y2 = networks[0].relu_layers[0](Y1)
Y3 = networks[0].lin_layers[1](Y2)
Y4 = networks[0].relu_layers[1](Y3)
# Y5 = networks[0].lin_layers[2](Y4)
# Y6 = networks[0].relu_layers[2](Y5)
print(torch.sum(Y2 != 0, dim=0))
print(torch.sum(Y4 != 0, dim=0))
# print(torch.sum(Y6 != 0, dim=0))
