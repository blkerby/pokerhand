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
from grouped_adam import GroupedAdam
from high_order_act_pytorch import HighOrderActivation, HighOrderActivationB
from tame_pytorch import ManifoldModule, L1Linear
from tame_pytorch import Network as TameNetwork

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


def approx_l1_ball_projection(x: torch.tensor, dim: int, num_iters: int) -> torch.tensor:
    sgn = torch.sgn(x)
    return sgn * approx_simplex_projection(torch.abs(x), dim, num_iters)


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

class Preprocessor():
    def fit(self, X):
        # suit = X[:, 0::2]
        # rank = X[:, 1::2]
        # self.encoder = sklearn.preprocessing.OneHotEncoder()
        # self.encoder.fit(suit)
        # self.scaler = sklearn.preprocessing.StandardScaler(with_std=False)
        # self.scaler.fit(rank)
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(X)

    def transform(self, X):
        return self.scaler.transform(X)
        # suit = X[:, 0::2]
        # rank = X[:, 1::2]
        # encoded_suit = self.encoder.transform(suit).todense()
        # scaled_rank = self.scaler.transform(rank)
        # return np.concatenate([encoded_suit, scaled_rank], axis=1)

preprocessor = Preprocessor()
preprocessor.fit(raw_train_X.T)
train_X = torch.from_numpy(preprocessor.transform(raw_train_X.T)).to(torch.float32)
test_X = torch.from_numpy(preprocessor.transform(raw_test_X.T)).to(torch.float32)

ensemble_size = 1
networks = [TameNetwork(widths=[train_X.shape[1]] + [32, 32, 32] + [10],
                    pen_lin_coef=0.0,  # 0.001,
                    pen_lin_exp=5.0,
                    scale_init=100.0,
                    scale_factor=30.0, #50.0,
                    bias_factor=1.0,
                    pen_act=0.0,
                    pen_scale=0.0,  #0.01,
                    dtype=torch.float32,
                    device=torch.device('cpu'))
            for _ in range(ensemble_size)]

_, Y_cnt = torch.unique(train_Y, return_counts=True)
Y_p = Y_cnt.to(torch.float32) / torch.sum(Y_cnt).to(torch.float32)
Y_log_p = torch.log(Y_p)
for net in networks:
    # net.output_bias.data[:] = Y_log_p
    net.lin_layers[-1].bias.data[:] = Y_log_p / net.lin_layers[-1].bias_factor


# optimizers = [torch.optim.Adam(networks[i].parameters(), lr=0.001, betas=(0.995, 0.995))
#               for i in range(ensemble_size)]
batch_size = 2048
lr0 = 0.02
lr1 = 0.02
beta0 = 0.98
beta1 = 0.98
reaper_factor0 = 0.0
reaper_factor1 = 0.0  #0.1
# act_reaper_factor0 = 0.0
# act_reaper_factor1 = 0.2
pen_act0 = 0.0
pen_act1 = 0.0 # 1e-4
# target_act0 = 0.3
# target_act1 = 0.1
# pen_lin_coef0 = 0.0
# pen_lin_coef1 = 0.001
# pen_scale_coef0 = 5e-5
# pen_scale_coef1 = 5e-5
# grad_max = 1e-6
# optimizers = [
#     torch.optim.Adam(networks[i].parameters(), lr=lr0, betas=(0.95, 0.95), eps=1e-15)
#     for i in range(ensemble_size)]
# optimizers = [
#     SAMOptimizer(base_optimizer=torch.optim.Adam(networks[i].parameters(), lr=lr0, betas=(0.95, 0.95), eps=1e-15), rho=0.05)
#     for i in range(ensemble_size)]
optimizers = [GroupedAdam(networks[i].parameters(), lr=lr0, betas=(beta0, beta0), eps=1e-15)
              for i in range(ensemble_size)]
# optimizers = [torch.optim.Adam(networks[i].parameters(), lr=lr0, betas=(beta0, beta0), eps=1e-15)
#               for i in range(ensemble_size)]
# optimizers = [torch.optim.SGD(networks[i].parameters(), lr=lr0, momentum=0.95)
#               for i in range(ensemble_size)]
# optimizers = [torch.optim.Adam(networks[i].parameters(), lr=0.003, betas=(0.5, 0.5))
#               for i in range(ensemble_size)]

logging.info(optimizers[0])

average_params = [[torch.zeros_like(p) for p in net.parameters()] for net in networks]
# average_batch_norm_running_mean = [[torch.zeros_like(b._buffers['running_mean']) for b in net.bn_layers] for net in networks]
# average_batch_norm_running_var = [[torch.zeros_like(b._buffers['running_var']) for b in net.bn_layers] for net in networks]
average_param_beta = 0.995
average_param_weight = 0.0
iteration = 1

with torch.no_grad():
    for net in networks:
        for mod in net.modules():
            if isinstance(mod, ManifoldModule):
                mod.project()
for _ in range(1, 50001):
    frac = min(iteration / 4000, 1.0)
    for net in networks:
        for layer in net.act_layers:
            layer.pen_act = frac * pen_act1 + (1 - frac) * pen_act0
    #         layer.target_act = frac * target_act1 + (1 - frac) * target_act0
    #     for layer in net.lin_layers:
    #         layer.pen_coef = frac * pen_lin_coef1 + (1 - frac) * pen_lin_coef0
    #         layer.pen_scale_coef = frac * pen_scale_coef1 + (1 - frac) * pen_scale_coef0

    total_loss = 0.0
    total_obj = 0.0
    for j, net in enumerate(networks):
        net.train()

        batch_ind = torch.randint(0, train_X.shape[0], [batch_size])
        batch_X = train_X[batch_ind, :]
        batch_Y = train_Y[batch_ind]

        net.zero_grad()
        P = net(batch_X)
        train_loss = compute_loss(P, batch_Y)
        obj = train_loss + net.penalty()
        obj.backward()

        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e-5)

        lr = lr0 * (1 - frac) + lr1 * frac
        beta = beta0 * (1 - frac) + beta1 * frac
        optimizers[j].param_groups[0]['lr'] = lr
        optimizers[j].param_groups[0]['betas'] = (beta, beta)
        # optimizers[j].param_groups[0]['betas'] = (0.99, 0.99)
        optimizers[j].step()
        total_loss += train_loss
        total_obj += obj
        # print(train_loss, obj)

        reaper_factor = frac * reaper_factor1 + (1 - frac) * reaper_factor0
        # act_factor = frac * act_reaper_factor1 + (1 - frac) * act_reaper_factor0
        with torch.no_grad():
            for mod in net.modules():
                if isinstance(mod, L1Linear):
                    # noise = torch.rand_like(mod.weights_pos_neg.param)
                    # mod.weights_pos_neg.param *= (1 + noise_factor * noise)
                    mod.weights_pos_neg.param *= (1 + reaper_factor * lr)
                    # mod.bias -= act_factor * lr

        with torch.no_grad():
            for mod in net.modules():
                if isinstance(mod, ManifoldModule):
                    mod.project()
            # for layer in net.lin_layers:
            #     weight = layer.weight
            #     weight_sgn = torch.sgn(weight)
            #     weight_abs = torch.abs(weight)
            #     lr = optimizers[j].param_groups[0]['lr']
            #     layer.weight.data[:, :] = weight_sgn * torch.clamp(weight_abs - l1_pen_coef * lr, min=0.0)
            for i, p in enumerate(net.parameters()):
                average_params[j][i] = average_param_beta * average_params[j][i] + (1 - average_param_beta) * p
            # for i, b in enumerate(net.bn_layers):
            #     average_batch_norm_running_mean[j][i] = average_param_beta * average_batch_norm_running_mean[j][i] + (
            #                 1 - average_param_beta) * b._buffers['running_mean']
            #     average_batch_norm_running_var[j][i] = average_param_beta * average_batch_norm_running_var[j][i] + (
            #                 1 - average_param_beta) * b._buffers['running_var']
        average_param_weight = average_param_beta * average_param_weight + (1 - average_param_beta)

    if iteration % 500 == 0:
        with torch.no_grad():
            saved_params = [[p.data.clone() for p in net.parameters()] for net in networks]
            # saved_batch_norm_running_mean = [[b._buffers['running_mean'].clone() for b in net.bn_layers] for net
            #                                  in networks]
            # saved_batch_norm_running_var = [[b._buffers['running_var'].clone() for b in net.bn_layers] for net
            #                                 in networks]
            for j, net in enumerate(networks):
                for i, p in enumerate(net.parameters()):
                    p.data = average_params[j][i] / average_param_weight
                # for i, b in enumerate(net.bn_layers):
                #     b._buffers['running_mean'].copy_(average_batch_norm_running_mean[j][i] / average_param_weight)
                #     b._buffers['running_var'].copy_(average_batch_norm_running_var[j][i] / average_param_weight)

            for net in networks:
                net.eval()
            test_P = sum(net(test_X) for net in networks) / ensemble_size
            test_loss = compute_loss(test_P, test_Y)
            test_acc = compute_accuracy(test_P, test_Y)

            for j, net in enumerate(networks):
                for i, p in enumerate(net.parameters()):
                    p.data = saved_params[j][i]
                # for i, b in enumerate(net.bn_layers):
                #     b._buffers['running_mean'].copy_(saved_batch_norm_running_mean[j][i])
                #     b._buffers['running_var'].copy_(saved_batch_norm_running_var[j][i])

            # sat_fracs = []
            # for layer in networks[0].act_layers:
            #     sat_fracs.append(torch.sum(layer.cnt_sat.to(torch.float32)) / torch.sum(layer.cnt_act))
            # sat_fracs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in sat_fracs))
            #

            # wt_fracs = []
            # for layer in networks[0].lin_layers:
            #     # wt_fracs.append(torch.mean((layer.weights_pos_neg.param > 0).to(torch.float32)))
            #     weights = layer.weights_pos_neg.param[:layer.input_width, :] - \
            #               layer.weights_pos_neg.param[layer.input_width:, :]
            #     wt_fracs.append(torch.mean((weights != 0).to(torch.float32)))

            # act_fracs = []
            # for layer in networks[0].act_layers:
            #     act_fracs.append(torch.mean((layer.X >= 0).to(torch.float32)))
            # act_fracs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_fracs))

            act_avgs = []
            act_abs_avgs = []
            act_nz_avgs = []
            for layer in networks[0].act_layers:
                act_avgs.append(torch.mean(layer.out))
                act_abs_avgs.append(torch.mean(torch.abs(layer.out)))
                act_nz_avgs.append(torch.mean((layer.out != 0).to(torch.float32)))
            act_avgs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_avgs))
            act_abs_avgs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_abs_avgs))
            act_nz_avgs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_nz_avgs))

            wt_fracs = []
            for layer in networks[0].lin_layers:
                weights = layer.weights_pos_neg.param[:layer.input_width, :] - layer.weights_pos_neg.param[layer.input_width:, :]
                wt_fracs.append(torch.sum(weights != 0).to(torch.float32) / layer.weights_pos_neg.param.shape[1])
                # weights = layer.weight
                # wt_fracs.append(torch.mean((weights != 0).to(torch.float32)))
            wt_fracs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in wt_fracs))

            # scales = []
            # for layer in networks[0].lin_layers:
            #     scales.append(torch.mean(torch.abs(layer.scale) * layer.scale_factor))
            # scales_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in scales))
            scales_fmt = '{:.3f}'.format(networks[0].scale * networks[0].scale_factor)

            lr = optimizers[0].param_groups[0]['lr']
            # lr = optimizers[0].param_groups[0]['lr']
            logging.info(
                "{}: lr={:.6f}, train={:.6f}, obj={:.6f}, test={:.6f}, acc={:.6f}, wt={}, scale={}, act={}, abs={}, nz={}, gn={}".format(
                    iteration,
                    lr, total_loss / ensemble_size / batch_X.shape[0],
                        total_obj / ensemble_size / batch_X.shape[0],
                    float(test_loss / test_X.shape[0]),
                    float(test_acc),
                    wt_fracs_fmt, scales_fmt, act_avgs_fmt, act_abs_avgs_fmt, act_nz_avgs_fmt, gn))
    iteration += 1

torch.set_printoptions(linewidth=120)
print(sklearn.metrics.confusion_matrix(test_Y, torch.argmax(test_P, dim=1)))

for net in networks:
    net.eval()
Y1L = networks[0].lin_layers[0](train_X)
Y1A = networks[0].act_layers[0](Y1L)
Y2L = networks[0].lin_layers[1](Y1A)
Y2A = networks[0].act_layers[1](Y2L)
Y3L = networks[0].lin_layers[2](Y2A)
Y3A = networks[0].act_layers[2](Y3L)
Y4L = networks[0].lin_layers[3](Y3A)

print(torch.sum(Y1A != 0, dim=0))
print(torch.sum(Y2A != 0, dim=0))
print(torch.sum(Y3A != 0, dim=0))
# print(torch.sum(Y6 != 0, dim=0))
