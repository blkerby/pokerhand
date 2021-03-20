import torch
from typing import List, Optional
import copy
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
from tame_pytorch import Network as TameNetwork, MultiGateNetwork, MultiplexerNetwork, SparseHighOrderActivationB, D2Activation

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


#
# class SAMOptimizer:
#     def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float):
#         self.base_optimizer = base_optimizer
#         self.rho = rho
#
#     def first_step(self):
#         # Save the current parameters, and move against the gradient to bring them to the SAM point
#         self.saved_params = []
#         for group in self.base_optimizer.param_groups:
#             for param in group['params']:
#                 self.saved_params.append(param.data.clone())
#                 eps = 1e-15
#                 param.data += param.grad * (self.rho / (torch.norm(param.grad) + eps))
#
#     def second_step(self):
#         # Restore the old parameters
#         i = 0
#         for group in self.base_optimizer.param_groups:
#             for param in group['params']:
#                 param.data = self.saved_params[i]
#                 i += 1
#
#         # Take a step in the SAM direction
#         self.base_optimizer.step()


raw_train_X, train_Y = load_data('~/nn/datasets/poker/poker-hand-training-true.data')
raw_test_X, test_Y = load_data('~/nn/datasets/poker/poker-hand-testing.data')


class Preprocessor():
    def fit(self, X):
        suit = X[:, 0::2]
        rank = X[:, 1::2]
        self.encoder = sklearn.preprocessing.OneHotEncoder()
        encoded_suit = self.encoder.fit_transform(suit.numpy()).todense()
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(rank.reshape(-1, 1).repeat([1, 5]))
        # unscaled = np.concatenate([encoded_suit, rank.numpy()], axis=1)
        # self.scaler.fit(unscaled)
        # # self.scaler = sklearn.preprocessing.StandardScaler()
        # # self.scaler.fit(X)

    def transform(self, X):
        # return self.scaler.transform(X)
        suit = X[:, 0::2]
        rank = X[:, 1::2]
        encoded_suit = self.encoder.transform(suit).todense()
        scaled_rank = self.scaler.transform(rank)
        out = np.concatenate([encoded_suit, scaled_rank], axis=1)
        return out
        # unscaled = np.concatenate([encoded_suit, rank.numpy()], axis=1)
        # scaled = self.scaler.transform(unscaled)
        # return scaled
        # scaled_rank = self.scaler.transform(rank)
        # return np.concatenate([encoded_suit, scaled_rank], axis=1)

    def augment(self, X):
        X = torch.clone(X)

        # Shuffle the cards
        card_perm = torch.randperm(5)
        X[:, -5:] = X[:, -5 + card_perm]  # Shuffle their ranks
        for suit in range(4):  # Shuffle their encoded suits
            X[:, suit:-5:4] = X[:, suit + card_perm * 4]

        # Shuffle the suits
        suit_perm = torch.randperm(4)
        for card in range(5):
            X[:, (card * 4):((card + 1) * 4)] = X[:, card * 4 + suit_perm]
        return X



# class Preprocessor():
#     def fit(self, X):
#         pass
#
#     def transform(self, X):
#         return X.numpy()
#
#     def augment(self, X):
#         X = torch.clone(X)
#
#         # Shuffle the cards
#         card_perm = torch.randperm(5)
#         X[:, ::2] = X[:, card_perm * 2]        # Shuffle their suits
#         X[:, 1::2] = X[:, card_perm * 2 + 1]   # Shuffle their ranks
#
#         # Shuffle the suits
#         suit_perm = torch.randperm(4)
#         for card in range(5):
#             X[:, ::2] = suit_perm[(X[:, ::2] - 1).to(torch.long)].to(torch.float32) + 1.0
#
#         return X


# class Preprocessor():
#     def fit(self, X):
#         suit = X[:, 0::2]
#         rank = X[:, 1::2]
#         self.suit_encoder = sklearn.preprocessing.OneHotEncoder()
#         self.rank_encoder = sklearn.preprocessing.OneHotEncoder()
#         self.suit_encoder.fit(suit.numpy())
#         self.rank_encoder.fit(rank.numpy())
#
#     def transform(self, X):
#         suit = X[:, 0::2]
#         rank = X[:, 1::2]
#         encoded_suit = self.suit_encoder.transform(suit).todense()
#         encoded_rank = self.rank_encoder.transform(rank).todense()
#         out = np.concatenate([encoded_suit, encoded_rank], axis=1)
#         return out
#
#     def augment(self, X):
#         X = torch.clone(X)
#
#         # Shuffle the cards
#         card_perm = torch.randperm(5)
#         for suit in range(4):  # Shuffle their encoded suits
#             X[:, suit:20:4] = X[:, suit + card_perm * 4]
#         for rank in range(13):  # Shuffle their encoded ranks
#             X[:, (20 + rank)::13] = X[:, 20 + rank + card_perm * 13]
#
#         # Shuffle the suits
#         suit_perm = torch.randperm(4)
#         for card in range(5):
#             X[:, (card * 4):((card + 1) * 4)] = X[:, card * 4 + suit_perm]
#         return X

preprocessor = Preprocessor()
preprocessor.fit(raw_train_X.T)
train_X = torch.from_numpy(preprocessor.transform(raw_train_X.T)).to(torch.float32)
test_X = torch.from_numpy(preprocessor.transform(raw_test_X.T)).to(torch.float32)

# batch_X = train_X[100:101, :]
# print(raw_train_X[:, 100])
# print(batch_X)
# print(preprocessor.augment(batch_X))
#

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = TameNetwork(widths=[train_X.shape[1]] + [32, 16] + [10],
                    bias_factor=1.0,
                    act_init=1.0,
                    act_factor=1.0,
                    scale_init=100.0,
                    scale_factor=50,
                    arity=5,
                    dtype=torch.float32,
                    device=torch.device('cpu'))

        _, Y_cnt = torch.unique(train_Y, return_counts=True)
        Y_p = Y_cnt.to(torch.float32) / torch.sum(Y_cnt).to(torch.float32)
        Y_log_p = torch.log(Y_p)
        self.network.bias.data[:] = Y_log_p

    def project(self):
        for mod in self.network.modules():
            if isinstance(mod, ManifoldModule):
                mod.project()

    def train_step(self, batch_X, batch_Y, optimizer):
        self.network.train()
        self.network.zero_grad()
        P = self.network(batch_X)
        train_loss = compute_loss(P, batch_Y)
        train_loss.backward()
        optimizer.step()
        self.project()
        self.train_loss = float(train_loss)

    def pull_towards(self, other_model, other_weight):
        self_params = list(self.network.parameters())
        other_params = list(other_model.parameters())
        assert len(self_params) == len(other_params)
        for ps, po in zip(self_params, other_params):
            ps.data = (1 - other_weight) * ps.data + other_weight * po.data

    def forward(self, X):
        self.network.eval()
        return self.network(X)

fast_model = Model()
avg_model = copy.deepcopy(fast_model)

batch_size = 2048
lr0 = 0.01
lr1 = 0.0005
beta0 = 0.99
beta1 = 0.99
reaper_factor0 = 0.05
reaper_factor1 = 0.05
scale_decay = 0.05

optimizer = GroupedAdam(fast_model.network.parameters(), lr=lr0, betas=(beta0, beta0), eps=1e-15)

logging.info(fast_model)
logging.info(optimizer)

avg_pull_weight = 0.005
fast_pull_weight = 0.005
iteration = 1

cumulative_preds = []
capture_frac = 0.2
max_capture = 20

fast_model.project()
avg_model.project()
for _ in range(1, 50001):
    frac = min(iteration / 8000, 1.0)
    lr = lr0 * (1 - frac) + lr1 * frac
    beta = beta0 * (1 - frac) + beta1 * frac
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[0]['betas'] = (beta, beta)

    batch_ind = torch.randint(0, train_X.shape[0], [batch_size])
    batch_X = preprocessor.augment(train_X[batch_ind, :])
    # batch_X = train_X[batch_ind, :]
    batch_Y = train_Y[batch_ind]

    fast_model.train_step(batch_X, batch_Y, optimizer)

    reaper_factor = frac * reaper_factor1 + (1 - frac) * reaper_factor0
    with torch.no_grad():
        avg_model.pull_towards(fast_model, avg_pull_weight)
        for mod in avg_model.network.modules():
            if isinstance(mod, L1Linear):
                mod.weights_pos_neg.param *= (1 + reaper_factor * lr)
        avg_model.network.scale.data *= 1 - scale_decay * lr
        avg_model.project()
        fast_model.pull_towards(avg_model, fast_pull_weight)

    if iteration % 500 == 0:
        with torch.no_grad():
            test_P = avg_model(test_X)

            if len(cumulative_preds) == 0:
                cumulative_preds.append(torch.zeros_like(test_P))
            cumulative_preds.append(test_P + cumulative_preds[-1])
            n_capture = min(int(math.ceil((len(cumulative_preds) - 1) * capture_frac)), max_capture)
            # test_P = (cumulative_preds[-1] - cumulative_preds[-1 - n_capture]) / n_capture

            test_loss = compute_loss(test_P, test_Y)
            test_acc = compute_accuracy(test_P, test_Y)

            act_avgs = []
            act_abs_avgs = []
            act_nz_avgs = []
            for layer in avg_model.network.act_layers:
                act_avgs.append(torch.mean(layer.out))
                act_abs_avgs.append(torch.mean(torch.abs(layer.out)))
                act_nz_avgs.append(torch.mean((torch.abs(layer.out) > 1e-5).to(torch.float32)))
            act_avgs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_avgs))
            act_abs_avgs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_abs_avgs))
            act_nz_avgs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_nz_avgs))

            wt_fracs = []
            for layer in avg_model.network.lin_layers:
                weights = layer.weights_pos_neg.param[:layer.input_width, :] - layer.weights_pos_neg.param[
                                                                               layer.input_width:, :]
                wt_fracs.append(torch.sum(weights != 0).to(torch.float32) / weights.shape[1])
            wt_fracs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in wt_fracs))

            # # scales = []
            # # for layer in networks[0].lin_layers:
            # #     scales.append(torch.mean(torch.abs(layer.scale) * layer.scale_factor))
            # # scales_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in scales))
            scales_fmt = '{:.3f}'.format(avg_model.network.scale * avg_model.network.scale_factor)

            lr = optimizer.param_groups[0]['lr']
            beta = optimizer.param_groups[0]['betas'][0]
            logging.info(
                "{}: lr={:.4f}, beta={:.4f}, train={:.6f}, test={:.6f}, acc={:.6f}, sc={}, wt={}, act={}, anz={}".format(
                    iteration,
                    lr,
                    beta,
                    fast_model.train_loss / batch_X.shape[0],
                    float(test_loss / test_X.shape[0]),
                    float(test_acc),
                    scales_fmt,
                    wt_fracs_fmt, act_abs_avgs_fmt, act_nz_avgs_fmt))
    iteration += 1

torch.set_printoptions(linewidth=120)
print(sklearn.metrics.confusion_matrix(test_Y, torch.argmax(test_P, dim=1)))
