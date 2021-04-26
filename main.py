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
from tame_pytorch import ManifoldModule, L1Linear, LpLinear, L1LinearScaled
from tame_pytorch import Network as TameNetwork, MultiGateNetwork, MultiplexerNetwork, SparseHighOrderActivationB, D2Activation, TransformerNetwork
from tame_pytorch import approx_simplex_projection

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("pokerhand.log"),
                              logging.StreamHandler()])



def compute_loss(P, Y, weight=None):
    # return torch.nn.functional.cross_entropy(P, Y, weight=weight, reduction='sum')
    n = P.shape[0]
    # P = approx_simplex_projection(P, dim=1, num_iters=6)
    PY = P[torch.arange(n), Y]
    # loss = torch.sum(torch.abs(P)) + torch.sum(torch.abs(PY - 1.0)) - torch.sum(torch.abs(PY))
    P0 = torch.clamp(P, min=0.0)
    PY0 = torch.clamp(PY, min=0.0)
    PY1 = torch.clamp(PY, max=1.0)
    loss = torch.sum(P0 ** 2) + torch.sum((1.0 - PY1) ** 2) - torch.sum(PY0 ** 2)
    # loss = torch.sum(P ** 2) + torch.sum((1.0 - PY) ** 2) - torch.sum(PY ** 2)
    # loss = torch.sum(P0) + torch.sum(1.0 - PY1) - torch.sum(PY0)
    return loss


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


# class Preprocessor():
#     def fit(self, X):
#         suit = X[:, 0::2]
#         rank = X[:, 1::2]
#         self.encoder = sklearn.preprocessing.OneHotEncoder()
#         encoded_suit = self.encoder.fit_transform(suit.numpy()).todense()
#         # self.scaler = sklearn.preprocessing.StandardScaler()
#         # self.scaler.fit(rank.reshape(-1, 1).repeat([1, 5]))
#
#     def transform(self, X):
#         # return self.scaler.transform(X)
#         suit = X[:, 0::2]
#         rank = X[:, 1::2]
#         encoded_suit = self.encoder.transform(suit).todense()
#         # scaled_rank = self.scaler.transform(rank)
#         scaled_rank = rank
#         out = np.concatenate([encoded_suit, scaled_rank], axis=1)
#         return out
#         # unscaled = np.concatenate([encoded_suit, rank.numpy()], axis=1)
#         # scaled = self.scaler.transform(unscaled)
#         # return scaled
#         # scaled_rank = self.scaler.transform(rank)
#         # return np.concatenate([encoded_suit, scaled_rank], axis=1)
#
#     def augment(self, X):
#         X = torch.clone(X)
#
#         # Shuffle the cards
#         card_perm = torch.randperm(5)
#         X[:, -5:] = X[:, -5 + card_perm]  # Shuffle their ranks
#         for suit in range(4):  # Shuffle their encoded suits
#             X[:, suit:-5:4] = X[:, suit + card_perm * 4]
#
#         # Shuffle the suits
#         suit_perm = torch.randperm(4)
#         for card in range(5):
#             X[:, (card * 4):((card + 1) * 4)] = X[:, card * 4 + suit_perm]
#         return X



class Preprocessor():
    def fit(self, X):
        pass

    def transform(self, X):
        return X.numpy()

    def augment(self, X):
        X = torch.clone(X)

        # Shuffle the cards
        card_perm = torch.randperm(5)
        X[:, ::2] = X[:, card_perm * 2]        # Shuffle their suits
        X[:, 1::2] = X[:, card_perm * 2 + 1]   # Shuffle their ranks

        # Shuffle the suits
        suit_perm = torch.randperm(4)
        for card in range(5):
            X[:, ::2] = suit_perm[(X[:, ::2] - 1).to(torch.long)].to(torch.float32) + 1.0

        return X


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

# Split up training set by label (to make balanced sampling easier)
train_X_splits = []
train_Y_splits = []
for i in reversed(range(10)):
    mask = train_Y == i
    train_X_splits.append(train_X[mask, :])
    train_Y_splits.append(train_Y[mask])  # These will just be constant tensors


def sample_batch(n, X_splits, Y_splits, balancing_exp):
    sizes = torch.tensor([Y.shape[0] for Y in Y_splits], dtype=torch.float32)
    sample_rate = sizes ** balancing_exp
    X_list = []
    Y_list = []
    n_remaining = n
    for i in range(len(X_splits)):
        ni = int(math.ceil(n_remaining * (sample_rate[i] / torch.sum(sample_rate[i:]))))
        ind = torch.randint(0, train_X_splits[i].shape[0], [ni])
        # X = preprocessor.augment(train_X_splits[i][ind, :])
        X = train_X_splits[i][ind, :]
        Y = train_Y_splits[i][ind]
        X_list.append(X)
        Y_list.append(Y)
        n_remaining -= ni
    weight = torch.tensor([X_splits[i].shape[0] / X_list[i].shape[0] for i in reversed(range(len(X_splits)))])
    weight = weight * (n / sum(X.shape[0] for X in X_splits))
    return torch.cat(X_list, dim=0), torch.cat(Y_list, dim=0), weight

# batch_X = train_X[100:101, :]
# print(raw_train_X[:, 100])
# print(batch_X)
# print(preprocessor.augment(batch_X))
#

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.network = TameNetwork(widths=[train_X.shape[1]] + [16, 16, 16] + [10],
        #             bias_factor=1.0,
        #             act_init=1.0,
        #             act_factor=1.0,
        #             scale_init=1.0,
        #             scale_factor=1e-20,
        #             arity=3,
        #             # dropout_p=0.2,
        #             dtype=torch.float32,
        #             device=torch.device('cpu'))
        self.network = TransformerNetwork(
            widths=[2, 12, 12, 10],
            arity=4,
            dtype=torch.float32,
            device=torch.device('cpu'))


        # _, Y_cnt = torch.unique(train_Y, return_counts=True)
        # Y_p = Y_cnt.to(torch.float32) / torch.sum(Y_cnt).to(torch.float32)
        # Y_log_p = torch.log(Y_p)
        # self.network.bias.data[:] = Y_log_p

    def project(self):
        self.network.project()
        # for mod in self.network.modules():
        #     if isinstance(mod, ManifoldModule):
        #         mod.project()

    def contract(self, reaper_factor, act_decay, scale_decay):
        with torch.no_grad():
            if reaper_factor != 0.0:
                for mod in self.network.modules():
                    if isinstance(mod, (L1Linear, LpLinear, L1LinearScaled)):
                        mod.weights_pos_neg.param *= (1 + reaper_factor)
            if act_decay != 0.0:
                for layer in self.network.act_layers:
                    # layer.params.data *= 1 - act_decay
                    d = layer.params.data
                    layer.params.data = torch.sgn(d) * torch.clamp(torch.abs(d) - act_decay, min=0.0)
            if scale_decay != 0.0:
            #     for layer in self.network.scale_layers:
            #         d = layer.scale.data
            #         layer.scale.data = torch.sgn(d) * torch.clamp(torch.abs(d) - scale_decay, min=0.0)
                for layer in self.network.lin_layers:
                    layer.scale.data *= 1 - scale_decay
                # self.network.scale.data *= 1 - scale_decay

    def train_step(self, batch_X, batch_Y, batch_w, optimizer):
        self.network.train()
        self.network.zero_grad()
        P = self.network(batch_X)
        train_loss = compute_loss(P, batch_Y, weight=batch_w)
        train_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1e-5)
        optimizer.step()
        self.train_loss = float(train_loss)

    def pull_towards(self, other_models, other_weight):
        self_params = list(self.network.parameters())
        other_params = [list(model.parameters()) for model in other_models]
        assert all(len(p) == len(self_params) for p in other_params)
        for i in range(len(self_params)):
            avg_params = sum(p[i] for p in other_params) / len(other_params)
            self_params[i].data = (1 - other_weight) * self_params[i].data + other_weight * avg_params

    def set_average(self, other_models):
        self_params = list(self.network.parameters())
        other_params = [list(model.parameters()) for model in other_models]
        assert all(len(p) == len(self_params) for p in other_params)
        for i in range(len(self_params)):
            self_params[i].data = sum(p[i] for p in other_params) / len(other_params)

    def copy(self, other_model):
        self_params = list(self.network.parameters())
        other_params = list(other_model.parameters())
        assert len(self_params) == len(other_params)
        for ps, po in zip(self_params, other_params):
            ps.data.copy_(po.data)

    def accumulate(self, other_models, weight):
        self_params = list(self.network.parameters())
        other_params = [list(model.parameters()) for model in other_models]
        assert all(len(p) == len(self_params) for p in other_params)
        for i in range(len(self_params)):
            self_params[i].data += sum(p[i] for p in other_params) * weight

    def zero(self):
        for param in self.network.parameters():
            param.data.zero_()

    def forward(self, X):
        self.network.eval()
        return self.network(X)

num_fast_models = 1
batch_size = 2048
eval_batch_size = 8192
balancing_exp = 0.5
lr0 = 0.015
lr1 = 0.003
p0 = 1.0
p1 = 1.0
lin_eps0 = 0.1
lin_eps1 = 0.1
beta0 = 0.99
beta1 = 0.99
reaper_factor0 = 0.0
reaper_factor1 = 0.0
scale_decay0 = 0.0
scale_decay1 = 0.0
act_decay0 = 0.0
act_decay1 = 0.0
sync_frequency = 500
eval_frequency = 1
# unstick_frequency = 5
# avg_pull_weight = 0.1
# fast_pull_weight = 0.1

init_model = Model()
fast_models = [copy.deepcopy(init_model) for _ in range(num_fast_models)]
avg_model = copy.deepcopy(init_model)
eval_model = copy.deepcopy(init_model)
# avg_model.zero()

# optimizers = [GroupedAdam(model.network.parameters(), lr=lr0, betas=(beta0, beta0), eps=1e-15) for model in fast_models]
optimizers = [torch.optim.Adam(model.network.parameters(), lr=lr0, betas=(beta0, beta0), eps=1e-15) for model in fast_models]
# optimizers = [torch.optim.SGD(model.network.parameters(), lr=lr0, momentum=0.99) for model in fast_models]

logging.info(init_model)
logging.info(optimizers[0])

iteration = 1

cumulative_preds = []
capture_frac = 0.2
max_capture = 20

def do_eval(model):
    with torch.no_grad():
        test_P_list = []
        num_batches = (test_X.shape[0] + eval_batch_size - 1) // eval_batch_size
        for i in range(num_batches):
            batch_X = test_X[(i * eval_batch_size):((i + 1) * eval_batch_size), :]
            P = model(batch_X)
            test_P_list.append(P)
        test_P = torch.cat(test_P_list, dim=0)
        # test_P = model(test_X)

        if len(cumulative_preds) == 0:
            cumulative_preds.append(torch.zeros_like(test_P))
        cumulative_preds.append(test_P + cumulative_preds[-1])
        # n_capture = min(int(math.ceil((len(cumulative_preds) - 1) * capture_frac)), max_capture)
        # test_P = (cumulative_preds[-1] - cumulative_preds[-1 - n_capture]) / n_capture

        test_loss = compute_loss(test_P, test_Y)
        test_acc = compute_accuracy(test_P, test_Y)

        act_avgs = []
        act_ranges = []
        act_abs_avgs = []
        act_nz_avgs = []
        act_activity_max = []
        for tlayer in model.network.transformer_layers:
        # for layer in model.network.act_layers:
        # for layer in model.network.scale_layers:
            layer = tlayer.act_layer
            act_ranges.append(torch.max(torch.max(layer.out, dim=0, keepdim=True)[0] - torch.min(layer.out, dim=0, keepdim=True)[0]))
            med = torch.median(layer.out, dim=0, keepdim=True)[0]
            act_abs_avgs.append(torch.mean(torch.abs(layer.out - med)))
            act_nz_avgs.append(torch.mean((torch.abs(layer.out - med) > 1e-5).to(torch.float32)))
            # act_activity_max.append(torch.max(layer._compute_activity()) * layer.act_factor)
        act_ranges_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_ranges))
        act_avgs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_avgs))
        act_abs_avgs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_abs_avgs))
        act_nz_avgs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_nz_avgs))
        # act_activity_max_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_activity_max))



        wt_fracs = []
        for tlayer in model.network.transformer_layers:
        # for layer in model.network.lin_layers:
            layer = tlayer.lin_layer
            weights = layer.weights_pos_neg.param[:layer.input_width, :] - layer.weights_pos_neg.param[
                                                                           layer.input_width:, :]
            # weights = layer.weights.param
            wt_fracs.append(torch.sum(weights != 0).to(torch.float32) / weights.shape[1])
        wt_fracs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in wt_fracs))

        # # scales = []
        # # for layer in networks[0].lin_layers:
        # #     scales.append(torch.mean(torch.abs(layer.scale) * layer.scale_factor))
        # # scales_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in scales))
        # scales_fmt = '{:.3f}'.format(model.network.scale * model.network.scale_factor)

        lr = optimizers[0].param_groups[0]['lr']
        beta = optimizers[0].param_groups[0]['betas'][0]
        logging.info(
            "{}: train={:.6f}, test={:.6f}, acc={:.6f}, wt={}, range={}, avg={}, nz={}".format(
                iteration,
                sum(fast_models[i].train_loss for i in range(num_fast_models)) / num_fast_models / batch_X.shape[0],
                float(test_loss / test_X.shape[0]),
                float(test_acc),
                # scales_fmt,
                wt_fracs_fmt, act_ranges_fmt, act_abs_avgs_fmt, act_nz_avgs_fmt))
        # logging.info(
        #     "{}: train={:.6f}, test={:.6f}, acc={:.6f}, sc={}, act={}".format(
        #         iteration,
        #         sum(fast_models[i].train_loss for i in range(num_fast_models)) / num_fast_models / batch_X.shape[0],
        #         float(test_loss / test_X.shape[0]),
        #         float(test_acc),
        #         scales_fmt,
        #         act_abs_avgs_fmt))
    return test_P

# for layer in model.network.dropout_layers:
#     layer.p = 0.01

for model in fast_models:
    model.project()
avg_model.zero()
eval_model.zero()
avg_model_ctr = 0
eval_ctr = 0
unstick_ctr = 0
for _ in range(1, 50001):
    frac = min(iteration / 5000, 1.0)
    lr = lr0 * (1 - frac) + lr1 * frac
    beta = beta0 * (1 - frac) + beta1 * frac
    reaper_factor = frac * reaper_factor1 + (1 - frac) * reaper_factor0
    scale_decay = frac * scale_decay1 + (1 - frac) * scale_decay0
    act_decay = frac * act_decay1 + (1 - frac) * act_decay0
    p = frac * p1 + (1 - frac) * p0
    lin_eps = frac * lin_eps1 + (1 - frac) * lin_eps0

    for i, model in enumerate(fast_models):
        optimizers[i].param_groups[0]['lr'] = lr
        optimizers[i].param_groups[0]['betas'] = (beta, beta)
        # # for layer in model.network.act_layers:
        # #     layer.activity_lr = lr * 3.0
        # for layer in model.network.lin_layers:
        #     # layer.weights.p = p
        #     # layer.weights.eps = lin_eps
        #     layer.weights_pos_neg.p = p
        #     layer.weights_pos_neg.eps = lin_eps
        # #     layer.weights_pos_neg.num_iters = 12

        # batch_ind = torch.randint(0, train_X.shape[0], [batch_size])
        # batch_X = preprocessor.augment(train_X[batch_ind, :])
        # # batch_X = train_X[batch_ind, :]
        # batch_Y = train_Y[batch_ind]
        # batch_w = None

        batch_X, batch_Y, batch_w = sample_batch(batch_size, train_X_splits, train_Y_splits, balancing_exp)

        model.train_step(batch_X, batch_Y, batch_w, optimizers[i])
        model.contract(reaper_factor * lr, act_decay * lr, scale_decay * lr)
        model.project()

    with torch.no_grad():
        # avg_model.pull_towards(fast_models, avg_pull_weight)
        # # avg_model.contract(reaper_factor * lr, act_decay * lr, scale_decay * lr)
        # # avg_model.project()
        # for model in fast_models:
        #     model.pull_towards([avg_model], fast_pull_weight)
        # eval_ctr += 1
        # if eval_ctr == eval_frequency:
        #     do_eval()
        #     eval_ctr = 0

        avg_model.accumulate(fast_models, weight=1 / sync_frequency / len(fast_models))
        avg_model_ctr += 1
        if avg_model_ctr == sync_frequency:
            avg_model_ctr = 0
            # avg_model.project()
            # for model in fast_models:
            #     model.copy(avg_model)
            eval_model.accumulate([avg_model], weight=1 / eval_frequency)
            eval_ctr += 1
            if eval_ctr == eval_frequency:
                test_P = do_eval(eval_model)
                eval_ctr = 0
                eval_model.zero()
                # unstick_ctr += 1
                # if unstick_ctr == unstick_frequency:
                #     for layer in model.network.lin_layers:
                #         layer.unstick()
            avg_model.zero()

    iteration += 1

torch.set_printoptions(linewidth=120)
print(sklearn.metrics.confusion_matrix(test_Y, torch.argmax(test_P, dim=1)))
