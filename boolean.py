import torch
import logging
from tame_pytorch import ManifoldModule, Network, L1Linear
from grouped_adam import GroupedAdam
import math


def all_boolean_vectors(n, dtype=torch.float32):
    if n == 0:
        return torch.zeros([1, 0], dtype=dtype)
    else:
        A = all_boolean_vectors(n - 1, dtype=dtype)
        return torch.cat([
            torch.cat([A, torch.full([A.shape[0], 1], 0, dtype=dtype)], dim=1),
            torch.cat([A, torch.full([A.shape[0], 1], 1, dtype=dtype)], dim=1),
        ])


def compute_loss(P, Y):
    return torch.sum((P - Y) ** 2)


def xor_train_set(num_inputs):
    X = all_boolean_vectors(num_inputs, dtype=torch.int)
    Y = (torch.sum(X, dim=1) % 2).to(torch.float32).view(-1, 1)
    return (X * 2 - 1).to(torch.float32), Y * 2 - 1


def sum_eq_train_set(num_inputs, k):
    X = all_boolean_vectors(num_inputs, dtype=torch.int)
    Y1 = torch.sum(X[:, :k], dim=1)
    Y2 = torch.sum(X[:, k:], dim=1)
    Y = (Y1 == Y2).to(torch.float32).view(-1, 1)
    return (X * 2 - 1).to(torch.float32), Y * 2 - 1

def from_bits(X):
    return torch.sum(X * 2 ** torch.arange(X.shape[0]))

def to_bits(n, num_bits):
    return [(n // 2 ** i) % 2 for i in range(num_bits)]

def adder_train_set(num_inputs):
    assert num_inputs % 2 == 0
    n = num_inputs // 2
    X = all_boolean_vectors(num_inputs, dtype=torch.int)
    out = []
    for i in range(X.shape[0]):
        inp1 = from_bits(X[i, :n])
        inp2 = from_bits(X[i, n:])
        out.append(to_bits(inp1 + inp2, n))
    Y = torch.tensor(out)
    return (X * 2 - 1).to(torch.float32), Y * 2 - 1


def rand_train_set(num_inputs):
    X = all_boolean_vectors(num_inputs, dtype=torch.float32)
    Y = torch.randint(0, 2, [X.shape[0], 1])
    return X * 2 - 1, (Y * 2 - 1).to(torch.float32)


num_inputs = 12
all_X, all_Y = adder_train_set(num_inputs)
# all_X, all_Y = prime_train_set(num_inputs)
# all_X, all_Y = sum_eq_train_set(num_inputs, 5)
# all_X, all_Y = xor_train_set(num_inputs)
# train_X, train_Y = rand_train_set(num_inputs)

torch.random.manual_seed(0)
train_mask = torch.rand([all_X.shape[0]]) < 0.25
torch.random.seed()
train_X = all_X[train_mask, :]
train_Y = all_Y[train_mask, :]
test_X = all_X[~train_mask, :]
test_Y = all_Y[~train_mask, :]

# train_X = all_X  # No separate test set here since we're not trying to generalize
# train_Y = all_Y
# test_X = all_X
# test_Y = all_Y
# # print(train_Y)

ensemble_size = 1
networks = [Network(widths=[num_inputs] + [128, 64, 32] + [train_Y.shape[1]],
                    pen_lin_coef=0.0,
                    pen_lin_exp=2.0,
                    pen_scale=0.0,
                    arity=2,
                    scale_init=1.0,
                    scale_factor=0.3,
                    dtype=torch.float32,
                    device=torch.device('cpu'))
            for _ in range(ensemble_size)]


reaper_factor0 = 0.0
repear_factor1 = 0.05
lr0 = 0.12
lr1 = 0.06
beta0 = 0.998
beta1 = 0.999
pen_act0 = 0.0
pen_act1 = 0.0
pen_lin_coef0 = 0.0
pen_lin_coef1 = 0.0
pen_scale_coef0 = 0.0
pen_scale_coef1 = 0.0
# optimizers = [torch.optim.Adam(networks[i].parameters(), lr=lr0, betas=(0.99, 0.99), eps=1e-15)
#               for i in range(ensemble_size)]
optimizers = [GroupedAdam(networks[i].parameters(), lr=lr0, betas=(beta0, beta0), eps=1e-15)
              for i in range(ensemble_size)]
# optimizers = [torch.optim.SGD(networks[i].parameters(), lr=lr0, momentum=0.9)
#               for i in range(ensemble_size)]

average_params = [[torch.zeros_like(p) for p in net.parameters()] for net in networks]
# average_batch_norm_running_mean = [[torch.zeros_like(b._buffers['running_mean']) for b in net.bn_layers] for net in networks]
# average_batch_norm_running_var = [[torch.zeros_like(b._buffers['running_var']) for b in net.bn_layers] for net in networks]
average_param_beta = 0.98
average_param_weight = 0.0
epoch = 1

with torch.no_grad():
    for net in networks:
        for mod in net.modules():
            if isinstance(mod, ManifoldModule):
                mod.project()
for _ in range(1, 10001):
    frac = min(epoch / 3000, 1.0)
    for net in networks:
        for layer in net.lin_layers:
            layer.pen_coef = frac * pen_lin_coef1 + (1 - frac) * pen_lin_coef0
            layer.pen_scale_coef = frac * pen_scale_coef1 + (1 - frac) * pen_scale_coef0

    total_loss = 0.0
    total_obj = 0.0
    for j, net in enumerate(networks):
        net.train()

        net.zero_grad()
        P = net(train_X)
        train_loss = compute_loss(P, train_Y)
        obj = train_loss + net.penalty()
        obj.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1e-5)

        lr = lr0 * (1 - frac) + lr1 * frac
        beta = beta0 * (1 - frac) + beta1 * frac
        reaper_factor = frac * repear_factor1 + (1 - frac) * reaper_factor0
        with torch.no_grad():
            for mod in net.modules():
                if isinstance(mod, L1Linear):
                    # noise = torch.rand_like(mod.weights_pos_neg.param)
                    # mod.weights_pos_neg.param *= (1 + noise_factor * noise)
                    mod.weights_pos_neg.param *= (1 + reaper_factor * lr)

        optimizers[j].param_groups[0]['lr'] = lr
        optimizers[j].param_groups[0]['betas'] = (beta, beta)
        optimizers[j].step()
        total_loss += train_loss
        total_obj += obj
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
            test_acc = torch.mean(torch.all(torch.sgn(test_P) == torch.sgn(test_Y), dim=1).to(torch.float32))

            for j, net in enumerate(networks):
                for i, p in enumerate(net.parameters()):
                    p.data = saved_params[j][i]

            # act_fracs = []
            # for layer in networks[0].act_layers:
            #     act_fracs.append(torch.mean((layer.X >= 0).to(torch.float32)))
            # act_fracs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_fracs))
            #
            # act_avgs = []
            # for layer in networks[0].act_layers:
            #     act_avgs.append(torch.mean(torch.clamp(layer.X, min=0.0)))
            # act_avgs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in act_avgs))

            wt_fracs = []
            for layer in networks[0].lin_layers:
                wt_fracs.append(torch.sum(layer.weights_pos_neg.param > 1e-8).to(torch.float32) / layer.weights_pos_neg.param.shape[1])
                # weights = layer.weight
                # wt_fracs.append(torch.mean((weights != 0).to(torch.float32)))
            wt_fracs_fmt = '[{}]'.format(', '.join('{:.3f}'.format(f) for f in wt_fracs))

            lr = optimizers[0].param_groups[0]['lr']
            # lr = optimizers[0].param_groups[0]['lr']
            logging.info(
                "{}: lr={:.6f}, train={:.6f}, obj={:.6f}, test={:.6f}, acc={:.6f}, wt={}, scale={:.3f}".format(
                    epoch,
                    lr, total_loss / ensemble_size / train_X.shape[0],
                        total_obj / ensemble_size / train_X.shape[0],
                    float(test_loss / test_X.shape[0]),
                    float(test_acc),
                    wt_fracs_fmt, torch.max(torch.abs(networks[0].scale) * networks[0].scale_factor).item()))
    epoch += 1
