"""Components for building neural networks that are "tame", i.e. Lipschitz-continuous with respect to the max norm
with Lipschitz constant 1. Such a component has the property that a simultaneous change of at most `epsilon` to its
inputs produces a change of at most `epsilon` to all its outputs. These could be used, for example, as a building
block for an RNN without risk of gradient explosion."""

import torch
from typing import List
import logging
import math
from high_order_act_pytorch import HighOrderActivation, HighOrderActivationB, high_order_act_b, cartesian_power

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
    def pre_step(self):
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
    def __init__(self, input_width, output_width,
                 pen_coef=0.0, pen_exp=2.0,
                 bias_factor=1.0,
                 # scale_factor=1.0,
                 noise_factor=0.0,
                 dtype=torch.float32, device=None,
                 num_iters=6):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.pen_coef = pen_coef
        self.pen_exp = pen_exp
        self.bias_factor = bias_factor
        self.noise_factor = noise_factor
        # self.scale_factor = scale_factor
        self.weights_pos_neg = Simplex([input_width * 2, output_width], dim=0, dtype=dtype, device=device,
                                       num_iters=num_iters)

        # self.active_input = torch.randint(0, input_width * 2, [output_width])
        # self.weights_pos_neg.param.data[:, :] = 0.0
        # self.weights_pos_neg.param.data[self.active_input, torch.arange(output_width)] = 1.0
        self.bias = torch.nn.Parameter(torch.zeros([output_width], dtype=dtype, device=device))
        # self.scale = torch.nn.Parameter(torch.ones([output_width], dtype=dtype, device=device))

    # def forward(self, X, max_scale):
    def forward(self, X):
        assert X.shape[1] == self.input_width
        self.cnt_rows = X.shape[0]
        raw_weights = self.weights_pos_neg.param[:self.input_width, :] - self.weights_pos_neg.param[self.input_width:,
                                                                         :]
        if self.training and self.noise_factor != 0.0:
            weights = raw_weights * (1 + torch.rand_like(raw_weights) * self.noise_factor)
        else:
            weights = raw_weights
        return torch.matmul(X, weights) + self.bias.view(1, -1) * self.bias_factor
        # return torch.matmul(X, weights) * (self.scale.view(1, -1) * self.scale_factor) + self.bias.view(1, -1) * self.bias_factor
        # scale = torch.maximum(torch.minimum(self.scale, max_scale), -max_scale)
        # return torch.matmul(X, weights) * scale + self.bias.view(1, -1) * self.bias_factor

    def penalty(self):
        w = self.weights_pos_neg.param
        pen_weight = self.pen_coef * self.cnt_rows * torch.mean(1 - (1 - w) ** self.pen_exp - w)
        return pen_weight


def concave_projection_iteration(x0: torch.tensor, y: torch.tensor, K: float, cost_fn, cost_grad, dim):
    mask = (x0 > 0)
    c = torch.where(mask, cost_fn(x0), torch.zeros_like(x0))
    g = torch.where(mask, cost_grad(x0), torch.zeros_like(x0))
    lam = (torch.sum(c + g * (y - x0), dim=dim) - K) / torch.sum(g ** 2, dim=dim)
    x1 = torch.clamp(y - lam.unsqueeze(dim) * g, min=0.0)
    x1_mask = torch.where(mask, x1, torch.zeros_like(x1))
    return x1_mask


def concave_projection(x0: torch.tensor, y: torch.tensor, K: float, cost_fn, cost_grad, dim, num_iters):
    x = x0
    for _ in range(num_iters):
        x = concave_projection_iteration(x, y, K, cost_fn, cost_grad, dim)
    return x


def soft_lp_cost_grad(eps, p):
    c = (1 + eps) ** p - eps ** p
    cost = lambda x: ((x + eps) ** p - eps ** p) / c
    grad = lambda x: p * (x + eps) ** (p - 1) / c
    return cost, grad


def lp_projection(y: torch.tensor, K: float, p: float, eps: float, dim: int, num_iters: int):
    x0 = torch.full_like(y, 1e-15)
    cost_fn, cost_grad = soft_lp_cost_grad(eps, p)
    return concave_projection(x0, y, K, cost_fn, cost_grad, dim, num_iters)


class LpSimplex(ManifoldModule):
    def __init__(self, shape: List[int], dim: int, p: float, eps: float, dtype=torch.float32, device=None, num_iters=8):
        super().__init__()
        self.shape = shape
        self.dim = dim
        self.p = p
        self.eps = eps
        self.num_iters = num_iters
        data = torch.rand(shape, dtype=dtype, device=device)
        data = torch.log(data + 1e-15)
        data = data / torch.sum(data, dim=dim).unsqueeze(dim)
        self.param = torch.nn.Parameter(data)
        self.project()

    def project(self):
        with torch.no_grad():
            self.param.data = lp_projection(self.param.data, K=1.0, p=self.p, eps=self.eps, dim=self.dim, num_iters=self.num_iters)


class LpLinear(torch.nn.Module):
    def __init__(self, input_width, output_width,
                 pen_coef=0.0, pen_exp=2.0,
                 bias_factor=1.0,
                 # scale_factor=1.0,
                 noise_factor=0.0,
                 eps=0.1,
                 p=1.0,
                 dtype=torch.float32, device=None,
                 num_iters=6):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.pen_coef = pen_coef
        self.pen_exp = pen_exp
        self.bias_factor = bias_factor
        self.noise_factor = noise_factor
        self.p = p
        self.eps = eps
        # self.scale_factor = scale_factor
        self.weights_pos_neg = LpSimplex([input_width * 2, output_width], dim=0, p=p, eps=eps,
                                         dtype=dtype, device=device,
                                          num_iters=num_iters)

        # self.active_input = torch.randint(0, input_width * 2, [output_width])
        # self.weights_pos_neg.param.data[:, :] = 0.0
        # self.weights_pos_neg.param.data[self.active_input, torch.arange(output_width)] = 1.0
        self.bias = torch.nn.Parameter(torch.zeros([output_width], dtype=dtype, device=device))
        # self.scale = torch.nn.Parameter(torch.ones([output_width], dtype=dtype, device=device))

    # def forward(self, X, max_scale):
    def forward(self, X):
        assert X.shape[1] == self.input_width
        weights = self.weights_pos_neg.param[:self.input_width, :] - self.weights_pos_neg.param[self.input_width:, :]
        return torch.matmul(X, weights) + self.bias.view(1, -1) * self.bias_factor

    def penalty(self):
        return 0.0

class L1LinearScaled(torch.nn.Module):
    def __init__(self, input_width, output_width,
                 pen_coef=0.0, pen_exp=2.0,
                 bias_factor=1.0,
                 noise_factor=0.0,
                 dtype=torch.float32, device=None,
                 num_iters=6):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.pen_coef = pen_coef
        self.pen_exp = pen_exp
        self.bias_factor = bias_factor
        self.noise_factor = noise_factor
        self.weights_pos_neg = Simplex([input_width * 2, output_width], dim=0, dtype=dtype, device=device,
                                       num_iters=num_iters)

        # self.active_input = torch.randint(0, input_width * 2, [output_width])
        # self.weights_pos_neg.param.data[:, :] = 0.0
        # self.weights_pos_neg.param.data[self.active_input, torch.arange(output_width)] = 1.0
        self.bias = torch.nn.Parameter(torch.zeros([output_width], dtype=dtype, device=device))
        self.scale = torch.nn.Parameter(torch.ones([output_width], dtype=dtype, device=device))

    def forward(self, X, max_scale):
        assert X.shape[1] == self.input_width
        self.cnt_rows = X.shape[0]
        raw_weights = self.weights_pos_neg.param[:self.input_width, :] - self.weights_pos_neg.param[self.input_width:,
                                                                         :]
        if self.training and self.noise_factor != 0.0:
            weights = raw_weights * (1 + torch.rand_like(raw_weights) * self.noise_factor)
        else:
            weights = raw_weights
        scale = torch.maximum(torch.minimum(self.scale, max_scale), -max_scale)
        return torch.matmul(X, weights) * scale + self.bias.view(1, -1) * self.bias_factor

    def penalty(self):
        w = self.weights_pos_neg.param
        pen_weight = self.pen_coef * self.cnt_rows * torch.mean(1 - (1 - w) ** self.pen_exp - w)
        return pen_weight


class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        out = torch.clamp(X, min=0.0)
        self.out = out
        return out

    def penalty(self):
        return 0.0


class PReLU(ManifoldModule):
    def __init__(self, num_inputs, dtype=torch.float32, device=None):
        super().__init__()
        self.num_inputs = num_inputs
        self.slope_left = torch.nn.Parameter(torch.zeros([num_inputs], dtype=dtype, device=device))
        self.slope_right = torch.nn.Parameter(torch.ones([num_inputs], dtype=dtype, device=device))

    def forward(self, X):
        out = self.slope_right * torch.clamp(X, min=0.0) - self.slope_left * torch.clamp(X, max=0.0)
        self.out = out
        return out

    def project(self):
        self.slope_left.data = torch.clamp(self.slope_left.data, min=-1.0, max=1.0)
        self.slope_right.data = torch.clamp(self.slope_right.data, min=-1.0, max=1.0)
        # pass

    def penalty(self):
        return 0.0


class MinOut(torch.nn.Module):
    def __init__(self, arity):
        super().__init__()
        self.arity = arity

    def forward(self, X):
        self.X = X
        X = X.view([X.shape[0], self.arity, -1])
        out = torch.min(X, dim=1)[0]
        self.out = out
        return out

    def penalty(self):
        return 0.0


class ClampedMinOut(torch.nn.Module):
    def __init__(self, arity, target_act, pen_act):
        super().__init__()
        self.arity = arity
        self.target_act = target_act
        self.pen_act = pen_act

    def forward(self, X):
        self.X = X
        X = X.view([X.shape[0], self.arity, -1])
        M = torch.min(X, dim=1)[0]
        out = torch.clamp(M, min=0.0)
        self.M = M
        self.out = out
        return out

    def penalty(self):
        return self.pen_act * torch.sum(
            self.out * (1 - self.target_act) - torch.clamp(self.M, max=0.0) * self.target_act)


class ClampedMaxOut(torch.nn.Module):
    def __init__(self, arity, target_act, pen_act):
        super().__init__()
        self.arity = arity
        self.target_act = target_act
        self.pen_act = pen_act

    def forward(self, X):
        self.X = X
        X = X.view([X.shape[0], self.arity, -1])
        M = torch.max(X, dim=1)[0]
        out = torch.clamp(M, min=0.0)
        self.M = M
        self.out = out
        return out

    def penalty(self):
        return self.pen_act * torch.sum(
            self.out * (1 - self.target_act) - torch.clamp(self.M, max=0.0) * self.target_act)


class D2Activation(ManifoldModule):
    """Activation based on the D_2 root system. The lines y = x and y = -x partition R^2 into the four Weyl chambers,
    and for each 2-input unit, this activation function can be an arbitrary continuous function on R^2 which is linear
    on each of the four chambers subject to the "tame" constraint |df/dx| + |df/dy| <= 1; such a function is uniquely
    determined by its values on the four points (-1, -1), (-1, 1), (1, -1), (1, 1), which can be any values in the
    interval [-1, 1]."""

    def __init__(self, input_groups, act_factor=1.0):
        super().__init__()
        self.input_groups = input_groups
        self.act_factor = act_factor
        self.params = torch.nn.Parameter((torch.rand([input_groups, 4]) * 2 - 1) / act_factor)
        # self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 4]) * 2 - 1).to(torch.float32) / act_factor)

    def forward(self, inp):
        inp_view = inp.view(inp.shape[0], self.input_groups, 2)
        X = inp_view[:, :, 0]
        Y = inp_view[:, :, 1]
        PP = self.params[:, 0].view(1, -1)  # Value on (1, 1)
        NP = self.params[:, 1].view(1, -1)  # Value on (-1, 1)
        NN = self.params[:, 2].view(1, -1)  # Value on (-1, -1)
        PN = self.params[:, 3].view(1, -1)  # Value on (1, -1)
        out = torch.where(Y <= X,
                          torch.where(Y >= -X,
                                      (PP + PN) / 2 * X + (PP - PN) / 2 * Y,
                                      (PN - NN) / 2 * X - (PN + NN) / 2 * Y),
                          torch.where(Y >= -X,
                                      (PP - NP) / 2 * X + (PP + NP) / 2 * Y,
                                      -(NP + NN) / 2 * X + (NP - NN) / 2 * Y))
        out = out * self.act_factor
        self.out = out
        return out

    def project(self):
        self.params.data = torch.clamp(self.params.data, min=-1.0 / self.act_factor, max=1.0 / self.act_factor)
        pass

    def penalty(self):
        return 0.0


class ClampedD2Activation(ManifoldModule):
    """Clamped version of D2Activation."""

    def __init__(self, input_groups):
        super().__init__()
        self.input_groups = input_groups
        # self.params = torch.nn.Parameter(torch.rand([input_groups, 4]) * 2 - 1)
        self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 4]) * 2 - 1).to(torch.float32))

    def forward(self, inp):
        inp_view = inp.view(inp.shape[0], self.input_groups, 2)
        X = inp_view[:, :, 0]
        Y = inp_view[:, :, 1]
        PP = self.params[:, 0].view(1, -1)  # Value on (1, 1)
        NP = self.params[:, 1].view(1, -1)  # Value on (-1, 1)
        NN = self.params[:, 2].view(1, -1)  # Value on (-1, -1)
        PN = self.params[:, 3].view(1, -1)  # Value on (1, -1)
        out = torch.where(Y <= X,
                          torch.where(Y >= -X,
                                      (PP + PN) / 2 * X + (PP - PN) / 2 * Y,
                                      (PN - NN) / 2 * X - (PN + NN) / 2 * Y),
                          torch.where(Y >= -X,
                                      (PP - NP) / 2 * X + (PP + NP) / 2 * Y,
                                      -(NP + NN) / 2 * X + (NP - NN) / 2 * Y))
        out = torch.clamp(out, min=0.0)
        self.out = out
        return out

    def project(self):
        self.params.data = torch.clamp(self.params.data, min=-1.0, max=1.0)

    def penalty(self):
        return 0.0


class B2Activation(ManifoldModule):
    """Activation based on the D_2 root system. The four lines y = x, y = -x, x = 0, and y = 0 partition R^2 into the
    eight Weyl chambers, and for each 2-input unit, this activation function can be an arbitrary continuous function on
    R^2 which is linear on each of the eight chambers subject to the "tame" constraint |df/dx| + |df/dy| <= 1; such a
    function is uniquely determined by its values on the eight points (+/-1, +/-1), (+/-1, 0), (0, +/-1), subject
    to certain constraints."""

    def __init__(self, input_groups, act_factor, pen_act=0.0):
        super().__init__()
        self.input_groups = input_groups
        self.act_factor = act_factor
        self.pen_act = pen_act
        self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 8]) * 2 - 1).to(torch.float32))
        # self.old_params = None
        self.project()

    def forward(self, inp):
        inp_view = inp.view(inp.shape[0], self.input_groups, 2)
        X = inp_view[:, :, 0]
        Y = inp_view[:, :, 1]
        # if self.training and self.noise_factor != 0.0:
        #     P_noise = self.params * (1 + self.noise_factor * torch.randn_like(self.params))
        # else:
        #     P_noise = self.params
        # P = P_noise * self.act_factor
        P = self.params * self.act_factor
        P0 = P[:, 0].view(1, -1)  # Value at (1, 0)
        P1 = P[:, 1].view(1, -1)  # Value at (1, 1)
        P2 = P[:, 2].view(1, -1)  # Value at (0, 1)
        P3 = P[:, 3].view(1, -1)  # Value at (-1, 1)
        P4 = P[:, 4].view(1, -1)  # Value at (-1, 0)
        P5 = P[:, 5].view(1, -1)  # Value at (-1, -1)
        P6 = P[:, 6].view(1, -1)  # Value at (0, -1)
        P7 = P[:, 7].view(1, -1)  # Value at (1, -1)
        XP = X >= 0
        YP = Y >= 0
        XY = torch.abs(X) >= torch.abs(Y)
        out = torch.where(XP,
                          torch.where(YP,
                                      torch.where(XY,
                                                  P0 * X + (P1 - P0) * Y,
                                                  P2 * Y + (P1 - P2) * X),
                                      torch.where(XY,
                                                  P0 * X + (P0 - P7) * Y,
                                                  -P6 * Y + (P7 - P6) * X)),
                          torch.where(YP,
                                      torch.where(XY,
                                                  -P4 * X + (P3 - P4) * Y,
                                                  P2 * Y + (P2 - P3) * X),
                                      torch.where(XY,
                                                  -P4 * X + (P4 - P5) * Y,
                                                  -P6 * Y + (P6 - P5) * X)))
        # if self.training and self.noise_factor != 0.0:
        #     out = pre_out * (1 + self.noise_factor * torch.randn_like(pre_out))
        # else:
        #     out = pre_out
        self.out = out
        return out

    # def pre_step(self):
    #     self.old_params = self.params.data.clone()
    #
    def project(self):
        # if self.old_params is not None:
        #     self.params.data = torch.where(torch.sgn(self.params.data) == -torch.sgn(self.old_params),
        #                                    torch.zeros_like(self.params.data), self.params.data)
        a = self.act_factor
        for i in [1, 3, 5, 7]:
            self.params.data[:, i].clamp_(min=-1.0 / a, max=1.0 / a)
        for i in [0, 2, 4, 6]:
            P0 = self.params.data[:, i - 1]
            P2 = self.params.data[:, (i + 1) % 8]
            upper_lim = torch.min((P0 + 1 / a) / 2, (P2 + 1 / a) / 2)
            lower_lim = torch.max((P0 - 1 / a) / 2, (P2 - 1 / a) / 2)
            self.params.data[:, i] = torch.max(torch.min(self.params.data[:, i], upper_lim), lower_lim)
        pass

    def penalty(self):
        return self.pen_act * torch.sum(torch.abs(self.out))


class B2ActivationModified(ManifoldModule):
    """Quick and dirty modification to B2Activation to force the output to be zero when any of the inputs are zero
    (i.e., keeping only the parameters for the 2^n corners with coordinates +/1, and setting all other parameters
    to zero)."""

    def __init__(self, input_groups):
        super().__init__()
        self.input_groups = input_groups
        self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 8]) * 2 - 1).to(torch.float32))
        self.project()

    def forward(self, inp):
        inp_view = inp.view(inp.shape[0], self.input_groups, 2)
        X = inp_view[:, :, 0]
        Y = inp_view[:, :, 1]
        P1 = self.params[:, 1].view(1, -1)  # Value at (1, 1)
        P3 = self.params[:, 3].view(1, -1)  # Value at (-1, 1)
        P5 = self.params[:, 5].view(1, -1)  # Value at (-1, -1)
        P7 = self.params[:, 7].view(1, -1)  # Value at (1, -1)
        P0 = torch.zeros_like(P1)
        P2 = P0
        P4 = P0
        P6 = P0
        XP = X >= 0
        YP = Y >= 0
        XY = torch.abs(X) >= torch.abs(Y)
        out = torch.where(XP,
                          torch.where(YP,
                                      torch.where(XY,
                                                  P0 * X + (P1 - P0) * Y,
                                                  P2 * Y + (P1 - P2) * X),
                                      torch.where(XY,
                                                  P0 * X + (P0 - P7) * Y,
                                                  -P6 * Y + (P7 - P6) * X)),
                          torch.where(YP,
                                      torch.where(XY,
                                                  -P4 * X + (P3 - P4) * Y,
                                                  P2 * Y + (P2 - P3) * X),
                                      torch.where(XY,
                                                  -P4 * X + (P4 - P5) * Y,
                                                  -P6 * Y + (P6 - P5) * X)))
        self.out = out
        return out

    def project(self):
        for i in [1, 3, 5, 7]:
            self.params.data[:, i].clamp_(min=-1.0, max=1.0)
        for i in [0, 2, 4, 6]:
            P0 = self.params.data[:, i - 1]
            P2 = self.params.data[:, (i + 1) % 8]
            upper_lim = torch.min((P0 + 1) / 2, (P2 + 1) / 2)
            lower_lim = torch.max((P0 - 1) / 2, (P2 - 1) / 2)
            self.params.data[:, i] = torch.max(torch.min(self.params.data[:, i], upper_lim), lower_lim)

    def penalty(self):
        return 0.0


class B2ActivationNonnegative(ManifoldModule):
    """Modification of B2Activation to restrict output to be nonnegative."""

    def __init__(self, input_groups):
        super().__init__()
        self.input_groups = input_groups
        self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 8]) * 2 - 1).to(torch.float32))
        self.project()

    def forward(self, inp):
        inp_view = inp.view(inp.shape[0], self.input_groups, 2)
        X = inp_view[:, :, 0]
        Y = inp_view[:, :, 1]
        P0 = self.params[:, 0].view(1, -1)  # Value at (1, 0)
        P1 = self.params[:, 1].view(1, -1)  # Value at (1, 1)
        P2 = self.params[:, 2].view(1, -1)  # Value at (0, 1)
        P3 = self.params[:, 3].view(1, -1)  # Value at (-1, 1)
        P4 = self.params[:, 4].view(1, -1)  # Value at (-1, 0)
        P5 = self.params[:, 5].view(1, -1)  # Value at (-1, -1)
        P6 = self.params[:, 6].view(1, -1)  # Value at (0, -1)
        P7 = self.params[:, 7].view(1, -1)  # Value at (1, -1)
        XP = X >= 0
        YP = Y >= 0
        XY = torch.abs(X) >= torch.abs(Y)
        out = torch.where(XP,
                          torch.where(YP,
                                      torch.where(XY,
                                                  P0 * X + (P1 - P0) * Y,
                                                  P2 * Y + (P1 - P2) * X),
                                      torch.where(XY,
                                                  P0 * X + (P0 - P7) * Y,
                                                  -P6 * Y + (P7 - P6) * X)),
                          torch.where(YP,
                                      torch.where(XY,
                                                  -P4 * X + (P3 - P4) * Y,
                                                  P2 * Y + (P2 - P3) * X),
                                      torch.where(XY,
                                                  -P4 * X + (P4 - P5) * Y,
                                                  -P6 * Y + (P6 - P5) * X)))
        self.out = out
        return out

    def project(self):
        for i in range(8):
            self.params.data[:, i].clamp_(min=0.0, max=1.0)
        for i in [0, 2, 4, 6]:
            P0 = self.params.data[:, i - 1]
            P2 = self.params.data[:, (i + 1) % 8]
            upper_lim = torch.min((P0 + 1) / 2, (P2 + 1) / 2)
            lower_lim = torch.max((P0 - 1) / 2, (P2 - 1) / 2)
            self.params.data[:, i] = torch.max(torch.min(self.params.data[:, i], upper_lim), lower_lim)

    def penalty(self):
        return 0.0



class ClampedB2Activation(ManifoldModule):
    def __init__(self, input_groups, act_factor, pen_act):
        super().__init__()
        self.input_groups = input_groups
        self.act_factor = act_factor
        self.pen_act = pen_act
        self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 8]) * 2 - 1).to(torch.float32))
        # self.old_params = None
        self.project()

    def forward(self, inp):
        inp_view = inp.view(inp.shape[0], self.input_groups, 2)
        X = inp_view[:, :, 0]
        Y = inp_view[:, :, 1]
        # if self.training and self.noise_factor != 0.0:
        #     P_noise = self.params * (1 + self.noise_factor * torch.randn_like(self.params))
        # else:
        #     P_noise = self.params
        # P = P_noise * self.act_factor
        P = self.params * self.act_factor
        P0 = P[:, 0].view(1, -1)  # Value at (1, 0)
        P1 = P[:, 1].view(1, -1)  # Value at (1, 1)
        P2 = P[:, 2].view(1, -1)  # Value at (0, 1)
        P3 = P[:, 3].view(1, -1)  # Value at (-1, 1)
        P4 = P[:, 4].view(1, -1)  # Value at (-1, 0)
        P5 = P[:, 5].view(1, -1)  # Value at (-1, -1)
        P6 = P[:, 6].view(1, -1)  # Value at (0, -1)
        P7 = P[:, 7].view(1, -1)  # Value at (1, -1)
        XP = X >= 0
        YP = Y >= 0
        XY = torch.abs(X) >= torch.abs(Y)
        out = torch.where(XP,
                          torch.where(YP,
                                      torch.where(XY,
                                                  P0 * X + (P1 - P0) * Y,
                                                  P2 * Y + (P1 - P2) * X),
                                      torch.where(XY,
                                                  P0 * X + (P0 - P7) * Y,
                                                  -P6 * Y + (P7 - P6) * X)),
                          torch.where(YP,
                                      torch.where(XY,
                                                  -P4 * X + (P3 - P4) * Y,
                                                  P2 * Y + (P2 - P3) * X),
                                      torch.where(XY,
                                                  -P4 * X + (P4 - P5) * Y,
                                                  -P6 * Y + (P6 - P5) * X)))
        out = torch.clamp(out, min=0.0)
        self.out = out
        return out

    # def pre_step(self):
    #     self.old_params = self.params.data.clone()
    #
    def project(self):
        # if self.old_params is not None:
        #     self.params.data = torch.where(torch.sgn(self.params.data) == -torch.sgn(self.old_params),
        #                                    torch.zeros_like(self.params.data), self.params.data)
        a = self.act_factor
        for i in [1, 3, 5, 7]:
            self.params.data[:, i].clamp_(min=-1.0 / a, max=1.0 / a)
        for i in [0, 2, 4, 6]:
            P0 = self.params.data[:, i - 1]
            P2 = self.params.data[:, (i + 1) % 8]
            upper_lim = torch.min((P0 + 1 / a) / 2, (P2 + 1 / a) / 2)
            lower_lim = torch.max((P0 - 1 / a) / 2, (P2 - 1 / a) / 2)
            self.params.data[:, i] = torch.max(torch.min(self.params.data[:, i], upper_lim), lower_lim)

    def penalty(self):
        return self.pen_act * torch.sum(torch.abs(self.out))


class A1Activation(ManifoldModule):
    """Activation based on the A_1 root system. The line y = x partitions R^2 into the two Weyl chambers, and for each
    2-input unit, this activation function can be an arbitrary continuous function on R^2 which is linear on each of
    the two chambers subject to the "tame" constraint |df/dx| + |df/dy| <= 1; such a function is uniquely determined
    by its values on the three points (0, 1), (1, 0), and (1, 1), subject to the constraints

       -1 <= f(1, 1) <= 1
       1/2 * (f(1, 1) - 1) <= f(1, 0) <= 1/2 * (f(1, 1) + 1)
       1/2 * (f(1, 1) - 1) <= f(0, 1) <= 1/2 * (f(1, 1) + 1)
       """

    def __init__(self, input_groups):
        super().__init__()
        self.input_groups = input_groups
        self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 3]) * 2 - 1).to(torch.float32))
        self.project()

    def forward(self, inp):
        inp_view = inp.view(inp.shape[0], self.input_groups, 2)
        X = inp_view[:, :, 0]
        Y = inp_view[:, :, 1]
        P0 = self.params[:, 0].view(1, -1)  # Value at (1, 0)
        P1 = self.params[:, 1].view(1, -1)  # Value at (1, 1)
        P2 = self.params[:, 2].view(1, -1)  # Value at (0, 1)
        out = torch.where(X >= Y,
                          P0 * X + (P1 - P0) * Y,
                          P2 * Y + (P1 - P2) * X)
        self.out = out
        return out

    def project(self):
        self.params.data[:, 1].clamp_(min=-1.0, max=1.0)
        upper_lim = (self.params.data[:, 1] + 1) / 2
        lower_lim = (self.params.data[:, 1] - 1) / 2
        self.params.data[:, 0] = torch.max(torch.min(self.params.data[:, 0], upper_lim), lower_lim)
        self.params.data[:, 2] = torch.max(torch.min(self.params.data[:, 2], upper_lim), lower_lim)

    def penalty(self):
        return 0.0


class MonotoneA1Activation(ManifoldModule):
    def __init__(self, input_groups):
        super().__init__()
        self.input_groups = input_groups
        self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 2]) * 2 - 1).to(torch.float32))

    def forward(self, inp):
        inp_view = inp.view(inp.shape[0], self.input_groups, 2)
        X = inp_view[:, :, 0]
        Y = inp_view[:, :, 1]
        ZP = self.params[:, 0].view(1, -1)  # Value on (0, 1)
        PZ = self.params[:, 1].view(1, -1)  # Value on (1, 0)
        out = torch.where(Y <= X,
                          PZ * X + (1 - PZ) * Y,
                          (1 - ZP) * X + ZP * Y)
        self.out = out
        return out

    def project(self):
        self.params.data = torch.clamp(self.params.data, min=0.0, max=1.0)

    def penalty(self):
        return 0.0


def high_order_act(A, params):
    A_sort, A_ind = torch.sort(A, dim=2)
    A_diff = A_sort[:, :, 1:] - A_sort[:, :, :-1]
    coef = torch.cat([A_sort[:, :, 0:1], A_diff], dim=2)
    params_A_ind = torch.flip(torch.cumsum(torch.flip(2 ** A_ind, dims=[2]), dim=2), dims=[2])
    ind0 = torch.unsqueeze(torch.unsqueeze(torch.arange(0, params.shape[0], dtype=torch.int64), 1), 2)
    ind1 = torch.transpose(params_A_ind, 0, 1)
    params_gather = params[ind0, ind1, :]
    out = torch.einsum('jikl,ijk->ijl', params_gather, coef)
    return out


class MonotoneActivation(ManifoldModule):
    def __init__(self, arity, input_groups, out_dim, act_factor=1.0):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.act_factor = act_factor
        self.params = torch.nn.Parameter(torch.randint(0, 2, [input_groups, 2 ** arity, out_dim]).to(torch.float32) / self.act_factor)
        self.params = torch.nn.Parameter(
            torch.rand([input_groups, 2 ** arity, out_dim]).to(torch.float32) / self.act_factor)
        self.project()
        self.params.data = torch.round(self.params.data * self.act_factor) / self.act_factor
        # self.params = torch.nn.Parameter(
        #     torch.full([input_groups, 2 ** arity, out_dim], 1.0 / self.act_factor))  # Initialize as maxout
        # self.params.data[:, 0, :] = 0.0

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)
        out1 = high_order_act(X1, self.params) * self.act_factor
        self.out = out1
        return out1.view(X.shape[0], self.input_groups * self.out_dim)

    def project(self):
        # Compute the greatest monotone lower bound and least monotone upper bound and average them.
        # Theoretically it would be better to use the Euclidean projection but that would be more
        # difficult to compute.
        self.params.data[:, 2 ** self.arity - 1, :] = 1.0 / self.act_factor
        clamped = torch.clamp(self.params, min=0.0, max=1.0 / self.act_factor)
        lower = clamped
        upper = clamped.clone()
        for i in range(self.arity):
            shape0 = 2 ** i
            shape1 = 2 ** (self.arity - i - 1)
            lower_view = lower.view(self.input_groups, shape0, 2, shape1, self.out_dim)
            lower_min = torch.min(lower_view[:, :, 0, :, :], lower_view[:, :, 1, :, :])
            lower_view[:, :, 0, :, :] = lower_min
            upper_view = upper.view(self.input_groups, shape0, 2, shape1, self.out_dim)
            upper_max = torch.max(upper_view[:, :, 0, :, :], upper_view[:, :, 1, :, :])
            upper_view[:, :, 1, :, :] = upper_max
        self.params.data = (lower + upper) / 2

    def penalty(self):
        return 0.0


class ClampedMonotoneActivation(ManifoldModule):
    def __init__(self, arity, input_groups, out_dim, act_factor=1.0):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.act_factor = act_factor
        # self.params = torch.nn.Parameter(torch.randint(0, 2, [input_groups, 2 ** arity, out_dim]).to(torch.float32) / self.act_factor)
        # self.params = torch.nn.Parameter(
        #     torch.rand([input_groups, 2 ** arity, out_dim]).to(torch.float32) / self.act_factor)
        self.params = torch.nn.Parameter(torch.full([input_groups, 2 ** arity, out_dim], 1.0 / self.act_factor))  # Initialize as maxout
        self.params.data[:, 0, :] = 0.0
        self.bias = torch.nn.Parameter(torch.zeros([input_groups, out_dim], dtype=torch.float32))
        # self.project()
        # self.params.data = torch.round(self.params.data * self.act_factor) / self.act_factor

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)
        out1 = torch.clamp(high_order_act(X1, self.params) * self.act_factor + self.bias.unsqueeze(0), min=0.0)
        self.out = out1
        return out1.view(X.shape[0], self.input_groups * self.out_dim)

    def project(self):
        # Compute the greatest monotone lower bound and least monotone upper bound and average them.
        # Theoretically it would be better to use the Euclidean projection but that would be more
        # difficult to compute.
        clamped = torch.clamp(self.params, min=0.0, max=1.0 / self.act_factor)
        lower = clamped
        upper = clamped.clone()
        for i in range(self.arity):
            shape0 = 2 ** i
            shape1 = 2 ** (self.arity - i - 1)
            lower_view = lower.view(self.input_groups, shape0, 2, shape1, self.out_dim)
            lower_min = torch.min(lower_view[:, :, 0, :, :], lower_view[:, :, 1, :, :])
            lower_view[:, :, 0, :, :] = lower_min
            upper_view = upper.view(self.input_groups, shape0, 2, shape1, self.out_dim)
            upper_max = torch.max(upper_view[:, :, 0, :, :], upper_view[:, :, 1, :, :])
            upper_view[:, :, 1, :, :] = upper_max
        self.params.data = (lower + upper) / 2
        self.params.data[:, 2 ** self.arity - 1, :] = 1.0 / self.act_factor

    def penalty(self):
        return 0.0


class GateActivation(torch.nn.Module):
    def __init__(self, num_outputs, pen_act):
        super().__init__()
        self.num_outputs = num_outputs
        self.pen_act = pen_act

    def forward(self, X):
        X1 = X.view(X.shape[0], 2, self.num_outputs)
        G = X1[:, 0, :]
        Y = X1[:, 1, :]
        G_clamp = torch.clamp(G, min=0.0)
        # out = torch.min(G_clamp, torch.abs(Y)) * torch.sgn(Y)
        out = torch.maximum(torch.minimum(Y, G_clamp), -G_clamp)
        # out = torch.minimum(Y, G_clamp)
        # out = torch.maximum(torch.minimum(Y, G), -G)

        self.out = out
        return out

    def penalty(self):
        return self.pen_act * torch.sum(torch.abs(self.out))
        # return self.pen_act * torch.sum(self.G)



class PassthroughGateActivation(torch.nn.Module):
    def __init__(self, num_outputs, pen_act):
        super().__init__()
        self.num_outputs = num_outputs
        self.pen_act = pen_act

    def forward(self, X):
        X1 = X.view(X.shape[0], 2, self.num_outputs)
        G = X1[:, 0, :]
        Y = X1[:, 1, :]
        G_clamp = torch.clamp(G, min=0.0)
        # out = torch.min(G_clamp, torch.abs(Y)) * torch.sgn(Y)
        Y_out = torch.maximum(torch.minimum(Y, G_clamp), -G_clamp)
        out = torch.cat([G_clamp, Y_out], dim=1)
        # out = torch.minimum(Y, G_clamp)
        # out = torch.maximum(torch.minimum(Y, G), -G)

        self.out = out
        return out

    def penalty(self):
        return self.pen_act * torch.sum(torch.abs(self.out))
        # return self.pen_act * torch.sum(self.G)


class GateAbsActivation(torch.nn.Module):
    def __init__(self, num_outputs, pen_act):
        super().__init__()
        self.num_outputs = num_outputs
        self.pen_act = pen_act

    def forward(self, X):
        X1 = X.view(X.shape[0], 2, self.num_outputs)
        G = X1[:, 0, :]
        Y = X1[:, 1, :]
        G_clamp = torch.clamp(G, min=0.0)
        out = torch.minimum(torch.abs(Y), G_clamp)
        self.out = out
        return out

    def penalty(self):
        return self.pen_act * torch.sum(torch.abs(self.out))
        # return self.pen_act * torch.sum(self.G)


class ParametricGateActivation(ManifoldModule):
    def __init__(self, num_outputs, pen_act, dtype=torch.float32, device=None):
        super().__init__()
        self.num_outputs = num_outputs
        self.pen_act = pen_act
        self.slope_left = torch.nn.Parameter(torch.zeros([num_outputs], dtype=dtype, device=device))
        self.slope_right = torch.nn.Parameter(torch.ones([num_outputs], dtype=dtype, device=device))

    def forward(self, X):
        X1 = X.view(X.shape[0], 2, self.num_outputs)
        G = X1[:, 0, :]
        Y = X1[:, 1, :]
        G_clamp = torch.clamp(G, min=0.0)
        Y1 = self.slope_right * torch.clamp(Y, min=0.0) - self.slope_left * torch.clamp(Y, max=0.0)
        # out = torch.min(G_clamp, torch.abs(Y)) * torch.sgn(Y)
        out = torch.maximum(torch.minimum(Y1, G_clamp), -G_clamp)
        self.out = out
        return out

    def penalty(self):
        return self.pen_act * torch.sum(torch.abs(self.out))
        # return self.pen_act * torch.sum(self.G)

    def project(self):
        self.slope_left.data = torch.clamp(self.slope_left.data, min=-1.0, max=1.0)
        self.slope_right.data = torch.clamp(self.slope_right.data, min=-1.0, max=1.0)


class CoherentSignActivation(torch.nn.Module):
    def __init__(self, num_outputs, pen_act):
        super().__init__()
        self.num_outputs = num_outputs
        self.pen_act = pen_act

    def forward(self, X):
        X1 = X.view(X.shape[0], 2, self.num_outputs)
        A = X1[:, 0, :]
        B = X1[:, 1, :]
        AP = A >= 0
        BP = B >= 0
        out = torch.where(AP,
                          torch.where(BP, torch.minimum(A, B), torch.zeros_like(A)),
                          torch.where(BP, torch.zeros_like(A), torch.max(A, B)))
        self.out = out
        return out

    def penalty(self):
        return self.pen_act * torch.sum(torch.abs(self.out))
        # return self.pen_act * torch.sum(self.G)


# act = CoherentSignActivation(1, 0.0)
# A = torch.tensor([
#     [-1.0, -0.5], [-0.3, -0.8],
#     [1.0, -0.5], [0.3, -0.8],
#     [-1.0, 0.5], [-0.3, 0.8],
#     [1.0, 0.5], [0.3, 0.8],
# ])
# act(A)


class DoubleGateActivation(torch.nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs

    def forward(self, X):
        X1 = X.view(X.shape[0], 3, self.num_outputs)
        GL = X1[:, 0, :]
        GH = X1[:, 1, :]
        Y = X1[:, 2, :]
        GL_clamp = torch.clamp(GL, max=0.0)
        GH_clamp = torch.clamp(GH, min=0.0)
        out = torch.maximum(torch.minimum(Y, GH_clamp), GL_clamp)
        self.out = out
        return out

    def penalty(self):
        return 0.0


class TopKSparsifier(torch.nn.Module):
    # TODO: Reimplement this using sparse output matrix
    def __init__(self, num_inputs, k):
        super().__init__()
        self.num_inputs = num_inputs
        self.k = k

    def forward(self, X):
        assert X.shape[1] == self.num_inputs
        k_int = int(math.ceil(self.k))
        X_abs = torch.abs(X)
        _, indices = torch.topk(X_abs, k_int, dim=1, sorted=True)
        values = X[torch.arange(X.shape[0]).view(-1, 1), indices]
        if k_int != self.k:
            scale = torch.ones([k_int], dtype=X.dtype, device=X.device)
            scale[-1] = 1.0 - (k_int - self.k)
            values = values * scale.view(1, -1)
        out = torch.zeros_like(X).scatter(1, indices, values)
        self.out = out
        return out.view(X.shape[0], X.shape[1])


class TopKContSparsifier(torch.nn.Module):
    # TODO: Reimplement this using sparse output matrix
    def __init__(self, num_inputs, k):
        super().__init__()
        self.num_inputs = num_inputs
        self.k = k

    def forward(self, X):
        assert X.shape[1] == self.num_inputs
        X_abs = torch.abs(X)
        Q = torch.quantile(X_abs, 1 - self.k / self.num_inputs, dim=1)
        X_trunc = torch.clamp(X_abs - Q.unsqueeze(1), min=0.0)
        out = torch.sgn(X) * X_trunc
        self.out = out
        return out


class GeneralizedMaxout(torch.nn.Module):
    def __init__(self, arity, terms, output_width, bias_factor, dtype=torch.float32, device=None, num_iters=6):
        super().__init__()
        self.arity = arity
        self.terms = terms
        self.output_width = output_width
        self.bias_factor = bias_factor
        self.weights_pos_neg = Simplex([arity * 2, terms, output_width], dim=0, dtype=dtype, device=device,
                                       num_iters=num_iters)
        self.bias = torch.nn.Parameter(torch.zeros([output_width, terms], dtype=dtype, device=device))

    def forward(self, X):
        X1 = X.view(X.shape[0], -1, self.arity)
        weights = self.weights_pos_neg.param[:self.arity, :, :] - self.weights_pos_neg.param[self.arity:, :, :]
        Y1 = torch.einsum('ijk,klj->ijl', X1, weights)
        Y2 = Y1 + self.bias.unsqueeze(0) * self.bias
        out = torch.max(Y2, dim=2)[0]
        self.out = out
        return out

    def penalty(self):
        return 0.0


class ClampedGeneralizedMaxout(torch.nn.Module):
    def __init__(self, arity, terms, output_width, bias_factor, dtype=torch.float32, device=None, num_iters=6):
        super().__init__()
        self.arity = arity
        self.terms = terms
        self.output_width = output_width
        self.bias_factor = bias_factor
        self.weights_pos_neg = Simplex([arity * 2, terms, output_width], dim=0, dtype=dtype, device=device,
                                       num_iters=num_iters)
        self.bias = torch.nn.Parameter(torch.zeros([output_width, terms], dtype=dtype, device=device))

    def forward(self, X):
        X1 = X.view(X.shape[0], -1, self.arity)
        weights = self.weights_pos_neg.param[:self.arity, :, :] - self.weights_pos_neg.param[self.arity:, :, :]
        Y1 = torch.einsum('ijk,klj->ijl', X1, weights)
        Y2 = Y1 + self.bias.unsqueeze(0) * self.bias
        out = torch.clamp(torch.max(Y2, dim=2)[0], min=0.0)
        self.out = out
        return out

    def penalty(self):
        return 0.0

#
# class MultiGate(torch.nn.Module):
#     def __init__(self, input_dim, output_dim, dtype=torch.float32):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.scale_a = torch.nn.Parameter(torch.randint(0, 2, [input_dim, input_dim, output_dim]).to(dtype) * 2 - 1)
#         self.scale_b = torch.nn.Parameter(torch.randint(0, 2, [input_dim, input_dim, output_dim]).to(dtype) * 2 - 1)
#         self.bias_a = torch.nn.Parameter(torch.zeros([input_dim, input_dim, output_dim], dtype=dtype))
#         self.weights_pos_neg = Simplex([2 * input_dim * input_dim, output_dim], dim=0)
#         self.bias_out = torch.nn.Parameter(torch.zeros([output_dim], dtype=dtype))
#
#     def forward(self, X):
#         A = X.unsqueeze(1)
#         B = X.unsqueeze(2)
#         scaled_a = A * self.scale_a.unsqueeze(0) + self.bias_a.unsqueeze(0)
#         scaled_b = B * self.scale_b.unsqueeze(0)
#         min_ab = torch.minimum(scaled_a, scaled_b)
#         weights = self.weights_pos_neg[:(self.input_dim * self.input_dim), :] - self.weights_pos_neg[(self.input_dim * self.input_dim):, :]
#


class MultiGate(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bias_factor=1.0, dtype=torch.float32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias_factor = bias_factor
        self.weights_pos_neg = Simplex([4 * input_dim * input_dim, output_dim], dim=0)
        self.bias = torch.nn.Parameter(torch.zeros([output_dim], dtype=dtype))

    def forward(self, X):
        weights = self.weights_pos_neg.param[:(2 * self.input_dim * self.input_dim), :] - \
                  self.weights_pos_neg.param[(2 * self.input_dim * self.input_dim):, :]
        weights_min = weights[:(self.input_dim * self.input_dim), :]
        weights_max = weights[(self.input_dim * self.input_dim):, :]
        A = X.unsqueeze(1)
        B = X.unsqueeze(2)
        min_AB = torch.min(A, B).view(X.shape[0], self.input_dim * self.input_dim)
        max_AB = torch.max(A, B).view(X.shape[0], self.input_dim * self.input_dim)
        out = torch.matmul(min_AB, weights_min) + torch.matmul(max_AB, weights_max) + self.bias.unsqueeze(0) * self.bias_factor
        self.out = out
        return out

    def penalty(self):
        return 0.0


class L1LinearMultiplexer(torch.nn.Module):
    def __init__(self, input_dim, num_units, gate_arity, bias_factor=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_units = num_units
        self.gate_arity = gate_arity
        self.bias_factor = bias_factor
        self.lin_gate = L1LinearScaled(input_dim, num_units * gate_arity, bias_factor=bias_factor)
        self.lin_payload = L1LinearScaled(input_dim, num_units * 2 ** gate_arity, bias_factor=bias_factor)

    def forward(self, X, max_scale):
        # Compute only the part of P that is needed, and use sparsity
        G = self.lin_gate(X, max_scale).view(X.shape[0], self.num_units, self.gate_arity)
        P = self.lin_payload(X, max_scale).view(X.shape[0], self.num_units, 2 ** self.gate_arity)
        min_G = torch.min(torch.abs(G), dim=2)[0]
        pow2 = 2 ** torch.arange(self.gate_arity).view(1, 1, self.gate_arity)
        ind = torch.sum((G > 0).to(torch.int64) * pow2, dim=2)
        P1 = P[torch.arange(X.shape[0]).view(-1, 1), torch.arange(self.num_units).view(1, -1), ind]
        out = torch.minimum(torch.maximum(P1, -min_G), min_G)
        self.out = out
        return out


class MultiGateNetwork(torch.nn.Module):
    def __init__(self,
                 widths: List[int],
                 pen_scale: float = 0.0,
                 bias_factor: float = 1.0,
                 scale_init: float = 1.0,
                 scale_factor: float = 1.0,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.widths = widths
        self.depth = len(widths) - 1
        self.pen_scale = pen_scale
        self.scale_factor = scale_factor
        self.lin_layer = L1Linear(widths[0], widths[1])
        self.act_layers = torch.nn.ModuleList([])
        self.scale = torch.nn.Parameter(torch.full([], scale_init / scale_factor, dtype=dtype, device=device))
        for i in range(1, self.depth):
            self.act_layers.append(MultiGate(widths[i], widths[i + 1], bias_factor=bias_factor, dtype=dtype))

    def forward(self, X, max_scale):
        self.cnt_rows = X.shape[0]
        overall_scale = torch.abs(self.scale) * self.scale_factor
        layer_scale = overall_scale ** (1 / self.depth)
        X = X * layer_scale
        X = self.lin_layer(X)
        for i in range(self.depth - 1):
            X = X * layer_scale
            X = self.act_layers[i](X)
        return X

    def penalty(self):
        # return 0.0
        return sum(layer.penalty() for layer in self.act_layers) + \
               self.lin_layer.penalty() + \
               self.cnt_rows * self.pen_scale * torch.abs(self.scale * self.scale_factor)
        # return sum(layer.penalty() for layer in self.lin_layers) + \
        #     self.pen_scale * torch.sum((self.scale * self.scale_factor) ** 2)


class MultiplexerNetwork(torch.nn.Module):
    def __init__(self,
                 widths: List[int],
                 gate_arity: int,
                 pen_scale: float = 0.0,
                 scale_init: float = 1.0,
                 scale_factor: float = 1.0,
                 bias_factor: float = 1.0,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.widths = widths
        self.depth = len(widths) - 1
        self.pen_scale = pen_scale
        self.scale_factor = scale_factor
        self.multiplexer_layers = torch.nn.ModuleList([])
        self.scale = torch.nn.Parameter(torch.full([], scale_init / scale_factor, dtype=dtype, device=device))
        for i in range(self.depth):
            self.multiplexer_layers.append(
                L1LinearMultiplexer(widths[i], num_units=widths[i + 1], gate_arity=gate_arity,
                                    bias_factor=bias_factor))

    def forward(self, X):
        self.cnt_rows = X.shape[0]
        overall_scale = torch.abs(self.scale) * self.scale_factor
        layer_scale = overall_scale ** (1 / self.depth)
        for i in range(self.depth):
            # X = X * layer_scale
            # X = self.multiplexer_layers[i](X)
            X = self.multiplexer_layers[i](X, layer_scale)
        return X

    def penalty(self):
        return self.cnt_rows * self.pen_scale * torch.abs(self.scale * self.scale_factor)


class OrthogonalButterfly(torch.nn.Module):
    def __init__(self, input_width, output_width, width_pow, depth, l2_interact, dtype=torch.float, device=None):
        super().__init__()
        self.dtype = dtype
        self.input_width = input_width
        self.output_width = output_width
        self.width_pow = width_pow
        self.width = 2 ** width_pow
        assert output_width <= self.width
        self.half_width = 2 ** (width_pow - 1)
        self.depth = depth
        self.l2_interact = l2_interact
        initial_params = torch.rand(self.half_width, depth, dtype=dtype, device=device) * math.pi * 2
        self.params = torch.nn.Parameter(initial_params)
        self.perm = torch.zeros([self.width], dtype=torch.long, device=device)
        self.bias = torch.nn.Parameter(torch.zeros([output_width], dtype=dtype, device=device))
        for i in range(self.width):
            if i % 2 == 0:
                self.perm[i] = i // 2
            else:
                self.perm[i] = i // 2 + self.half_width

    def forward(self, X):
        assert X.dtype == self.dtype
        assert X.shape[1] == self.input_width
        X = torch.cat([X, torch.zeros([X.shape[0], self.width - X.shape[1]], dtype=X.dtype, device=X.device)], dim=1)
        for i in range(self.depth):
            theta = self.params[:, i]
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            X0 = X[:, :self.half_width]
            X1 = X[:, self.half_width:]
            new_X0 = X0 * cos_theta + X1 * sin_theta
            new_X1 = X0 * -sin_theta + X1 * cos_theta
            X = torch.cat([new_X0, new_X1], dim=1)
            X = X[:, self.perm]
        return X[:, :self.output_width] + self.bias.unsqueeze(0)

    def penalty(self):
        if self.l2_interact == 0:
            return 0.0
        else:
            return torch.sum(self.l2_interact * torch.sin(2*self.params)**2)


class SparseHighOrderActivationB(ManifoldModule):
    def __init__(self, arity, input_groups, out_dim, dtype=torch.float32, device=None):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.params = torch.nn.Parameter((torch.randint(0, 2, [input_groups, 2 ** arity, out_dim], device=device) * 2 - 1).to(dtype))
        # self.params = torch.nn.Parameter(
        #     (torch.rand([input_groups, 2 ** arity, out_dim], device=device) * 2 - 1).to(dtype))

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)
        min_X1 = torch.min(torch.abs(X1), dim=2)[0]
        pow2 = (2 ** torch.arange(self.arity, dtype=torch.int64))
        # ind = torch.einsum('ijk,k->ij', (X1 >= 0).to(torch.int64), pow2)
        ind = torch.sum((X1 >= 0).to(torch.int64) * pow2.view(1, 1, -1), dim=2)
        param = self.params[
            torch.arange(self.input_groups).view(1, -1, 1), ind.unsqueeze(2), torch.arange(self.out_dim).view(1, 1, -1)]
        out1 = param * min_X1.unsqueeze(2)
        out = out1.view(X.shape[0], self.input_groups * self.out_dim)
        self.out = out
        return out

    def penalty(self):
        return 0.0

    def project(self):
        self.params.data = torch.clamp(self.params.data, min=-1.0, max=1.0)
        # pass

# self = SparseHighOrderActivationB(2, 1, 1)
# X = torch.tensor([
#     [-10.0, -10.0],
#     [10.0, -10.0],
#     [-10.0, 10.0],
#     [1.0, 1.0],
#     [1.0, 0.0],
#     [0.0, 1.0],
#     [0.0, 0.0],
# ])
# self(X)


class Shuffle(torch.nn.Module):
    def __init__(self, input_width, output_width, device=None):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        rep_indices = torch.arange(input_width, device=device).repeat((output_width + input_width - 1) // input_width)
        self.indices = rep_indices[torch.randperm(output_width)][:output_width]

    def forward(self, X):
        assert X.shape[1] == self.input_width
        return X[:, self.indices]

    def penalty(self):
        return 0.0


class CornerActivationB(ManifoldModule):
    def __init__(self, arity, input_groups, out_dim, act_init=1.0, act_factor=1.0):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.act_factor = act_factor
        r = torch.randint(2, [input_groups, 2 ** arity, out_dim])
        self.params = torch.nn.Parameter((r * 2 - 1).to(torch.float32) * act_init / act_factor)
        # self.params = torch.nn.Parameter(torch.full([input_groups, 2 ** arity, out_dim], -act_init / act_factor))
        # self.params.data[:, 2 ** arity - 1, :] = act_init / act_factor

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)

        expanded_params = self.params
        for i in range(self.arity):
            shape0 = 3 ** i
            shape1 = 2 ** (self.arity - i - 1)
            param_view = expanded_params.view(self.input_groups, shape0, 2, shape1, self.out_dim)
            expanded_params = torch.stack([
                param_view[:, :, 0, :, :],
                (param_view[:, :, 0, :, :] + param_view[:, :, 1, :, :]) / 2,
                param_view[:, :, 1, :, :],
            ], dim=2)
        expanded_params = expanded_params.view(self.input_groups, 3 ** self.arity, self.out_dim)

        out1 = high_order_act_b(X1, expanded_params)
        out = out1.view(X.shape[0], self.input_groups * self.out_dim) * self.act_factor
        self.out = out
        return out

    def project(self):
        a = 1 / self.act_factor
        self.params.data = torch.clamp(self.params.data, min=-a, max=a)

    def penalty(self):
        return 0.0


# act = CornerActivationB(3, 1, 1)
# print(act.params)
# A = torch.randint(2, [10, 3], dtype=torch.float32) * 2 - 1
# print(act(A))
#


class RestrictedHighOrderActivationA(ManifoldModule):
    """Version of HighOrderActivation with parameters clamped to between 0.0 and 1.0. This isn't actually a tame
    activation."""
    def __init__(self, arity, input_groups, out_dim):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.params = torch.nn.Parameter(torch.randn([input_groups, 2 ** arity, out_dim]))
        param_coords = cartesian_power([0.0, 1.0], arity, dtype=torch.float32)
        self.params.data[:, :, :] = torch.max(param_coords, dim=1)[0].view(1, 2 ** arity, 1)

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)
        out1 = high_order_act(X1, self.params)
        out = out1.view(X.shape[0], self.input_groups * self.out_dim)
        self.out = out
        return out

    def penalty(self):
        return 0.0

    def project(self):
        self.params.data = torch.clamp(self.params, min=0.0, max=1.0)


class RestrictedHighOrderActivationB(ManifoldModule):
    def __init__(self, arity, input_groups, out_dim, act_init=1.0, act_factor=1.0):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.act_factor = act_factor
        self.params = torch.nn.Parameter(torch.randn([input_groups, 3 ** arity, out_dim]))
        param_coords = cartesian_power([-1.0, 0.0, 1.0], arity, dtype=torch.float32)
        self.params.data[:, :, :] = torch.max(param_coords, dim=1)[0].view(1, 3 ** arity, 1) * act_init / act_factor  # Initialize as maxout
        self.valency = self.arity + torch.sum((param_coords == 0.0).to(torch.float32), dim=1)  # Number of neighbors of each parameter
        self.project()

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)
        out1 = high_order_act_b(X1, self.params)
        out = out1.view(X.shape[0], self.input_groups * self.out_dim) * self.act_factor
        self.out = out
        return out

    def smoothen(self, weight_zero, weight_neighbor):
        accum = torch.zeros_like(self.params)
        for i in range(self.arity):
            shape0 = 3 ** i
            shape1 = 3 ** (self.arity - i - 1)
            param_view = self.params.data.view(self.input_groups, shape0, 3, shape1, self.out_dim)
            accum.view_as(param_view)[:, :, 0, :, :] += param_view[:, :, 1, :, :]
            accum.view_as(param_view)[:, :, 1, :, :] += param_view[:, :, 0, :, :]
            accum.view_as(param_view)[:, :, 1, :, :] += param_view[:, :, 2, :, :]
            accum.view_as(param_view)[:, :, 2, :, :] += param_view[:, :, 1, :, :]
        average_neighbor = accum / self.valency.unsqueeze(0).unsqueeze(2)
        a = (1 - weight_zero) * (1 - weight_neighbor)
        b = (1 - weight_zero) * weight_neighbor
        self.params.data = a * self.params.data + b * average_neighbor

    def penalty(self):
        return 0.0

    def project(self):
        self.params.data = torch.clamp(self.params, min=0.0, max=1.0 / self.act_factor)


class TameHighOrderActivationA(ManifoldModule):
    """Tame version of HighOrderActivationA"""
    def __init__(self, arity, input_groups, out_dim):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        # self.params = torch.nn.Parameter(torch.rand([input_groups, 2 ** arity, out_dim]) * 2 - 1)
        self.params = torch.nn.Parameter(torch.randint(0, 2, [input_groups, 2 ** arity, out_dim]).to(torch.float32) * 2 - 1)
        self.params.data[:, 0, :] = 0.0
        self.project()
        # param_coords = cartesian_power([0.0, 1.0], arity, dtype=torch.float32)
        # self.params.data[:, :, :] = torch.max(param_coords, dim=1)[0].view(1, 2 ** arity, 1)

    def _compute_vertex_id(self, x):
        return torch.sum(x * 2 ** torch.arange(self.arity)).to(torch.long)

    def _compute_neighbor_graph(self):
        param_coords = cartesian_power([0.0, 1.0], self.arity, dtype=torch.float32)
        out_tensor_pairs = []
        for num_zeros in range(self.arity + 1):
            parent_vertices = param_coords[torch.sum(param_coords == 0.0, dim=1) == num_zeros]
            parent_ids_list = []
            child_ids_list_list = []
            for i in range(parent_vertices.shape[0]):
                parent_vertex = parent_vertices[i, :]
                parent_id = self._compute_vertex_id(parent_vertex)
                parent_ids_list.append(parent_id)
                child_ids_list = []
                for j in range(self.arity):
                    if parent_vertex[j] == 0.0:
                        child_vertex = parent_vertex.clone()
                        child_vertex[j] = 1.0
                        child_id = self._compute_vertex_id(child_vertex)
                        child_ids_list.append(child_id)
                child_ids_list_list.append(child_ids_list)
            parent_ids_tensor = torch.tensor(parent_ids_list, dtype=torch.long)
            child_ids_tensor = torch.tensor(child_ids_list_list, dtype=torch.long)
            out_tensor_pairs.append((parent_ids_tensor, child_ids_tensor))
        return out_tensor_pairs

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)
        out1 = high_order_act(X1, self.params)
        out = out1.view(X.shape[0], self.input_groups * self.out_dim)
        self.out = out
        return out

    def penalty(self):
        return 0.0

    def _compute_activity(self):
        activity = torch.zeros_like(self.params)
        edge_tensor_pairs = self._compute_neighbor_graph()
        for num_zeros in range(1, self.arity + 1):
            parent_ids, child_ids = edge_tensor_pairs[num_zeros]
            parent_params = self.params[:, parent_ids, :]
            child_params = self.params[:, child_ids, :]
            child_activity = activity[:, child_ids, :]
            activity[:, parent_ids, :] = torch.max(torch.abs(parent_params.unsqueeze(2) - child_params) + child_activity, dim=2)[0]
        return activity[:, 0, :]

    def project(self):
        remaining_var = torch.ones_like(self.params)
        edge_tensor_pairs = self._compute_neighbor_graph()
        self.params.data = torch.clamp(self.params.data, min=-1.0, max=1.0)
        for num_zeros in range(1, self.arity + 1):
            parent_ids, child_ids = edge_tensor_pairs[num_zeros]
            parent_params = self.params[:, parent_ids, :]
            child_params = self.params[:, child_ids, :]
            child_var = remaining_var[:, child_ids, :]
            upper_lim = torch.min(child_params + child_var, dim=2)[0] / 2.0
            lower_lim = torch.max(child_params - child_var, dim=2)[0] / 2.0
            new_parent_params = torch.maximum(torch.minimum(parent_params, upper_lim), lower_lim)
            parent_var = torch.min(child_var - torch.abs(new_parent_params.unsqueeze(2) - child_params), dim=2)[0]
            self.params.data[:, parent_ids, :] = new_parent_params
            remaining_var[:, parent_ids, :] = parent_var


class TameHighOrderActivationB(ManifoldModule):
    """Tame version of HighOrderActivationA"""
    def __init__(self, arity, input_groups, out_dim):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.params = torch.nn.Parameter(torch.rand([input_groups, 3 ** arity, out_dim]) * 2 - 1)
        # self.params = torch.nn.Parameter(torch.randint(0, 2, [input_groups, 3 ** arity, out_dim]).to(torch.float32) * 2 - 1)
        self.params.data[:, (3 ** arity - 1) // 2, :] = 0.0
        self.project()
        # param_coords = cartesian_power([0.0, 1.0], arity, dtype=torch.float32)
        # self.params.data[:, :, :] = torch.max(param_coords, dim=1)[0].view(1, 2 ** arity, 1)

    def _compute_vertex_id(self, x):
        return torch.sum((x + 1) * 3 ** torch.arange(self.arity)).to(torch.long)

    def _compute_neighbor_graph(self):
        param_coords = cartesian_power([-1.0, 0.0, 1.0], self.arity, dtype=torch.float32)
        out_tensor_pairs = []
        for num_zeros in range(self.arity + 1):
            parent_vertices = param_coords[torch.sum(param_coords == 0.0, dim=1) == num_zeros]
            parent_ids_list = []
            child_ids_list_list = []
            for i in range(parent_vertices.shape[0]):
                parent_vertex = parent_vertices[i, :]
                parent_id = self._compute_vertex_id(parent_vertex)
                parent_ids_list.append(parent_id)
                child_ids_list = []
                for j in range(self.arity):
                    if parent_vertex[j] == 0.0:
                        child_vertex = parent_vertex.clone()
                        child_vertex[j] = 1.0
                        child_id = self._compute_vertex_id(child_vertex)
                        child_ids_list.append(child_id)

                        child_vertex[j] = -1.0
                        child_id = self._compute_vertex_id(child_vertex)
                        child_ids_list.append(child_id)
                child_ids_list_list.append(child_ids_list)
            parent_ids_tensor = torch.tensor(parent_ids_list, dtype=torch.long)
            child_ids_tensor = torch.tensor(child_ids_list_list, dtype=torch.long)
            out_tensor_pairs.append((parent_ids_tensor, child_ids_tensor))
        return out_tensor_pairs

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)
        out1 = high_order_act_b(X1, self.params)
        out = out1.view(X.shape[0], self.input_groups * self.out_dim)
        self.out = out
        return out

    def penalty(self):
        return 0.0

    def _compute_activity(self):
        activity = torch.zeros_like(self.params)
        edge_tensor_pairs = self._compute_neighbor_graph()
        for num_zeros in range(1, self.arity + 1):
            parent_ids, child_ids = edge_tensor_pairs[num_zeros]
            parent_params = self.params[:, parent_ids, :]
            child_params = self.params[:, child_ids, :]
            child_activity = activity[:, child_ids, :]
            activity[:, parent_ids, :] = torch.max(torch.abs(parent_params.unsqueeze(2) - child_params) + child_activity, dim=2)[0]
        return activity[:, (3 ** self.arity - 1) // 2, :]

    def project(self):
        remaining_var = torch.ones_like(self.params)
        edge_tensor_pairs = self._compute_neighbor_graph()
        self.params.data = torch.clamp(self.params.data, min=-1.0, max=1.0)
        for num_zeros in range(1, self.arity + 1):
            parent_ids, child_ids = edge_tensor_pairs[num_zeros]
            parent_params = self.params[:, parent_ids, :]
            child_params = self.params[:, child_ids, :]
            child_var = remaining_var[:, child_ids, :]
            upper_lim = torch.min(child_params + child_var, dim=2)[0] / 2.0
            lower_lim = torch.max(child_params - child_var, dim=2)[0] / 2.0
            new_parent_params = torch.maximum(torch.minimum(parent_params, upper_lim), lower_lim)
            parent_var = torch.min(child_var - torch.abs(new_parent_params.unsqueeze(2) - child_params), dim=2)[0]
            self.params.data[:, parent_ids, :] = new_parent_params
            remaining_var[:, parent_ids, :] = parent_var


# # self = TameHighOrderActivationA(arity=2, input_groups=1, out_dim=1)
# self = TameHighOrderActivationB(arity=2, input_groups=1, out_dim=1)
# self.params.data = (torch.randint(0, 2, self.params.shape) * 2 - 1).to(torch.float32)
# # self.params.data = torch.rand_like(self.params) * 2 - 1
# self.params.data[:, (3 ** self.arity - 1) // 2, :] = 0.0
# print(self.params, self._compute_activity())
# self.project()
# print(self.params, self._compute_activity())

class Network(torch.nn.Module):
    def __init__(self,
                 widths: List[int],
                 bias_factor: float = 1.0,
                 act_factor: float = 1.0,
                 act_init: float = 1.0,
                 scale_factor: float = 1.0,
                 scale_init: float = 1.0,
                 arity: int = 2,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.widths = widths
        self.depth = len(widths) - 1
        # self.pen_scale = pen_scale
        self.scale_factor = scale_factor
        self.lin_layers = torch.nn.ModuleList([])
        self.act_layers = torch.nn.ModuleList([])
        # self.sparsifier_layers = torch.nn.ModuleList([])
        # self.scale = torch.nn.Parameter(torch.full([widths[-1]], scale_init / scale_factor, dtype=dtype, device=device))
        self.scale = torch.nn.Parameter(torch.full([], scale_init / scale_factor, dtype=dtype, device=device))
        self.bias = torch.nn.Parameter(torch.zeros([widths[-1]], dtype=dtype, device=device))
        for i in range(self.depth):
            # if i != self.depth - 1:
                # self.lin_layers.append(L1Linear(widths[i], widths[i + 1],
                #                                 pen_coef=pen_lin_coef, pen_exp=pen_lin_exp,
                #                                 dtype=dtype, device=device))
                # self.act_layers.append(ReLU())

                # self.lin_layers.append(L1LinearScaled(widths[i], widths[i + 1],
                self.lin_layers.append(L1Linear(widths[i], widths[i + 1] * arity,
                                                bias_factor=bias_factor,
                                                dtype=dtype, device=device))
                # self.lin_layers.append(LpLinear(widths[i], widths[i + 1] * arity,
                #                                 bias_factor=bias_factor,
                #                                 eps=0.1, p=1.0,
                #                                 dtype=dtype, device=device,
                #                                 num_iters=6))
                # self.lin_layers.append(Shuffle(widths[i], widths[i + 1] * arity))

                # self.lin_layers.append(OrthogonalButterfly(widths[i], widths[i + 1] * arity,
                #                                            width_pow=butterfly_width_pow,
                #                                            depth=butterfly_depth,
                #                                            l2_interact=0.0, dtype=dtype, device=device))
                # self.act_layers.append(GeneralizedMaxout(arity=arity, terms=2, output_width=widths[i + 1], bias_factor=bias_factor))
                # self.act_layers.append(ClampedGeneralizedMaxout(arity=arity, terms=2, output_width=widths[i + 1], bias_factor=bias_factor))
                # self.act_layers.append(GateActivation(widths[i + 1], pen_act=0.0))
                # self.act_layers.append(PassthroughGateActivation(widths[i + 1] // 2, pen_act=0.0))
                # self.act_layers.append(ParametricGateActivation(widths[i + 1], pen_act))
                # self.act_layers.append(D2Activation(widths[i + 1], act_factor=act_factor))
                # self.act_layers.append(B2Activation(widths[i + 1], act_factor=act_factor))
                # self.act_layers.append(ClampedB2Activation(widths[i + 1], act_factor=act_factor, pen_act=1.0))
                # self.act_layers.append(ReLU())
                # self.act_layers.append(DoubleGateActivation(widths[i + 1]))
                # self.act_layers.append(B2ActivationModified(widths[i + 1]))
                # self.act_layers.append(B2ActivationNonnegative(widths[i + 1]))
                # self.act_layers.append(A1Activation(widths[i + 1]))
                # self.act_layers.append(ClampedD2Activation(widths[i + 1]))
                # self.act_layers.append(MonotoneA1Activation(widths[i + 1]))
                # self.act_layers.append(ClampedMonotoneActivation(arity, widths[i + 1] // arity, arity, act_factor=act_factor))
                # self.act_layers.append(ClampedMonotoneActivation(arity, widths[i + 1], 1, act_factor=act_factor))
                # self.act_layers.append(MonotoneActivation(arity, widths[i + 1] // arity, arity, act_factor=act_factor))
                # self.act_layers.append(MonotoneActivation(arity, widths[i + 1], 1, act_factor=act_factor))
                # self.act_layers.append(MinOut(arity=arity))
                # self.act_layers.append(ClampedMaxOut(arity=arity, target_act=0.0, pen_act=0.0))
                # self.act_layers.append(ClampedMinOut(arity=arity, target_act=target_act, pen_act=pen_act))
                # self.act_layers.append(GateAbsActivation(widths[i + 1], pen_act=pen_act))
                # self.act_layers.append(CoherentSignActivation(widths[i + 1], target_act=target_act, pen_act=pen_act))
                # self.act_layers.append(HighOrderActivationB(arity, widths[i + 1] // arity, arity))
                # self.act_layers.append(HighOrderActivationB(arity, widths[i + 1], 1, act_init=act_init, act_factor=act_factor))
                # self.act_layers.append(CornerActivationB(arity, widths[i + 1], 1, act_init=act_init, act_factor=act_factor))
                self.act_layers.append(TameHighOrderActivationB(arity, widths[i + 1], 1))
                # self.act_layers.append(SparseHighOrderActivationB(arity, widths[i + 1] // arity, arity))
                # self.act_layers.append(SparseHighOrderActivationB(arity, widths[i + 1], 1))
                # self.act_layers.append(HighOrderActivation(arity, widths[i + 1], 1))
                # self.act_layers.append(TameHighOrderActivationA(arity, widths[i + 1], 1))
                # self.act_layers.append(HighOrderActivation(arity, widths[i + 1] // arity, arity))
                # self.act_layers.append(PReLU(widths[i + 1]))
                # self.act_layers.append(Top1Activation(widths[i + 1], 4))
                # self.sparsifier_layers.append(TopKContSparsifier(widths[i + 1], top_k))
                # self.act_layers.append(RestrictedHighOrderActivationA(arity, widths[i + 1] // arity, arity))
                # self.act_layers.append(RestrictedHighOrderActivationB(arity, widths[i + 1] // arity, arity))
                # self.act_layers.append(RestrictedHighOrderActivationB(arity, widths[i + 1], 1))
        # else:
        #         # self.lin_layers.append(LpLinear(widths[i], widths[i + 1],
        #         #                                 bias_factor=bias_factor,
        #         #                                 eps=0.1, p=1.0,
        #         #                                 dtype=dtype, device=device,
        #         #                                 num_iters=12))
        #         self.lin_layers.append(L1Linear(widths[i], widths[i + 1], # * arity,
        #                                         bias_factor=bias_factor,
        #                                         dtype=dtype, device=device))

    def forward(self, X):
        self.cnt_rows = X.shape[0]
        overall_scale = torch.abs(self.scale) * self.scale_factor
        layer_scale = overall_scale ** (1 / self.depth)
        for i in range(self.depth):
            X = X * layer_scale
            X = self.lin_layers[i](X)
            # X = self.lin_layers[i](X, layer_scale)
            # if i != self.depth - 1:
            X = self.act_layers[i](X)
        out = X + self.bias.unsqueeze(0)
        return out

    def penalty(self):
        return 0.0
        # return sum(layer.penalty() for layer in self.act_layers) + \
        #        sum(layer.penalty() for layer in self.lin_layers) # + \
               # self.cnt_rows * self.pen_scale * torch.abs(self.scale * self.scale_factor)
        # return sum(layer.penalty() for layer in self.lin_layers) + \
        #     self.pen_scale * torch.sum((self.scale * self.scale_factor) ** 2)

