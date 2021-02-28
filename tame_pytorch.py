import torch
from typing import List
import logging
import math
from high_order_act_pytorch import HighOrderActivation, HighOrderActivationB

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
    def __init__(self, input_width, output_width,
                 pen_coef=0.0, pen_exp=2.0,
                 bias_factor=1.0,
                 # scale_factor=1.0,
                 dtype=torch.float32, device=None,
                 num_iters=6):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.pen_coef = pen_coef
        self.pen_exp = pen_exp
        self.bias_factor = bias_factor
        # self.scale_factor = scale_factor
        self.weights_pos_neg = Simplex([input_width * 2, output_width], dim=0, dtype=dtype, device=device,
                                       num_iters=num_iters)

        # self.active_input = torch.randint(0, input_width * 2, [output_width])
        # self.weights_pos_neg.param.data[:, :] = 0.0
        # self.weights_pos_neg.param.data[self.active_input, torch.arange(output_width)] = 1.0
        self.bias = torch.nn.Parameter(torch.zeros([output_width], dtype=dtype, device=device))
        # self.scale = torch.nn.Parameter(torch.ones([output_width], dtype=dtype, device=device))

    def forward(self, X):
        assert X.shape[1] == self.input_width
        self.cnt_rows = X.shape[0]
        weights = self.weights_pos_neg.param[:self.input_width, :] - self.weights_pos_neg.param[self.input_width:, :]
        return torch.matmul(X, weights) + self.bias.view(1, -1) * self.bias_factor
        # return torch.matmul(X, weights) * (self.scale.view(1, -1) * self.scale_factor) + self.bias.view(1, -1) * self.bias_factor

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


class D2Activation(ManifoldModule):
    """Activation based on the D_2 root system. The lines y = x and y = -x partition R^2 into the four Weyl chambers,
    and for each 2-input unit, this activation function can be an arbitrary continuous function on R^2 which is linear
    on each of the four chambers subject to the "tame" constraint |df/dx| + |df/dy| <= 1; such a function is uniquely
    determined by its values on the four points (-1, -1), (-1, 1), (1, -1), (1, 1), which can be any values in the
    interval [-1, 1]."""

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
        self.out = out
        return out

    def project(self):
        self.params.data = torch.clamp(self.params.data, min=-1.0, max=1.0)

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
        lower_lim = (self.params.data[:, 1] + 1) / 2
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
    def __init__(self, arity, input_groups, out_dim):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.params = torch.nn.Parameter(torch.randint(0, 2, [input_groups, 2 ** arity, out_dim]).to(torch.float32))
        self.project()
        self.params.data = torch.round(self.params.data)

    def forward(self, X):
        assert len(X.shape) == 2
        assert X.shape[1] == self.input_groups * self.arity
        X1 = X.view(X.shape[0], self.input_groups, self.arity)
        out1 = high_order_act(X1, self.params)
        self.out = out1
        return out1.view(X.shape[0], self.input_groups * self.out_dim)

    def project(self):
        # Compute the greatest monotone lower bound and least monotone upper bound and average them.
        # Theoretically it would be better to use the Euclidean projection but that would be more
        # difficult to compute.
        clamped = torch.clamp(self.params, min=0.0, max=1.0)
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
        self.params.data[:, 2 ** self.arity - 1, :] = 1.0

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
        # out = torch.min(torch.clamp(G, min=0.0), torch.abs(Y)) * torch.sgn(Y)
        out = torch.maximum(torch.minimum(Y, G), -G)
        self.out = out
        return out

    def penalty(self):
        return self.pen_act * torch.sum(torch.abs(self.out))
        # return self.pen_act * torch.sum(self.G)


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


# self = TopKContSparsifier(8, 8.0)
# X = torch.tensor([[1.0, 9.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
#                   [40.0, 45.0, 30.0, 40.0, 50.0, 0.0, -100.0, 80.0]])
# self(X)


class Network(torch.nn.Module):
    def __init__(self,
                 widths: List[int],
                 pen_lin_coef: float = 0.0,
                 pen_lin_exp: float = 2.0,
                 pen_scale: float = 0.0,
                 pen_act: float = 0.0,
                 target_act: float = 0.0,
                 scale_init: float = 1.0,
                 scale_factor: float = 1.0,
                 bias_factor: float = 1.0,
                 arity: int = 8,
                 top_k: float = 8,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.widths = widths
        self.depth = len(widths) - 1
        self.pen_scale = pen_scale
        self.scale_factor = scale_factor
        self.lin_layers = torch.nn.ModuleList([])
        self.act_layers = torch.nn.ModuleList([])
        self.sparsifier_layers = torch.nn.ModuleList([])
        # self.scale = torch.nn.Parameter(torch.full([widths[-1]], scale_init / scale_factor, dtype=dtype, device=device))
        self.scale = torch.nn.Parameter(torch.full([], scale_init / scale_factor, dtype=dtype, device=device))
        for i in range(self.depth):
            if i != self.depth - 1:
                # self.lin_layers.append(L1Linear(widths[i], widths[i + 1],
                #                                 pen_coef=pen_lin_coef, pen_exp=pen_lin_exp,
                #                                 dtype=dtype, device=device))
                # self.act_layers.append(ReLU())

                self.lin_layers.append(L1Linear(widths[i], widths[i + 1],
                                                pen_coef=pen_lin_coef, pen_exp=pen_lin_exp,
                                                # scale_factor=scale_factor,
                                                bias_factor=bias_factor,
                                                dtype=dtype, device=device))
                # self.act_layers.append(GateActivation(widths[i + 1], pen_act))
                # self.act_layers.append(D2Activation(widths[i + 1]))
                # self.act_layers.append(B2Activation(widths[i + 1]))
                # self.act_layers.append(A1Activation(widths[i + 1]))
                # self.act_layers.append(ClampedD2Activation(widths[i + 1]))
                # self.act_layers.append(MonotoneA1Activation(widths[i + 1]))
                self.act_layers.append(MonotoneActivation(arity, widths[i + 1] // arity, arity))
                # self.act_layers.append(MonotoneActivation(arity, widths[i + 1], 1))
                # self.act_layers.append(ClampedMinOut(arity=arity, target_act=target_act, pen_act=pen_act))
                # self.act_layers.append(HighOrderActivationB(2, widths[i + 1], 1))
                # self.act_layers.append(PReLU(widths[i + 1]))
                # self.act_layers.append(Top1Activation(widths[i + 1], 4))
                # self.sparsifier_layers.append(TopKContSparsifier(widths[i + 1], top_k))
            else:
                self.lin_layers.append(L1Linear(widths[i], widths[i + 1],
                                                pen_coef=pen_lin_coef, pen_exp=pen_lin_exp,
                                                # scale_factor=scale_factor,
                                                dtype=dtype, device=device))

    def forward(self, X):
        overall_scale = torch.abs(self.scale) * self.scale_factor
        layer_scale = overall_scale ** (1 / self.depth)
        for i in range(self.depth):
            X = X * layer_scale
            X = self.lin_layers[i](X)
            if i != self.depth - 1:
                X = self.act_layers[i](X)
                # X = self.sparsifier_layers[i](X)
        return X

    def penalty(self):
        # return 0.0
        return sum(layer.penalty() for layer in self.act_layers)
        #        self.pen_scale * (self.scale * self.scale_factor) ** 2
        # return sum(layer.penalty() for layer in self.lin_layers) + \
        #     self.pen_scale * torch.sum((self.scale * self.scale_factor) ** 2)
