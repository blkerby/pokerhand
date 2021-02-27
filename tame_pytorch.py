import torch
from typing import List
import logging

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
                 dtype=torch.float32, device=None,
                 num_iters=6):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.pen_coef = pen_coef
        self.pen_exp = pen_exp
        self.bias_factor = bias_factor
        self.weights_pos_neg = Simplex([input_width * 2, output_width], dim=0, dtype=dtype, device=device,
                                       num_iters=num_iters)

        # self.active_input = torch.randint(0, input_width * 2, [output_width])
        # self.weights_pos_neg.param.data[:, :] = 0.0
        # self.weights_pos_neg.param.data[self.active_input, torch.arange(output_width)] = 1.0
        self.bias = torch.nn.Parameter(torch.zeros([output_width], dtype=dtype, device=device))

    def forward(self, X):
        assert X.shape[1] == self.input_width
        self.cnt_rows = X.shape[0]
        weights = self.weights_pos_neg.param[:self.input_width, :] - self.weights_pos_neg.param[self.input_width:, :]
        return torch.matmul(X, weights) + self.bias.view(1, -1) * self.bias_factor

    def penalty(self):
        w = self.weights_pos_neg.param
        pen_weight = self.pen_coef * self.cnt_rows * torch.mean(1 - (1 - w) ** self.pen_exp - w)
        return pen_weight


# class ReLU(torch.nn.Module):
#     def __init__(self,
#                  pen_act: float,
#                  target_act: float):
#         super().__init__()
#         self.pen_act = pen_act
#         self.target_act = target_act
#
#     def forward(self, X):
#         self.X = X
#         if self.X.requires_grad:
#             self.X.retain_grad()
#         out = torch.clamp(X, min=0.0)
#         self.cnt_rows = X.shape[0]
#         return out
#
#     def penalty(self):
#         return self.pen_act * torch.sum(
#             torch.clamp(self.X, min=0.0) * (1 - self.target_act) - torch.clamp(self.X, max=0.0) * self.target_act)
#
#


class MinOut(torch.nn.Module):
    def __init__(self, arity):
        super().__init__()
        self.arity = arity

    def forward(self, X):
        self.X = X
        X = X.view([X.shape[0], self.arity, -1])
        out = torch.clamp(torch.min(X, dim=1)[0], min=0.0)
        return out

    def penalty(self):
        return 0.0


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
        return out

    def project(self):
        self.params.data = torch.clamp(self.params.data, min=-1.0, max=1.0)

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


class Network(torch.nn.Module):
    def __init__(self,
                 widths: List[int],
                 pen_lin_coef: float = 0.0,
                 pen_lin_exp: float = 2.0,
                 pen_scale: float = 0.0,
                 scale_init: float = 0.0,
                 scale_factor: float = 1.0,
                 bias_factor: float = 1.0,
                 arity: int = 2,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.widths = widths
        self.depth = len(widths) - 1
        self.pen_scale = pen_scale
        self.scale_factor = scale_factor
        self.lin_layers = torch.nn.ModuleList([])
        self.act_layers = torch.nn.ModuleList([])
        # self.scale = torch.nn.Parameter(torch.full([widths[-1]], scale_init / scale_factor, dtype=dtype, device=device))
        self.scale = torch.nn.Parameter(torch.full([], scale_init / scale_factor, dtype=dtype, device=device))
        for i in range(self.depth):
            if i != self.depth - 1:
                # self.lin_layers.append(L1Linear(widths[i], widths[i + 1],
                #                                 pen_coef=pen_lin_coef, pen_exp=pen_lin_exp,
                #                                 dtype=dtype, device=device))
                # self.act_layers.append(torch.nn.ReLU())

                self.lin_layers.append(L1Linear(widths[i], widths[i + 1] * arity,
                                                pen_coef=pen_lin_coef, pen_exp=pen_lin_exp,
                                                bias_factor=bias_factor,
                                                dtype=dtype, device=device))
                # self.act_layers.append(D2Activation(widths[i + 1]))
                # self.act_layers.append(MonotoneA1Activation(widths[i + 1]))
                self.act_layers.append(MonotoneActivation(arity, widths[i + 1], 1))
                # self.act_layers.append(MinOut(arity))
            else:
                self.lin_layers.append(L1Linear(widths[i], widths[i + 1],
                                                pen_coef=pen_lin_coef, pen_exp=pen_lin_exp,
                                                dtype=dtype, device=device))

    def forward(self, X):
        overall_scale = self.scale * self.scale_factor
        layer_scale = overall_scale ** (1 / self.depth)
        for i in range(self.depth):
            X = X * layer_scale
            X = self.lin_layers[i](X)
            if i != self.depth - 1:
                X = self.act_layers[i](X)
        return X

    def penalty(self):
        return sum(layer.penalty() for layer in self.lin_layers) + \
            self.pen_scale * (self.scale * self.scale_factor) ** 2
        # return sum(layer.penalty() for layer in self.lin_layers) + \
        #     self.pen_scale * torch.sum((self.scale * self.scale_factor) ** 2)

