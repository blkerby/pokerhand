"""PyTorch version of high-order activation"""

import torch


def fast_high_order_act(A, params):
    A_sort, A_ind = torch.sort(A, dim=2)
    A_diff = A_sort[:, :, 1:] - A_sort[:, :, :-1]
    coef = torch.cat([A_sort[:, :, 0:1], A_diff], dim=2)
    params_A_ind = torch.flip(torch.cumsum(torch.flip(2 ** A_ind, dims=[2]), dim=2), dims=[2])
    ind0 = torch.unsqueeze(torch.unsqueeze(torch.arange(0, params.shape[0], dtype=torch.int64), 1), 2)
    ind1 = torch.transpose(params_A_ind, 0, 1)
    params_gather = params[ind0, ind1, :]
    out = torch.einsum('jikl,ijk->ijl', params_gather, coef)
    return out


class HighOrderActivation(torch.nn.Module):
    def __init__(self, arity, input_groups, out_dim):
        super().__init__()
        self.arity = arity
        self.input_groups = input_groups
        self.out_dim = out_dim
        self.params = torch.nn.Parameter(torch.randn([input_groups, 2 ** arity, out_dim]))

    def forward(self, X):
        assert len(X.shape) == 3
        assert X.shape[1] == self.input_groups
        assert X.shape[2] == self.arity
        return fast_high_order_act(X, self.params)


# m = 4
# n = 3
# k = 2
# params = torch.randn([k, 2 ** n, 5])
# # A = torch.randn([m, k, n])
# A = torch.tensor([
#     [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
#     [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#     [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
#     [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
# ])
# out = fast_high_order_act(A, params)
