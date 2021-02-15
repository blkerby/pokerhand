from typing import Optional
import torch
import math

def sparsify(A, k):
    n_row = A.shape[0]
    n_col = A.shape[1]
    vals, indices = torch.topk(A, k, dim=1, sorted=False)
    coo_row_ind = torch.arange(n_row).view(-1, 1).repeat([1, k])
    coo_indices = torch.stack([coo_row_ind.view(-1), indices.view(-1)], dim=0)
    out = torch.sparse_coo_tensor(coo_indices, vals.view(-1), size=[n_row, n_col])
    return out


def sparsify_abs(A, k):
    n_row = A.shape[0]
    n_col = A.shape[1]
    _, indices = torch.topk(torch.abs(A), k, dim=1, sorted=False)
    coo_row_ind = torch.arange(n_row).view(-1, 1).repeat([1, k])
    coo_indices = torch.stack([coo_row_ind.view(-1), indices.view(-1)], dim=0)
    vals = A[torch.arange(n_row).view(-1, 1), indices]
    out = torch.sparse_coo_tensor(coo_indices, vals.view(-1), size=[n_row, n_col])
    return out


class SparseLinear(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 sparse_input: bool = False,
                 sparse_output: Optional[int] = None,
                 sparse_backprop: Optional[int] = None,
                 sparse_weights: Optional[int] = None,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sparse_input = sparse_input
        self.sparse_output = sparse_output
        self.sparse_backprop = sparse_backprop
        self.sparse_weights = sparse_weights
        weight_norm = math.sqrt(in_dim if self.sparse_output is None else self.sparse_output)
        self.weight = torch.nn.Parameter(torch.randn([in_dim, out_dim], dtype=dtype, device=device) / weight_norm)
        self.bias = torch.nn.Parameter(torch.zeros([out_dim], dtype=dtype, device=device))

    def forward(self, X):
        # TODO: Handle sparse inputs, weights, and backprop (like meProp)
        Y = torch.matmul(X, self.weight) + self.bias
        if self.sparse_output is not None:
            return sparsify_abs(Y, self.sparse_output).to_dense()  # TODO: Keep in sparse form
        else:
            return Y


# A = torch.randn([5, 4])
# k = 2
