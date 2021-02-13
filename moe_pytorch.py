from typing import List
import torch


def split_distinct(arr):
    """Given an input 1D tensor, return two lists of tensors, one list giving the unique values in the
    input, and the other the list of tensors each of which gives the indices where the corresponding
    value occurs.

    This implementation involves sorting the input, which shouldn't be required, so it should be possible
    to implement this more efficiently some other way, e.g. using a custom kernel."""
    assert len(arr.shape) == 1
    vals, indices = torch.sort(arr)
    split_points = torch.nonzero(vals[1:] != vals[:-1])[:, 0] + 1
    split_sizes = [split_points[0].item(),
                   *(split_points[1:] - split_points[:-1]).tolist(),
                   (arr.shape[0] - split_points[-1]).item()]
    indices_list = torch.split(indices, split_sizes)
    # indices_list = torch.tensor_split(indices, list(split_points + 1))  # Use this once `tensor_split` is stable
    values_list = [vals[0].item(), *vals[split_points].tolist()]
    return values_list, indices_list


class MixtureOfExperts(torch.nn.Module):
    def __init__(self, experts: List[torch.nn.Module], selection_size: int):
        super().__init__()
        self.experts = experts
        self.selection_size = selection_size

    def forward(self, X: torch.tensor, G: torch.tensor):
        assert len(X.shape) == 2
        assert len(G.shape) == 2
        assert X.shape[0] == G.shape[0]
        assert G.shape[1] == len(self.experts)
        n_rows = X.shape[0]
        raw_weights = torch.softmax(G, dim=1)
        top_weights, top_indices = torch.topk(raw_weights, self.selection_size, dim=1)
        normalized_weights = top_weights / torch.sum(top_weights, dim=1, keepdim=True)
        split_expert_ids, split_ind = split_distinct(top_indices.view(-1))
        split_row_nums = [p // self.selection_size for p in split_ind]
        split_X = [X[r, :] for r in split_row_nums]
        split_Y = [self.experts[split_expert_ids[i]](Xi) for i, Xi in enumerate(split_X)]
        cat_Y = torch.cat(split_Y, dim=0)
        cat_split_ind = torch.cat(split_ind)
        expert_Y = torch.zeros([n_rows * self.selection_size, cat_Y.shape[1]], dtype=X.dtype, device=X.device)
        expert_Y[cat_split_ind, :] = cat_Y
        Y = torch.einsum('ij,ijk->ik', normalized_weights, expert_Y.view(n_rows, self.selection_size, cat_Y.shape[1]))
        return Y


# arr = torch.randint(high=10, size=[10])
N = 6
in_dim = 3
out_dim = 4
num_experts = 5
selection_size = 2
X = torch.randn(N, in_dim)
G = torch.randn(N, num_experts)
experts = [torch.nn.Linear(in_dim, out_dim) for _ in range(num_experts)]
moe = MixtureOfExperts(experts, selection_size)
Y = moe(X, G)