import torch
# from tame_pytorch import approx_simplex_projection

def weighted_simplex_projection_iteration(x0: torch.tensor, y: torch.tensor, w: torch.tensor, K: float, dim: int):
    mask = (x0 > 0).to(x0.dtype)
    w_mask = w * mask
    w2_sum = torch.sum(w_mask ** 2, dim=dim)
    y_sum = torch.sum(y * w_mask, dim=dim)
    t = (y_sum - K) / w2_sum
    x1 = y - t.unsqueeze(dim=dim) * w
    return torch.clamp(x1, min=0.0)


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
#
# # cost_fn, cost_grad = soft_lp_cost_grad(1e-2, 0.8)
# cost_fn, cost_grad = soft_lp_cost_grad(1e-2, 1.0)
#
# y = torch.rand([8])
# x0 = torch.full_like(y, 1e-15)
# K = 1.0
# dim = 0
# print(y)
# x1 = x0
# for _ in range(10):
#     x1 = concave_projection_iteration(x1, y, K, cost_fn, cost_grad, dim)
#     print(x1, torch.sum(cost_fn(x1)))

# print(approx_simplex_projection(y, dim, 10))