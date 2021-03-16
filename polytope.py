# Try generating simplicial spheres that are approximately uniform

import torch
import logging

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler("polytope.log"),
                              logging.StreamHandler()])

d = 3
n = 8
A = torch.randn([n, d])
A = A / torch.norm(A, dim=1, keepdim=True)

def stable_norm(v, p):
    m = torch.max(torch.abs(v))
    v1 = v / m
    return m * torch.sum(v1 ** p) ** (1 / p)

def compute_obj(A):
    A = torch.cat([A, -A], dim=0)
    A = A / torch.norm(A, dim=1, keepdim=True)
    d2 = torch.sum((A.unsqueeze(0) - A.unsqueeze(1)) ** 2, dim=2)
    eps = 1e-15
    out = stable_norm(torch.triu(1 / (d2 + eps), diagonal=1), p=500.0)
    # out = torch.norm(torch.triu(1 / (d2 + eps), diagonal=1), p=100.0)
    # out = torch.max(torch.triu(1 / (d2 + eps), diagonal=1))
    return out

def compute_min_d2(A):
    A = torch.cat([A, -A], dim=0)
    A = A / torch.norm(A, dim=1, keepdim=True)
    d2 = torch.sum((A.unsqueeze(0) - A.unsqueeze(1)) ** 2, dim=2)
    out = 1 / torch.max(torch.triu(1 / d2, diagonal=1))
    return out

avg_beta = 0.99
A_avg = A.clone()
A.requires_grad = True
optimizer = torch.optim.Adam([A], lr=0.005, betas=(0.98, 0.98))


for iteration in range(50000):
    optimizer.zero_grad()
    obj = compute_obj(A)
    obj.backward()
    optimizer.step()
    A.data = A / torch.norm(A, dim=1, keepdim=True)
    A_avg = avg_beta * A_avg + (1 - avg_beta) * A.data
    if iteration % 100 == 0:
        with torch.no_grad():
            min_d2 = compute_min_d2(A)
            min_d2_avg = compute_min_d2(A_avg)
        logging.info("{}: obj={}, min_d2={}, min_d2_avg={}".format(iteration, obj, min_d2, min_d2_avg))

A = A_avg

d = 4
A = torch.tensor([
    # [1, 1, 1, 1],
    # [1, -1, 1, 1],
    # [1, 1, -1, 1],
    # [1, -1, -1, 1],
    # [1, 1, 1, -1],
    # [1, -1, 1, -1],
    # [1, 1, -1, -1],
    # [1, -1, -1, -1],
    # [1, 0, 0, 0],
    # [0, 1, 0, 0],
    # [0, 0, 1, 0],
    # [0, 0, 0, 1]
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [1, -1, 0, 0],
    [1, 0, -1, 0],
    [1, 0, 0, -1],
    [0, 1, -1, 0],
    [0, 1, 0, -1],
    [0, 0, 1, -1]
], dtype=torch.float32)

M = 100000
B = torch.randn([M, d])
B = B / torch.norm(B, dim=1, keepdim=True)
B_sgn = torch.sgn(torch.matmul(A, B.t()))

sgns_ctr_dict = {}
for r in range(M):
    s = B_sgn[:, r]
    t = tuple(s.tolist())
    if t not in sgns_ctr_dict:
        sgns_ctr_dict[t] = 0
    sgns_ctr_dict[t] += 1

sgns_ctr_list = list(sgns_ctr_dict.keys())

def generate_indices_rec(num_indices, start, end):
    if num_indices == 0:
        return [[]]
    out = []
    for i in range(start, end):
        base_indices = generate_indices_rec(num_indices - 1, i + 1, end)
        out.extend([[i] + b for b in base_indices])
    return out

def generate_vecs_rec(dim):
    if dim == 0:
        return [[]]
    base_vecs = generate_vecs_rec(dim - 1)
    return [[-1.0] + b for b in base_vecs] + [[1.0] + b for b in base_vecs]

print("minimum number of representatives: {}".format(min(sgns_ctr_dict.values())))
print("{} facets".format(len(sgns_ctr_list)))
all_indices = generate_indices_rec(d, 0, A.shape[0])
all_vecs = generate_vecs_rec(d)
result_dict = {}
for v in all_vecs:
    for ind in all_indices:
        cnt = 0
        result = None
        for j, s in enumerate(sgns_ctr_list):
            if all(s[ind[i]] == v[i] for i in range(d)):
                cnt += 1
                result = j
        if cnt == 1:
            result_dict[result] = ind
print("{} simplicial facets".format(len(result_dict)))

lines_set = set()
for cone in result_dict.values():
    for i in range(d):
        lines_set.add(tuple(cone[:i] + cone[(i+1):]))
print("{} vertices".format(2 * len(lines_set)))

