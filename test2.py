import numpy as np 
import torch


def sample_zero_truncated_poisson(rate_vector):
    rate_vector = rate_vector.to(dtype=torch.float)
    u = torch.rand_like(rate_vector) * (1 - torch.exp(-rate_vector)) + torch.exp(-rate_vector)
    t = -torch.log(u)
    return 1 + torch.poisson(rate_vector - t)


def generate_moments(num_moments):
    mean_moment = torch.tensor([0.01, 2, 3, 4, 5])
    moment = torch.sort(sample_zero_truncated_poisson(mean_moment))[0]
    return moment




print(generate_moments(1))