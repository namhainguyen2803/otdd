import numpy as np 
import torch


def sample_zero_truncated_poisson(rate_vector):
    rate_vector = rate_vector.to(dtype=torch.float)
    u = torch.rand_like(rate_vector) * (1 - torch.exp(-rate_vector)) + torch.exp(-rate_vector)
    t = -torch.log(u)
    return 1 + torch.poisson(rate_vector - t)


def generate_moments(num_moments):
    # mean_moment = torch.tensor([0.01, 2, 3, 4, 5])
    mean_moment = torch.tensor([1, 2, 3, 4, 5])
    moment = sample_zero_truncated_poisson(mean_moment)
    return moment

# cac = generate_moments(5)
# print(cac)

# data, label = torch.load("saved_text_data_2/AmazonReviewPolarity_text.pt")
# print(data.shape, label.shape)
# print(data.min(), data.max())

# data_mean = data.mean()
# data_std = data.std()
# normalized_data = (data - data_mean) / data_std

# print(normalized_data.min(), normalized_data.max())