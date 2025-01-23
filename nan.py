# import torch
# import torch.optim as optim
# import torch.nn as nn
# from otdd.pytorch.datasets import load_torchvision_data
# from otdd.pytorch.method_linear_gaussian import compute_pairwise_distance
# from otdd.pytorch.distance import DatasetDistance
# from otdd.pytorch.moments import compute_label_stats

# import os
# import torch.nn.functional as F

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression

# from scipy import stats
# import json

# from trainer import *


# mnist_dataloader = load_torchvision_data("MNIST", valid_size=0, resize=28, download=False, maxsize=None, maxsize_for_each_class=None)[0]["train"]
# kmnist_dataloader = load_torchvision_data("KMNIST", valid_size=0, resize=28, download=False, maxsize=None, maxsize_for_each_class=None)[0]["train"]


# list_dataset = [mnist_dataloader, kmnist_dataloader]
# DEVICE = "cpu"
# num_projection = 10000
# kwargs = {
#     "dimension": 28 * 28,
#     "num_channels": 1,
#     "precision": "float",
#     "p": 2,
#     "chunk": 1000
# }

# sw_list, time_process = compute_pairwise_distance(list_D=list_dataset, device=DEVICE, num_projections=num_projection, evaluate_time=True, **kwargs)
# print(sw_list, time_process)

# num_projections = 3
# flatten_dim = 2
# A = torch.randn(num_projections, flatten_dim, flatten_dim)
# log_cov = torch.randn(5, flatten_dim, flatten_dim)
# lon = A.unsqueeze(0) * log_cov.unsqueeze(1)
# print(lon.shape)


import torch

# Define sample input data
num_classes = 3  # Number of classes
flatten_dim = 4  # Dimensionality of the flattened space
num_projections = 5  # Number of projections

# Generate random samples for the inputs
mean = torch.randn(num_classes, flatten_dim)  # Shape: (num_classes, flatten_dim)
cov = torch.randn(num_classes, flatten_dim, flatten_dim)  # Shape: (num_classes, flatten_dim, flatten_dim)
w = torch.randn(num_projections, 2)  # Shape: (num_projections, 2), normalized weights
theta = torch.randn(num_projections, flatten_dim)  # Shape: (num_projections, flatten_dim)

# Function to test
def _project_distribution(mean, cov, w, theta):
    """
    mean has shape R^(num_cls, flatten_dim)
    cov has shape R^(num_cls, flatten_dim, flatten_dim)
    w has shape R^(L, 2)
    theta has shape R^(L, flatten_dim)
    A has shape R^(L, flatten_dim, flatten_dim)
    """
    num_classes = cov.shape[0]
    flatten_dim = cov.shape[1]
    num_projections = w.shape[0]
    projected_mean = torch.matmul(mean, theta.transpose(0, 1))
    projected_cov = torch.sum(torch.matmul(cov, theta.transpose(0, 1)) * theta.transpose(0, 1), dim=1)

    projected_cov_2 = torch.matmul(torch.matmul(theta.unsqueeze(0), cov).unsqueeze(2), theta.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    projected_distributions = torch.stack([projected_mean, torch.log(projected_cov)], dim=-1)
    print(projected_distributions.shape)
    avg_projected_distributions = torch.sum(projected_distributions * w, dim=-1)
    return avg_projected_distributions




# Test the function
output = _project_distribution(mean, cov, w, theta)
print(output.shape)
