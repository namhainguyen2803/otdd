import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.method_gaussian import compute_pairwise_distance
from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.moments import compute_label_stats

import os
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from scipy import stats
import json

from trainer import *


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

num_projections = 3
flatten_dim = 2
A = torch.randn(num_projections, flatten_dim, flatten_dim)
log_cov = torch.randn(5, flatten_dim, flatten_dim)
lon = A.unsqueeze(0) * log_cov.unsqueeze(1)
print(lon.shape)
# projected_cov = (A[None] * log_cov[:, None]).reshape(flatten_dim, num_projections, -1).sum(-1)

# print(projected_cov)