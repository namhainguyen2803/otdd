import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data, load_imagenet
from otdd.pytorch.method5 import compute_pairwise_distance
from otdd.pytorch.distance import DatasetDistance

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
from models.resnet import ResNet50
import random
from datetime import datetime, timedelta
import time
from torch.utils.data import Dataset, DataLoader

from otdd.pytorch.utils import *
from otdd.pytorch.utils import generate_and_plot_data


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

saved_path = "saved_augmentation_2/aug_1"

class CustomTensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_dataloader(datadir, maxsize=None, batch_size=64):
    images_tensor, labels_tensor = torch.load(datadir)
    if maxsize is not None:
        if maxsize < images_tensor.size(0):
            indices = torch.randperm(images_tensor.size(0))[:maxsize]
            selected_images = images_tensor[indices]
            selected_labels = labels_tensor[indices]
            dataset = CustomTensorDataset(selected_images, selected_labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            dataset = CustomTensorDataset(images_tensor, labels_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        dataset = CustomTensorDataset(images_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


cifar10_dataloader = get_dataloader(datadir=f'{saved_path}/transformed_test_cifar10.pt', maxsize=1000, batch_size=64)
imagenet_dataloader = get_dataloader(datadir=f'{saved_path}/transformed_test_imagenet.pt', maxsize=1000, batch_size=64)
dataloaders = [cifar10_dataloader, imagenet_dataloader]


kwargs = {
    "dimension": 32,
    "num_channels": 3,
    "num_moments": 4,
    "use_conv": True,
    "precision": "float",
    "p": 2,
    "chunk": 1000
}

list_dist = list()
for i in range(10):
    list_pairwise_dist, sotdd_time_taken = compute_pairwise_distance(list_D=dataloaders, num_projections=10000, device=DEVICE, evaluate_time=True, **kwargs)
    dist = list_pairwise_dist[0]
    list_dist.append(dist)
    print(dist)
print(list_dist)





# for x, y in cifar10_dataloader:
#     img, label = x, y
#     break
# print(img.shape, img.min(), img.max())

# U_list = generate_unit_convolution_projections()


# proj_img = img
# for conv in U_list:
#     proj_img = conv(proj_img).detach()
# proj_img = proj_img.squeeze(-1).squeeze(-1)

# print(proj_img.shape, proj_img.min(), proj_img.max())

# max_idx = torch.argmax(proj_img)

# row_idx = max_idx // proj_img.shape[1]
# col_idx = max_idx % proj_img.shape[1]

# print(proj_img[row_idx, col_idx])

# img_idx = row_idx
# generate_and_plot_data(img[img_idx], "cac1.png")
# print(proj_img[img_idx].shape, proj_img[img_idx].min(), proj_img[img_idx].max())
# generate_and_plot_data(proj_img[img_idx], "cac2.png")