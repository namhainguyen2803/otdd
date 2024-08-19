import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.method import NewDatasetDistance
from otdd.pytorch.distance import DatasetDistance

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

import json

from trainer import *

LIST_DATASETS = ["MNIST", "FMNIST", "EMNIST", "KMNIST", "USPS"]
ACC_ADAPT = dict()
DIST = dict()

# Load data
MAXSIZE = 8000
loaders_mnist  = load_torchvision_data('MNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)[0]
loaders_kmnist  = load_torchvision_data('KMNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)[0]
loaders_emnist  = load_torchvision_data('EMNIST', valid_size=0, resize = 28, download=True, maxsize=MAXSIZE)[0]
loaders_fmnist  = load_torchvision_data('FashionMNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)[0]
loaders_usps  = load_torchvision_data('USPS',  valid_size=0, resize = 28, download=False, maxsize=MAXSIZE, datadir="data/USPS")[0]


def compute_distance_dataset(name_src, name_tgt, maxsamples=MAXSIZE, num_projection=5000):
    if name_src == "MNIST":
        loaders_src = loaders_mnist
    elif name_src == "KMNIST":
        loaders_src = loaders_kmnist
    elif name_src == "EMNIST":
        loaders_src = loaders_emnist
    elif name_src == "FMNIST":
        loaders_src = loaders_fmnist
    elif name_src == "USPS":
        loaders_src = loaders_usps
    else:
        raise("Unknown src dataset")

    if name_tgt == "MNIST":
        loaders_tgt = loaders_mnist
    elif name_tgt == "KMNIST":
        loaders_tgt = loaders_kmnist
    elif name_tgt == "EMNIST":
        loaders_tgt = loaders_emnist
    elif name_tgt == "FMNIST":
        loaders_tgt = loaders_fmnist
    elif name_tgt == "USPS":
        loaders_tgt = loaders_usps
    else:
        raise("Unknown tgt dataset")

    # dist = NewDatasetDistance(loaders_src['train'], loaders_tgt['train'], p=2, device='cpu')
    # d = dist.distance(maxsamples=maxsamples, num_projection=num_projection)

    dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'],
                            inner_ot_method = 'exact',
                            debiased_loss = True,
                            p = 2, entreg = 1e-1,
                            device='cpu')
    d = dist.distance(maxsamples = maxsamples)

    print(f'DIST({name_src}, {name_tgt})={d:8.2f}')
    return d


def compute_pairwise_distance():
    all_dist_dict = dict()
    for i in range(len(LIST_DATASETS)):
        all_dist_dict[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            all_dist_dict[LIST_DATASETS[i].upper()][LIST_DATASETS[j].upper()] = 0

    for i in range(len(LIST_DATASETS)):
        for j in range(i + 1, len(LIST_DATASETS)):
            dist = compute_distance_dataset(LIST_DATASETS[i].upper(), LIST_DATASETS[j].upper()).item()
            all_dist_dict[LIST_DATASETS[i].upper()][LIST_DATASETS[j].upper()] = dist
            all_dist_dict[LIST_DATASETS[j].upper()][LIST_DATASETS[i].upper()] = dist
    return all_dist_dict

DIST = compute_pairwise_distance()

dist_file_path = 'saved/dist3.json'
with open(dist_file_path, 'w') as json_file:
    json.dump(DIST, json_file, indent=4)
print(f"DIST: {DIST}")
