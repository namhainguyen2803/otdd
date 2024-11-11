import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from wte.distance import WTE
from otdd.pytorch.datasets import *
from scipy.spatial import distance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_reference(num, dim_low, dim, attached_dim, seed=0):
    torch.manual_seed(seed)
    med = torch.rand(num, dim_low, dim_low).unsqueeze(0)
    s = dim/dim_low
    m = nn.Upsample(scale_factor=s, mode='bilinear')
    attached = torch.randn(num, attached_dim)
    return torch.cat((m(med).reshape(num, -1), attached), dim=1).float()

MAXSIZE = 60000
dataset_mnist  = load_torchvision_data('MNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)
dataset_kmnist  = load_torchvision_data('KMNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)
dataset_emnist  = load_torchvision_data('EMNIST', valid_size=0, resize = 28, download=True, maxsize=MAXSIZE)
dataset_fmnist  = load_torchvision_data('FashionMNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)
dataset_usps  = load_torchvision_data('USPS',  valid_size=0, resize = 28, download=False, maxsize=MAXSIZE, datadir="data/USPS")

reference = generate_reference(60000, 4, 28, 10)
train_all = [dataset_mnist[1]['train'], dataset_kmnist[1]['train'], dataset_emnist[1]['train'], dataset_fmnist[1]['train'], dataset_usps[1]['train']]
wtes = WTE(train_all, label_dim=10, device=DEVICE, ref=reference.cpu(), maxsamples=60000)
wtes = wtes.reshape(wtes.shape[0], -1)
wte_distance = distance.cdist(wtes, wtes, 'euclidean')

print(wte_distance)