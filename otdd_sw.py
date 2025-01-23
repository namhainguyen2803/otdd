import torch
import time
import ot

import numpy as np
import matplotlib.pyplot as plt

from geoopt import Lorentz as Lorentz_geoopt

from hswfs_otdd.utils.hmds import HyperMDS
from hswfs_otdd.utils.datasets import *
from hswfs_otdd.utils.bures_wasserstein import LabelsBW

from hswfs_otdd.hswfs.manifold.euclidean import Euclidean
from hswfs_otdd.hswfs.manifold.product import ProductManifold
from hswfs_otdd.hswfs.manifold.poincare import Poincare
from hswfs_otdd.hswfs.manifold.lorentz import Lorentz
from hswfs_otdd.hswfs.sw import sliced_wasserstein

from tqdm import tqdm
from functools import partialmethod

from otdd.pytorch.datasets import *


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MAXSIZE = 500
dataset_mnist  = load_torchvision_data('MNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)
dataset_kmnist  = load_torchvision_data('KMNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)
# dataset_emnist  = load_torchvision_data('EMNIST', valid_size=0, resize = 28, download=True, maxsize=MAXSIZE)
# dataset_fmnist  = load_torchvision_data('FashionMNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)
# dataset_usps  = load_torchvision_data('USPS',  valid_size=0, resize = 28, download=False, maxsize=MAXSIZE, datadir="data/USPS")
train_all = [dataset_mnist[1]['train'], dataset_kmnist[1]['train']]



def otdd_hswfs(train_all, scaling=0.1, d=10, n_projs=500, device="cpu"):
    emb = LabelsBW(device=device, maxsamples=MAXSIZE)
    distance_array = emb.dissimilarity_for_all(train_all)
    lorentz_geoopt = Lorentz_geoopt()
    scaling = 0.1
    d = 10
    n_epochs = 500
    embedding = HyperMDS(d, lorentz_geoopt, torch.optim.Adam, scaling=scaling, loss="ads")
    mds, L = embedding.fit_transform(torch.tensor(distance_array, dtype=torch.float64), n_epochs=n_epochs, lr=1e-3)
    dist_mds = lorentz_geoopt.dist(mds[None], mds[:,None]).detach().cpu().numpy()
    diff_dist = np.abs(scaling * distance_array - dist_mds)
    data_X = [] # data
    data_Y = [] # labels
    for idx, dataset in enumerate(train_all):
        X, Y = emb.preprocess_dataset(dataset)
        label_emb = mds[emb.class_num*idx:emb.class_num*(idx+1)].detach().numpy()
        labels = torch.stack([torch.from_numpy(label_emb[target])
                            for target in Y], dim=0).squeeze(1).to(device)
        data_X.append(X)
        data_Y.append(labels)
    d_y = data_Y[0].shape[1]
    manifolds = [Euclidean(28*28, device=device), Lorentz(d_y, projection="horospheric", device=device)]
    product_manifold = ProductManifold(manifolds, torch.ones((2,), device=device)/np.sqrt(2))
    d_sw = np.zeros((len(train_all), len(train_all)))
    for i in range(len(train_all)):
        for j in range(i):    
            sw = sliced_wasserstein([data_X[i], data_Y[i]], [data_X[j], data_Y[j]], n_projs, product_manifold)
            print(sw.shape)
            d_sw[i, j] = sw.item()
            d_sw[j, i] = sw.item()
    return d_sw


otdd_hswfs(train_all, device=DEVICE)