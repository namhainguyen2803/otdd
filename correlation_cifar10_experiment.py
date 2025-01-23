import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
from otdd.pytorch.method5 import compute_pairwise_distance
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
from torch.utils.data.sampler import SubsetRandomSampler
import time
from trainer import train, test_func, frozen_module
from models.resnet import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from PIL import Image
import random

from wte.distance import WTE
from scipy.spatial import distance


from geoopt import Lorentz as Lorentz_geoopt

from hswfs_otdd.utils.hmds import HyperMDS
from hswfs_otdd.utils.bures_wasserstein import LabelsBW

from hswfs_otdd.hswfs.manifold.euclidean import Euclidean
from hswfs_otdd.hswfs.manifold.product import ProductManifold
from hswfs_otdd.hswfs.manifold.poincare import Poincare
from hswfs_otdd.hswfs.manifold.lorentz import Lorentz
from hswfs_otdd.hswfs.sw import sliced_wasserstein


def generate_reference(num, dim_low, dim, attached_dim, seed=0):
    torch.manual_seed(seed)
    med = torch.rand(num, 3, dim_low, dim_low)
    s = dim / dim_low
    m = nn.Upsample(scale_factor=s, mode='bilinear')
    upsampled = m(med).reshape(num, -1)
    attached = torch.randn(num, attached_dim)
    return torch.cat((upsampled, attached), dim=1).float()



class Subset(Dataset):
    def __init__(self, dataset, original_indices, transform):
        self._dataset = dataset
        self._original_indices = original_indices
        self.transform = transform
        self.indices = torch.arange(start=0, end=len(self._original_indices), step=1)
        self.data = self._dataset.data[self._original_indices]
        self.targets = torch.tensor(self._dataset.targets)[self._original_indices]
        self.classes = sorted(torch.unique(torch.tensor(self._dataset.targets)).tolist())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.targets[idx]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

# (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

def main():
    parser = argparse.ArgumentParser(description='Arguments for sOTDD and OTDD computations')
    parser.add_argument('--parent_dir', type=str, default="saved_corr_cifar10_v100_19_01_2025_2", help='Parent directory')
    args = parser.parse_args()

    parent_dir = f'{args.parent_dir}/correlation/CIFAR10'
    os.makedirs(parent_dir, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = "cpu"
    print(f"Use CUDA or not: {DEVICE}")

    dataset = CIFAR10(root=f'data/CIFAR10', train=True, download=False)
    test_dataset = CIFAR10(root=f'data/CIFAR10', train=False, download=False, transform=transform)

    num_classes = len(torch.unique(torch.tensor(dataset.targets)))

    indices = np.arange(len(dataset))
    shuffled_indices = np.random.permutation(indices)
    
    max_dataset_size = len(dataset) // 2
    print(f"Maximum number of datapoint for each dataset: {max_dataset_size}")

    # list_dataset_size = [1000 * (i + 1) for i in range(int(max_dataset_size // 1000))]
    # list_dataset_size = [1000, 3000, 5000, 7000, 10000] * 10
    list_dataset_size = [random.randint(5, 10) * 1000 for i in range(10)]

    print(list_dataset_size)

    for idx in range(len(list_dataset_size)):
        dataset_size = list_dataset_size[idx]
        save_dir = f"{parent_dir}/seed_{idx + 3}_size_{dataset_size}"
        os.makedirs(save_dir, exist_ok=True)

        shuffled_indices = np.random.permutation(indices)
        print(f"Setting dataset to size of {dataset_size}..")
        idx1 = shuffled_indices[:dataset_size]
        idx2 = shuffled_indices[-dataset_size:]

        assert len(idx1) == len(idx2) == dataset_size, "wrong number of dataset size"

        sub1 = Subset(dataset=dataset, original_indices=idx1, transform=transform)
        sub2 = Subset(dataset=dataset, original_indices=idx2, transform=transform)

        subdatasets = [sub1, sub2]

        dataloader1 = DataLoader(sub1, batch_size=128, shuffle=True)
        dataloader2 = DataLoader(sub2, batch_size=128, shuffle=True)

        dataloaders = [dataloader1, dataloader2]


        # NEW METHOD
        projection_list = [10000, 5000, 3000, 1000, 500]
        for proj_id in projection_list:
            pairwise_dist = torch.zeros(len(dataloaders), len(dataloaders))
            print("Compute sOTDD...")
            print(f"Number of datasets: {len(dataloaders)}")
            kwargs = {
                "dimension": 32,
                "num_channels": 3,
                "num_moments": 5,
                "use_conv": True,
                "precision": "float",
                "p": 2,
                "chunk": 1000
            }
            list_pairwise_dist, sotdd_time_taken = compute_pairwise_distance(list_D=dataloaders, num_projections=proj_id, device=DEVICE, evaluate_time=True, **kwargs)
            t = 0
            for i in range(len(dataloaders)):
                for j in range(i+1, len(dataloaders)):
                    pairwise_dist[i, j] = list_pairwise_dist[t]
                    pairwise_dist[j, i] = list_pairwise_dist[t]
                    t += 1
            torch.save(pairwise_dist, f'{save_dir}/sotdd_{proj_id}_dist.pt')


        # OTDD
        dict_OTDD = torch.zeros(len(dataloaders), len(dataloaders))
        print("Compute OTDD (exact)...")
        for i in range(len(dataloaders)):
            for j in range(i+1, len(dataloaders)):
                dist = DatasetDistance(dataloaders[i],
                                        dataloaders[j],
                                        inner_ot_method='exact',
                                        debiased_loss=True,
                                        p=2,
                                        entreg=1e-3,
                                        device=DEVICE)
                d = dist.distance(maxsamples=None).item()
                dict_OTDD[i][j] = d
                dict_OTDD[j][i] = d
        torch.save(dict_OTDD, f'{save_dir}/exact_otdd_dist.pt')


        # OTDD
        dict_OTDD = torch.zeros(len(dataloaders), len(dataloaders))
        print("Compute OTDD (gaussian_approx, iter 20)...")
        for i in range(len(dataloaders)):
            for j in range(i+1, len(dataloaders)):
                dist = DatasetDistance(dataloaders[i],
                                        dataloaders[j],
                                        inner_ot_method='gaussian_approx',
                                        debiased_loss=True,
                                        p=2,
                                        sqrt_method='approximate',
                                        nworkers_stats=0,
                                        sqrt_niters=20,
                                        entreg=1e-3,
                                        device=DEVICE)
                d = dist.distance(maxsamples=None).item()
                dict_OTDD[i][j] = d
                dict_OTDD[j][i] = d
        torch.save(dict_OTDD, f'{save_dir}/ga_otdd_dist.pt')


        # WTE
        reference = generate_reference(dataset_size, 4, 32, 10)
        wtes = WTE(subdatasets, label_dim=10, device=DEVICE, ref=reference.cpu(), maxsamples=dataset_size)
        wtes = wtes.reshape(wtes.shape[0], -1)
        wte_distance = distance.cdist(wtes, wtes, 'euclidean')
        torch.save(wte_distance, f'{save_dir}/wte.pt')


        # HSWFS_OTDD
        scaling = 0.1
        d = 10
        n_epochs = 20000
        emb = LabelsBW(device=DEVICE, maxsamples=dataset_size)
        distance_array = emb.dissimilarity_for_all(subdatasets)
        lorentz_geoopt = Lorentz_geoopt()
        embedding = HyperMDS(d, lorentz_geoopt, torch.optim.Adam, scaling=scaling, loss="ads")
        mds, L = embedding.fit_transform(torch.tensor(distance_array, dtype=torch.float64), n_epochs=n_epochs, lr=1e-3)
        dist_mds = lorentz_geoopt.dist(mds[None], mds[:,None]).detach().cpu().numpy()
        diff_dist = np.abs(scaling * distance_array - dist_mds)
        data_X = [] # data
        data_Y = [] # labels
        for cac_idx, cac_dataset in enumerate(subdatasets):
            X, Y = emb.preprocess_dataset(cac_dataset)
            label_emb = mds[emb.class_num*cac_idx:emb.class_num*(cac_idx+1)].detach().numpy()
            labels = torch.stack([torch.from_numpy(label_emb[target])
                                for target in Y], dim=0).squeeze(1).to(DEVICE)
            data_X.append(X)
            data_Y.append(labels)
        d_y = data_Y[0].shape[1]
        manifolds = [Euclidean(32*32*3, device=DEVICE), Lorentz(d_y, projection="horospheric", device=DEVICE)]
        product_manifold = ProductManifold(manifolds, torch.ones((2,), device=DEVICE)/np.sqrt(2))

        projection_list = [100, 500, 1000, 5000, 10000]
        d_sw = np.zeros((len(projection_list), len(subdatasets), len(subdatasets)))
        for i in range(len(subdatasets)):
            for j in range(i): 
                sw = sliced_wasserstein([data_X[i], data_Y[i]], [data_X[j], data_Y[j]], projection_list[-1], product_manifold)
                for proj_id in range(len(projection_list)):
                    num_proj = projection_list[proj_id]
                    d_sw[proj_id, i, j] = sw[num_proj - 1].item()
                    d_sw[proj_id, j, i] = sw[num_proj - 1].item()
        for proj_id in range(len(projection_list)):
            num_proj = projection_list[proj_id]
            torch.save(d_sw[proj_id, :, :], f'{save_dir}/hswfs_{num_proj}_dist.pt')


if __name__ == "__main__":
    main()

