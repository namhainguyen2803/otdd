import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
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


def main():
    parser = argparse.ArgumentParser(description='Arguments for sOTDD and OTDD computations')
    parser.add_argument('--num_splits', type=int, default=2, help='Number of splits for dataset')
    parser.add_argument('--split_size', type=int, default=200, help='Size of each dataset split')
    parser.add_argument('--num_projections', type=int, default=1000, help='Number of projections for sOTDD')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes in the dataset')

    args = parser.parse_args()

    num_splits = args.num_splits
    split_size = args.split_size
    num_projections = args.num_projections
    num_classes = args.num_classes

    save_dir = f'saved_3/time_comparison/CIFAR100/dataset_size/SS{split_size}_NS{num_splits}_NP{num_projections}'
    os.makedirs(save_dir, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use CUDA or not: {DEVICE}")

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
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    dataset = CIFAR100(root=f'data2/CIFAR{num_classes}', train=True, download=False)
    test_dataset = CIFAR100(root=f'data2/CIFAR{num_classes}', train=False, download=False, transform=transform)

    print(split_size, len(dataset))
    indices = np.arange(len(dataset))

    data_index_cls = dict()
    classes = torch.unique(torch.tensor(dataset.targets))
    for cls_id in classes:
        data_index_cls[cls_id] = indices[torch.tensor(dataset.targets) == cls_id]

    for cls_id in data_index_cls.keys():
        np.random.shuffle(data_index_cls[cls_id])

    subsets = []
    for i in range(num_splits):
        subset_indices = list()
        for cls_id in data_index_cls.keys():
            num_dataset_cls = split_size // num_classes
            start_idx = i * num_dataset_cls
            end_idx = min(start_idx + num_dataset_cls, len(data_index_cls[cls_id]))
            subset_indices.extend(data_index_cls[cls_id][start_idx:end_idx])

        np.random.shuffle(subset_indices)
        print(len(subset_indices))
        sub = Subset(dataset=dataset, original_indices=subset_indices, transform=transform)
        subsets.append(sub)

    dataloaders = []
    for subset in subsets:
        dataloader = DataLoader(subset, batch_size=32, shuffle=True)
        dataloaders.append(dataloader)


    # NEW METHOD
    pairwise_dist = torch.zeros(len(dataloaders), len(dataloaders))
    print("Compute sOTDD...")
    print(f"Number of datasets: {len(dataloaders)}")
    list_pairwise_dist, duration_periods = compute_pairwise_distance(list_D=dataloaders, num_projections=num_projections, device=DEVICE, evaluate_time=True)
    for i in duration_periods.keys():
        print(i, duration_periods[i])

    t = 0
    for i in range(len(dataloaders)):
        for j in range(i+1, len(dataloaders)):
            pairwise_dist[i, j] = list_pairwise_dist[t]
            pairwise_dist[j, i] = list_pairwise_dist[t]
            t += 1

    torch.save(pairwise_dist, f'{save_dir}/sotdd_dist.pt')
    with open(f'{save_dir}/time_running.txt', 'a') as file:
        for i in duration_periods.keys():
            file.write(f"Time proccesing for sOTDD ({i} projections): {duration_periods[i]} \n")

    # OTDD
    dict_OTDD = torch.zeros(len(dataloaders), len(dataloaders))
    print("Compute OTDD (exact)...")
    start_time_otdd = time.time()
    for i in range(len(dataloaders)):
        for j in range(i+1, len(dataloaders)):
            start_time_otdd = time.time()
            dist = DatasetDistance(dataloaders[i],
                                    dataloaders[j],
                                    inner_ot_method='exact',
                                    debiased_loss=True,
                                    p=2,
                                    entreg=1e-1,
                                    device=DEVICE)
            d = dist.distance(maxsamples=None).item()
            dict_OTDD[i][j] = d
            dict_OTDD[j][i] = d

    end_time_otdd = time.time()
    otdd_time_taken = end_time_otdd - start_time_otdd
    print(otdd_time_taken)

    torch.save(dict_OTDD, f'{save_dir}/exact_otdd_dist.pt')
    with open(f'{save_dir}/time_running.txt', 'a') as file:
        file.write(f"Time proccesing for OTDD (exact): {otdd_time_taken} \n")


    # # OTDD
    # dict_OTDD = torch.zeros(len(dataloaders), len(dataloaders))
    # print("Compute OTDD (gaussian_approx)...")
    # start_time_otdd = time.time()
    # for i in range(len(dataloaders)):
    #     for j in range(i+1, len(dataloaders)):
    #         start_time_otdd = time.time()
    #         dist = DatasetDistance(dataloaders[i],
    #                                 dataloaders[j],
    #                                 inner_ot_method='gaussian_approx',
    #                                 debiased_loss=True,
    #                                 p=2,
    #                                 entreg=1e-1,
    #                                 device=DEVICE)
    #         d = dist.distance(maxsamples=100).item()
    #         dict_OTDD[i][j] = d
    #         dict_OTDD[j][i] = d

    # end_time_otdd = time.time()
    # otdd_time_taken = end_time_otdd - start_time_otdd
    # print(otdd_time_taken)

    # torch.save(dict_OTDD, f'{save_dir}/ga_otdd_dist.pt')
    # with open(f'{save_dir}/time_running.txt', 'a') as file:
    #     file.write(f"Time proccesing for OTDD (gaussian_approx): {otdd_time_taken} \n")

if __name__ == "__main__":
    main()