import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
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



def main():
    parser = argparse.ArgumentParser(description='Arguments for sOTDD and OTDD computations')
    parser.add_argument('--parent_dir', type=str, default="saved_runtime_cifar10", help='Parent directory')
    parser.add_argument('--num_projections', type=int, default=10000, help='Number of projections for sOTDD')
    args = parser.parse_args()

    num_projections = args.num_projections

    save_dir = f'{args.parent_dir}/time_comparison/MNIST/'
    os.makedirs(save_dir, exist_ok=True)

    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = "cpu"
    print(f"Use CUDA or not: {DEVICE}")

    dataset = CIFAR10(root=f'data/CIFAR{num_classes}', train=True, download=False)
    test_dataset = CIFAR10(root=f'data/CIFAR{num_classes}', train=False, download=False, transform=transform)

    num_classes = len(torch.unique(dataset.targets))

    indices = np.arange(len(dataset))
    shuffled_indices = np.random.permutation(indices)
    
    max_dataset_size = len(dataset) // 2
    print(f"Maximum number of datapoint for each dataset: {max_dataset_size}")

    pointer_dataset1 = 0
    pointer_dataset2 = max_dataset_size

    list_dataset_size = [2000 * (i + 1) for i in range(int(max_dataset_size // 2000))]

    print(list_dataset_size)

    for dataset_size in list_dataset_size:
        print(f"Setting dataset to size of {dataset_size}..")
        idx1 = shuffled_indices[pointer_dataset1: pointer_dataset1 + dataset_size]
        idx2 = shuffled_indices[pointer_dataset2: pointer_dataset2 + dataset_size]

        sub1 = Subset(dataset=dataset, original_indices=idx1, transform=transform)
        sub2 = Subset(dataset=dataset, original_indices=idx2, transform=transform)

        dataloader1 = DataLoader(sub1, batch_size=128, shuffle=True)
        dataloader2 = DataLoader(sub2, batch_size=128, shuffle=True)

        dataloaders = [dataloader1, dataloader2]


        # NEW METHOD
        projection_list = [1000, 2000, 3000, 4000, 5000, 10000]
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
            list_pairwise_dist, duration_periods = compute_pairwise_distance(list_D=dataloaders, num_projections=proj_id, device=DEVICE, evaluate_time=True, **kwargs)
            for i in duration_periods.keys():
                print(i, duration_periods[i])
            t = 0
            for i in range(len(dataloaders)):
                for j in range(i+1, len(dataloaders)):
                    pairwise_dist[i, j] = list_pairwise_dist[t]
                    pairwise_dist[j, i] = list_pairwise_dist[t]
                    t += 1
            torch.save(pairwise_dist, f'{save_dir}/sotdd_{proj_id}_dist.pt')
            with open(f'{save_dir}/time_running.txt', 'a') as file:
                file.write(f"Time proccesing for sOTDD ({proj_id} projections): {duration_periods[-1]} \n")




        # OTDD
        dict_OTDD = torch.zeros(len(dataloaders), len(dataloaders))
        print("Compute OTDD (exact)...")
        start_time_otdd = time.time()
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

        end_time_otdd = time.time()
        otdd_time_taken = end_time_otdd - start_time_otdd
        print(otdd_time_taken)

        torch.save(dict_OTDD, f'{save_dir}/exact_otdd_dist.pt')
        with open(f'{save_dir}/time_running.txt', 'a') as file:
            file.write(f"Time proccesing for OTDD (exact): {otdd_time_taken} \n")




        # OTDD
        dict_OTDD = torch.zeros(len(dataloaders), len(dataloaders))
        print("Compute OTDD (gaussian_approx, iter 20)...")
        start_time_otdd = time.time()
        for i in range(len(dataloaders)):
            for j in range(i+1, len(dataloaders)):
                start_time_otdd = time.time()
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
        end_time_otdd = time.time()
        otdd_time_taken = end_time_otdd - start_time_otdd
        print(otdd_time_taken)
        torch.save(dict_OTDD, f'{save_dir}/ga_otdd_dist.pt')
        with open(f'{save_dir}/time_running.txt', 'a') as file:
            file.write(f"Time proccesing for OTDD (gaussian_approx, iter 20): {otdd_time_taken} \n")




if __name__ == "__main__":
    main()

