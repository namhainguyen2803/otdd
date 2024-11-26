from scipy import stats
from matplotlib.ticker import FormatStrFormatter
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
import argparse

from otdd.pytorch.utils import *


class CustomTensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.targets = labels

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
    dataloader.datasets = dataset
    return dataloader


def main():
    parser = argparse.ArgumentParser(description='Arguments for sOTDD and OTDD computations')
    parser.add_argument('--method', type=str, default="sotdd", help="Method name")
    parser.add_argument('--maxsize', type=int, default=50000, help='Parent directory')
    args = parser.parse_args()

    saved_path = 'saved_augmentation_2'

    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = "cpu"


    result = dict()
    result_list = list()


    for seed_file_name in os.listdir(saved_path):
        if "png" in seed_file_name or "pdf" in seed_file_name or "csv" in seed_file_name:
            continue
        else:

            seed_id = int(seed_file_name.split("_")[-1])
            seed_path = f"{saved_path}/{seed_file_name}"
            log_file = f"{seed_path}/log_seed_{seed_id}.log"
            with open(log_file, 'r') as file:
                lines = file.readlines()
            for line in lines:
                if "CIFAR10 Test Accuracy after Adaptation:" in line:
                    acc = float(line.split("(")[-1].split(",")[0])
                    
                    train_imagenet_path = f"{seed_path}/transformed_train_imagenet.pt"
                    train_cifar10_path = f"{seed_path}/transformed_train_cifar10.pt"

                    cifar10_dataloader = get_dataloader(datadir=train_cifar10_path, maxsize=args.maxsize, batch_size=64)
                    imagenet_dataloader = get_dataloader(datadir=train_imagenet_path, maxsize=args.maxsize, batch_size=64)
                    dataloaders = [cifar10_dataloader, imagenet_dataloader]

                    if args.method == "sotdd":
                        # sOTDD
                        kwargs = {
                            "dimension": 32,
                            "num_channels": 3,
                            "num_moments": 4,
                            "use_conv": True,
                            "precision": "float",
                            "p": 2,
                            "chunk": 1000
                        }
                        list_pairwise_dist, sotdd_time_taken = compute_pairwise_distance(list_D=dataloaders, num_projections=10000, device=DEVICE, evaluate_time=True, **kwargs)
                        sotdd_dist = list_pairwise_dist[0]
                        print(f"sOTDD distance: {sotdd_dist}")
                        dist = sotdd_dist

                    elif args.method == "otdd_exact":
                        # OTDD (Exact)
                        cifar10_dataloader = get_dataloader(datadir=train_cifar10_path, maxsize=args.maxsize, batch_size=64)
                        imagenet_dataloader = get_dataloader(datadir=train_imagenet_path, maxsize=args.maxsize, batch_size=64)
                        otdd_dist = DatasetDistance(cifar10_dataloader,
                                                    imagenet_dataloader,
                                                    inner_ot_method='exact',
                                                    debiased_loss=True,
                                                    p=2,
                                                    entreg=1e-3,
                                                    device=DEVICE)
                        otdd_exact_dist = otdd_dist.distance(maxsamples=None).item()
                        print(f"OTDD (Exact): {otdd_exact_dist}")
                        dist = otdd_exact_dist

                    elif args.method == "otdd_ga":
                        # OTDD (Gaussian)
                        cifar10_dataloader = get_dataloader(datadir=train_cifar10_path, maxsize=args.maxsize, batch_size=64)
                        imagenet_dataloader = get_dataloader(datadir=train_imagenet_path, maxsize=args.maxsize, batch_size=64)
                        otdd_dist = DatasetDistance(cifar10_dataloader,
                                                    imagenet_dataloader,
                                                    inner_ot_method='gaussian_approx',
                                                    debiased_loss=True,
                                                    p=2,
                                                    sqrt_method='approximate',
                                                    nworkers_stats=0,
                                                    sqrt_niters=20,
                                                    entreg=1e-3,
                                                    device=DEVICE)
                        otdd_ga_dist = otdd_dist.distance(maxsamples=None).item()
                        print(f"OTDD (Gaussian): {otdd_ga_dist}")
                        dist = otdd_ga_dist

                    result[seed_id] = [acc, dist]
                    result_list.append([acc, dist])
    
    with open(f'{saved_path}/acc_dist_method_{args.method}_maxsize_{args.maxsize}.txt', 'w') as file:
        for seed_id, list_acc_dist in result.items():
            file.write(f"seed id: {seed_id}, accuracy: {list_acc_dist[0]}, distance: {list_acc_dist[1]}")
    
    result_list = torch.tensor(result_list)
    torch.save(result_list, f'{saved_path}/acc_dist_method_{args.method}_maxsize_{args.maxsize}.pt')

if __name__ == "__main__":
    main()