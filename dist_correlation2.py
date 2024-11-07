import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.method4 import compute_pairwise_distance
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

import time
from datetime import datetime, timedelta

LIST_DATASETS = ["MNIST", "FashionMNIST", "EMNIST", "KMNIST", "USPS"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAXSIZE_DIST = 5000

def create_dataset(maxsamples=MAXSIZE_DIST, maxsize_for_each_class=None):

    METADATA_DATASET = dict()
    for dataset_name in LIST_DATASETS:

        METADATA_DATASET[dataset_name] = dict()

        if dataset_name == "USPS":
            dataloader = load_torchvision_data(dataset_name, valid_size=0, resize=28, download=False, maxsize=maxsamples, datadir="data2/USPS", maxsize_for_each_class=maxsize_for_each_class)[0]
        else:
            dataloader = load_torchvision_data(dataset_name, valid_size=0, resize=28, download=False, maxsize=maxsamples, maxsize_for_each_class=maxsize_for_each_class)[0]

        METADATA_DATASET[dataset_name]["train_loader"] = dataloader['train']
        METADATA_DATASET[dataset_name]["test_loader"] = dataloader['test']

        if dataset_name == "MNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        elif dataset_name == "KMNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        elif dataset_name == "EMNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 26
        elif dataset_name == "FashionMNIST":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        elif dataset_name == "USPS":
            METADATA_DATASET[dataset_name]["num_classes"] = 10
        else:
            raise("Unknown src dataset")
    
    return METADATA_DATASET

def compute_new_distance_dataset(maxsamples=MAXSIZE_DIST, maxsize_for_each_class=None, num_projection=1000):

    METADATA_DATASET = create_dataset(maxsamples=maxsamples, maxsize_for_each_class=maxsize_for_each_class)

    list_dataset = [METADATA_DATASET[source_dataset]["train_loader"] for source_dataset in LIST_DATASETS]

    res = compute_pairwise_distance(list_dataset, maxsamples=maxsamples, num_projection=1000, chunk=100, num_moments=3, image_size=28, dimension=None, num_channels=1, device='cpu', dtype=torch.FloatTensor)

    return res

if __name__ == "__main__":

    print("Compute new method...")
    start_time_new_method = time.time()
    dict_new_dist = compute_new_distance_dataset(maxsamples=60000, num_projection=1000)
    end_time_new_method = time.time()
    
    new_method_time_taken = end_time_new_method - start_time_new_method
    print(f"Finish computing new method. Time taken: {new_method_time_taken:.2f} seconds")

    with open(f'result.txt', 'a') as f:
        f.write(f"Start computing new method \n New Method Distance: \n")

        k = 0
        for i in range(len(LIST_DATASETS)):
            for j in range(i+1, len(LIST_DATASETS)):
                
                source_dataset = LIST_DATASETS[i]
                target_dataset = LIST_DATASETS[j]

                f.write(f" From {source_dataset} to {target_dataset}, distance: {dict_new_dist[i][j]} \n")
                k += 1
        
        f.write(f"Finish computing New method. Time taken: {new_method_time_taken:.2f} seconds \n \n")

