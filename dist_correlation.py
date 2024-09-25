import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.method3 import NewDatasetDistance
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
# LIST_DATASETS = ["MNIST", "FashionMNIST"]
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


def compute_all_distance_dataset(maxsamples=MAXSIZE_DIST, maxsize_for_each_class=None):

    METADATA_DATASET = create_dataset(maxsamples=maxsamples, maxsize_for_each_class=maxsize_for_each_class)

    list_otdd = list()
    list_new_dist = list()

    for i in range(len(LIST_DATASETS)):
        for j in range(i+1, len(LIST_DATASETS)):
            
            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]

            dist = DatasetDistance(METADATA_DATASET[source_dataset]["train_loader"], 
                                    METADATA_DATASET[target_dataset]["train_loader"],
                                    inner_ot_method='exact',
                                    debiased_loss=True,
                                    p=2, 
                                    entreg=1e-1,
                                    device='cpu')
            d = dist.distance(maxsamples = maxsamples).item()

            list_otdd.append(d)

            print(f'DIST({source_dataset}, {target_dataset})={d:8.2f}')

            new_dist = NewDatasetDistance(METADATA_DATASET[source_dataset]["train_loader"], 
                                            METADATA_DATASET[target_dataset]["train_loader"], 
                                            p=2, 
                                            device='cpu')
            new_d = new_dist.distance(maxsamples=maxsamples, num_projection=1000).item()

            list_new_dist.append(new_d)

            print(f'NEW DIST({source_dataset}, {target_dataset})={new_d:8.2f}')

    return list_otdd, list_new_dist

def compute_OTDD(maxsamples=MAXSIZE_DIST, maxsize_for_each_class=None):

    METADATA_DATASET = create_dataset(maxsamples=maxsamples, maxsize_for_each_class=maxsize_for_each_class)

    dict_OTDD_dist = dict()
    for i in range(len(LIST_DATASETS)):
        dict_OTDD_dist[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            dict_OTDD_dist[LIST_DATASETS[i]][LIST_DATASETS[j]] = 0

    for i in range(len(LIST_DATASETS)):
        for j in range(i+1, len(LIST_DATASETS)):
            
            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]

            dist = DatasetDistance(METADATA_DATASET[source_dataset]["train_loader"], 
                                    METADATA_DATASET[target_dataset]["train_loader"],
                                    inner_ot_method='exact',
                                    debiased_loss=True,
                                    p=2, 
                                    entreg=1e-1,
                                    device='cpu')
            d = dist.distance(maxsamples = maxsamples).item()

            dict_OTDD_dist[source_dataset][target_dataset] = d
            dict_OTDD_dist[target_dataset][source_dataset] = d

            print(f'DIST({source_dataset}, {target_dataset})={d:8.2f}')

    return dict_OTDD_dist

def compute_new_distance_dataset(maxsamples=MAXSIZE_DIST, maxsize_for_each_class=None, num_projection=1000):

    METADATA_DATASET = create_dataset(maxsamples=maxsamples, maxsize_for_each_class=maxsize_for_each_class)

    dict_new_dist = dict()
    for i in range(len(LIST_DATASETS)):
        dict_new_dist[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            dict_new_dist[LIST_DATASETS[i]][LIST_DATASETS[j]] = 0
        
    for i in range(len(LIST_DATASETS)):
        for j in range(i+1, len(LIST_DATASETS)):
            
            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]

            print(f"{source_dataset} -> {target_dataset}")

            new_dist = NewDatasetDistance(METADATA_DATASET[source_dataset]["train_loader"], 
                                            METADATA_DATASET[target_dataset]["train_loader"], 
                                            p=2, 
                                            device='cpu')
            new_d = new_dist.distance(maxsamples=maxsamples, num_projection=num_projection).item()

            dict_new_dist[source_dataset][target_dataset] = new_d
            dict_new_dist[target_dataset][source_dataset] = new_d

            print(f'NEW DIST({source_dataset}, {target_dataset})={new_d:8.2f}')

    return dict_new_dist

def compute_wasserstein(maxsamples=MAXSIZE_DIST, maxsize_for_each_class=None, num_projection=1000):

    METADATA_DATASET = create_dataset(maxsamples=maxsamples, maxsize_for_each_class=maxsize_for_each_class)
    list_new_dist = list()

    for i in range(len(LIST_DATASETS)):
        for j in range(i+1, len(LIST_DATASETS)):
            
            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]

            new_dist = NewDatasetDistance(METADATA_DATASET[source_dataset]["train_loader"], 
                                            METADATA_DATASET[target_dataset]["train_loader"], 
                                            p=2, 
                                            device='cpu')
            new_d = new_dist.distance_without_labels(maxsamples=maxsamples, num_projection=num_projection).item()

            list_new_dist.append(new_d)

            print(f'NEW DIST({source_dataset}, {target_dataset})={new_d:8.2f}')

    return list_new_dist

if __name__ == "__main__":

    compute_otdd = False
    compute_new_dist = False

    if compute_new_dist:
        time_difference = timedelta(hours=7)
        current_utc = datetime.utcnow()
        current_vietnam_time = current_utc + time_difference
        current_datetime_vn = current_vietnam_time.strftime('%Y-%m-%d_%H-%M-%S')
        parent_dir = f"saved/corr/{current_datetime_vn}"
        os.makedirs(parent_dir, exist_ok=True)

    if compute_otdd:
        print("Compute OTDD...")
        start_time_otdd = time.time()
        dict_OTDD_dist = compute_OTDD(maxsamples=5000)
        end_time_otdd = time.time()
        
        dict_OTDD_dist_saved_path = f'{parent_dir}/OTDD_dist.json'
        with open(dict_OTDD_dist_saved_path, 'w') as json_file:
            json.dump(dict_OTDD_dist, json_file, indent=4)
        
        otdd_time_taken = end_time_otdd - start_time_otdd
        print(f"Finish computing OTDD. Time taken: {otdd_time_taken:.2f} seconds")

        with open(f'{parent_dir}/result.txt', 'a') as f:
            f.write(f"Start computing OTDD \n OTDD Distance: \n")

            k = 0
            for i in range(len(LIST_DATASETS)):
                for j in range(i+1, len(LIST_DATASETS)):
                    
                    source_dataset = LIST_DATASETS[i]
                    target_dataset = LIST_DATASETS[j]

                    f.write(f" From {source_dataset} to {target_dataset}, distance: {dict_OTDD_dist[source_dataset][target_dataset]} \n")
                    k += 1
            
            f.write(f"Finish computing OTDD. Time taken: {otdd_time_taken:.2f} seconds \n \n")
    else:
        dict_OTDD_dist_saved_path = 'saved/corr/OTDD_dist.json'
        with open(dict_OTDD_dist_saved_path, 'r') as file:
            dict_OTDD_dist = json.load(file)


    if compute_new_dist:
        print("Compute new method...")
        start_time_new_method = time.time()
        dict_new_dist = compute_new_distance_dataset(maxsamples=60000, num_projection=1000)
        end_time_new_method = time.time()

        dict_new_dist_saved_path = f'{parent_dir}/new_dist.json'
        with open(dict_new_dist_saved_path, 'w') as json_file:
            json.dump(dict_new_dist, json_file, indent=4)
        
        new_method_time_taken = end_time_new_method - start_time_new_method
        print(f"Finish computing new mthod. Time taken: {new_method_time_taken:.2f} seconds")

        with open(f'{parent_dir}/result.txt', 'a') as f:
            f.write(f"Start computing new method \n New Method Distance: \n")

            k = 0
            for i in range(len(LIST_DATASETS)):
                for j in range(i+1, len(LIST_DATASETS)):
                    
                    source_dataset = LIST_DATASETS[i]
                    target_dataset = LIST_DATASETS[j]

                    f.write(f" From {source_dataset} to {target_dataset}, distance: {dict_new_dist[source_dataset][target_dataset]} \n")
                    k += 1
            
            f.write(f"Finish computing New method. Time taken: {new_method_time_taken:.2f} seconds \n \n")
    else:
        dict_new_dist_saved_path = 'result.json'
        with open(dict_new_dist_saved_path, 'r') as file:
            dict_new_dist = json.load(file)

    list_otdd = list()
    list_new_dist = list()
    for i in range(len(LIST_DATASETS)):
        for j in range(i+1, len(LIST_DATASETS)):
            
            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]
            if source_dataset == "USPS" and target_dataset == "FashionMNIST":
                continue
            if source_dataset == "FashionMNIST" and target_dataset == "USPS":
                continue

            list_otdd.append(dict_OTDD_dist[source_dataset][target_dataset])
            list_new_dist.append(dict_new_dist[source_dataset][target_dataset])

    rho, p_value = stats.pearsonr(list_otdd, list_new_dist)
    print(f"Overall Pearson correlation coefficient: {rho}")
    print(f"Overall P-value: {p_value}")

    with open(f'correlation_values.txt', 'a') as f:
        f.write(f"Pearson correlation coefficient: {rho}\n")
        f.write(f"P-value: {p_value}\n")

    list_X = np.array(list_otdd).reshape(-1, 1)
    list_y = np.array(list_new_dist)
    model = LinearRegression().fit(list_X, list_y)
    list_y_pred = model.predict(list_X)

    plt.figure(figsize=(10, 8))
    # sns.regplot(x=x, y=y, ci=95, scatter_kws={'s': 100}, line_kws={'color': 'blue'})
    plt.scatter(list_otdd, list_new_dist, s=100, color='blue', label='Data points')
    plt.plot(list_otdd, list_y_pred, color='red', linewidth=2, label='Fitted line')

    plt.xlabel('OTDD')
    plt.ylabel('NEW DISTANCE')
    plt.title(f'Distance Correlation: *NIST Datasets {rho:.4f}, {p_value:.4f}')

    plt.legend()
    plt.savefig(f'distance_correlation.png')


