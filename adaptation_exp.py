import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data
import otdd.pytorch.method5 as method5
import otdd.pytorch.method_gaussian as method_gaussian
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



LIST_DATASETS = ["MNIST", "FashionMNIST", "EMNIST", "KMNIST", "USPS"]
ACC_ADAPT = dict()
DIST = dict()

parent_dir = f"saved_nist/dist"
pretrained_path = parent_dir + "/pretrained_weights"
adapt_path = parent_dir + "/finetune_weights"

os.makedirs(parent_dir, exist_ok=True)
os.makedirs(pretrained_path, exist_ok=True)
os.makedirs(adapt_path, exist_ok=True)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

# Load data
MAXSIZE_DIST = 5000
MAXSIZE_TRAINING = None


def create_dataset(maxsamples=MAXSIZE_DIST, maxsize_for_each_class=None):

    METADATA_DATASET = dict()
    for dataset_name in LIST_DATASETS:

        METADATA_DATASET[dataset_name] = dict()

        if dataset_name == "USPS":
            dataloader = load_torchvision_data(dataset_name, valid_size=0, resize=28, download=False, maxsize=maxsamples, datadir="data/USPS", maxsize_for_each_class=maxsize_for_each_class)[0]
        else:
            dataloader = load_torchvision_data(dataset_name, valid_size=0, resize=28, download=False, maxsize=maxsamples, maxsize_for_each_class=maxsize_for_each_class)[0]

        METADATA_DATASET[dataset_name]["train_loader"] = dataloader['train']
        METADATA_DATASET[dataset_name]["test_loader"] = dataloader['test']
        METADATA_DATASET[dataset_name]["pretrained_extractor_path"] = f'{pretrained_path}/{dataset_name}/extractor_layers.pth'
        METADATA_DATASET[dataset_name]["pretrained_classifier_path"] = f'{pretrained_path}/{dataset_name}/fc_layers.pth'

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


def compute_otdd_distance(maxsamples=MAXSIZE_DIST, METADATA_DATASET=None):

    if METADATA_DATASET is None:
        METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    all_dist_dict = dict()
    for i in range(len(LIST_DATASETS)):
        all_dist_dict[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            all_dist_dict[LIST_DATASETS[i]][LIST_DATASETS[j]] = 0

    for i in range(len(LIST_DATASETS)):
        for j in range(i+1, len(LIST_DATASETS)):
            
            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]

            # dist = DatasetDistance(METADATA_DATASET[source_dataset]["train_loader"], 
            #                         METADATA_DATASET[target_dataset]["train_loader"],
            #                         inner_ot_method='gaussian_approx',
            #                         sqrt_method='approximate',
            #                         nworkers_stats=0,
            #                         sqrt_niters=20,
            #                         debiased_loss=True,
            #                         p=2,
            #                         entreg=1e-3,
            #                         device=DEVICE)

            dist = DatasetDistance(METADATA_DATASET[source_dataset]["train_loader"], 
                                    METADATA_DATASET[target_dataset]["train_loader"],
                                    inner_ot_method='exact',
                                    debiased_loss=True,
                                    p=2,
                                    entreg=1e-3,
                                    device=DEVICE)

            d = dist.distance(maxsamples = maxsamples).item()

            all_dist_dict[source_dataset][target_dataset] = d
            all_dist_dict[target_dataset][source_dataset] = d

            print(f'DIST({source_dataset}, {target_dataset})={d:8.2f}')

    return all_dist_dict


def compute_sotdd_gaussian_distance(maxsamples=MAXSIZE_DIST, num_projection=10000, METADATA_DATASET=None):

    if METADATA_DATASET is None:
        METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    list_dataset = list()
    for i in range(len(LIST_DATASETS)):
        dt_name = LIST_DATASETS[i]
        list_dataset.append(METADATA_DATASET[dt_name]["train_loader"])
    
    kwargs = {
        "dimension": 28 * 28,
        "num_channels": 1,
        "precision": "float",
        "p": 2,
        "chunk": 100
    }

    sw_list = method_gaussian.compute_pairwise_distance(list_D=list_dataset, device=DEVICE, num_projections=num_projection, evaluate_time=False, **kwargs)

    all_dist_dict = dict()
    for i in range(len(LIST_DATASETS)):
        all_dist_dict[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            all_dist_dict[LIST_DATASETS[i]][LIST_DATASETS[j]] = 0

    k = 0
    for i in range(len(LIST_DATASETS)):
        for j in range(i + 1, len(LIST_DATASETS)):

            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]
            all_dist_dict[source_dataset][target_dataset] = sw_list[k].item()
            all_dist_dict[target_dataset][source_dataset] = sw_list[k].item()

            k += 1
    
    assert k == len(sw_list), "k != len(sw_list)"

    return all_dist_dict


def compute_sotdd_distance(maxsamples=MAXSIZE_DIST, num_projection=10000, METADATA_DATASET=None):

    if METADATA_DATASET is None:
        METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    list_dataset = list()
    for i in range(len(LIST_DATASETS)):
        dt_name = LIST_DATASETS[i]
        list_dataset.append(METADATA_DATASET[dt_name]["train_loader"])
    
    kwargs = {
        "dimension": 28 * 28,
        "num_channels": 1,
        "num_moments": 5,
        "use_conv": False,
        "precision": "float",
        "p": 2,
        "chunk": 1000
    }

    sw_list = method5.compute_pairwise_distance(list_D=list_dataset, device=DEVICE, num_projections=num_projection, evaluate_time=False, **kwargs)

    all_dist_dict = dict()
    for i in range(len(LIST_DATASETS)):
        all_dist_dict[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            all_dist_dict[LIST_DATASETS[i]][LIST_DATASETS[j]] = 0

    k = 0
    for i in range(len(LIST_DATASETS)):
        for j in range(i + 1, len(LIST_DATASETS)):

            source_dataset = LIST_DATASETS[i]
            target_dataset = LIST_DATASETS[j]
            all_dist_dict[source_dataset][target_dataset] = sw_list[k].item()
            all_dist_dict[target_dataset][source_dataset] = sw_list[k].item()

            k += 1
    
    assert k == len(sw_list), "k != len(sw_list)"

    return all_dist_dict


def train_source(num_epoch_source=20, maxsamples=MAXSIZE_TRAINING, device=DEVICE):

    METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    # TRAIN SOURCE PHASE
    ACC_NO_ADAPT = dict()
    for source in LIST_DATASETS:
        os.makedirs(f"{pretrained_path}/{source}", exist_ok=True)

        ft_extractor = FeatureExtractor(input_size=28).to(device)
        classifier = FullyConnectedNetwork(feat_dim=ft_extractor.feat_dim, num_classes=METADATA_DATASET[source]["num_classes"]).to(device)
        ft_extractor_optimizer = optim.Adam(ft_extractor.parameters(), lr=1e-3, weight_decay=1e-6)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)

        list_acc = list()
        for epoch in range(1, num_epoch_source + 1):
            train(feature_extractor=ft_extractor, 
                classifier=classifier, 
                device=device,
                train_loader=METADATA_DATASET[source]["train_loader"], 
                criterion=nn.CrossEntropyLoss(), 
                ft_extractor_optimizer=ft_extractor_optimizer, 
                classifier_optimizer=classifier_optimizer)
        
            acc_no_adapt = test_func(feature_extractor=ft_extractor, classifier=classifier, device=device, test_loader=METADATA_DATASET[source]["test_loader"])
            list_acc.append(acc_no_adapt)

        ACC_NO_ADAPT[source] = list_acc
        
        ft_extractor_path = f'{pretrained_path}/{source}/extractor_layers.pth'
        torch.save(ft_extractor.state_dict(), ft_extractor_path)

        classifier_path = f'{pretrained_path}/{source}/fc_layers.pth'
        torch.save(classifier.state_dict(), classifier_path)

        with open(f"{pretrained_path}/{source}/accuracy.txt", "a") as file:
            for i in range(len(ACC_NO_ADAPT[source])):
                file.write(f"Epoch: {i}, accuracy: {ACC_NO_ADAPT[source][i]} \n")


def training_and_adaptation(num_epochs=10, maxsamples=MAXSIZE_TRAINING, device=DEVICE):

    METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    # TRAIN ADAPTATION PHASE
    for i in range(len(LIST_DATASETS)):
        
        target = LIST_DATASETS[i]

        if target not in ACC_ADAPT:
            ACC_ADAPT[target] = dict()

        for j in range(len(LIST_DATASETS)):

            source = LIST_DATASETS[j]

            if target == source:
                continue
            
            else:
                
                os.makedirs(f"{adapt_path}/{target}/{source}", exist_ok=True)

                ft_extractor = FeatureExtractor(input_size=28).to(device)
                ft_extractor.load_state_dict(torch.load(METADATA_DATASET[source]["pretrained_extractor_path"]))
                ft_extractor_optimizer = None
                # ft_extractor_optimizer = optim.Adam(ft_extractor.parameters(), lr=1e-3, weight_decay=1e-6)

                classifier = FullyConnectedNetwork(feat_dim=ft_extractor.feat_dim, num_classes=METADATA_DATASET[source]["num_classes"]).to(device)
                classifier.load_state_dict(torch.load(METADATA_DATASET[source]["pretrained_classifier_path"]))
                classifier.change_head(new_num_classes=METADATA_DATASET[target]["num_classes"])
                classifier = classifier.to(device)
                classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)

                list_accuracy = list()
                for epoch in range(1, num_epochs + 1):
                    train(feature_extractor=ft_extractor, 
                        classifier=classifier, 
                        device=device, 
                        train_loader=METADATA_DATASET[target]["train_loader"], 
                        epoch=epoch,
                        criterion=nn.CrossEntropyLoss(), 
                        ft_extractor_optimizer=ft_extractor_optimizer, 
                        classifier_optimizer=classifier_optimizer)
                    
                    acc_adapt = test_func(ft_extractor, classifier, device, METADATA_DATASET[target]["test_loader"])
                    list_accuracy.append(acc_adapt)

                ACC_ADAPT[target][source] = list_accuracy

                ft_extractor_path = f'{adapt_path}/{target}/{source}/extractor_layers.pth'
                torch.save(ft_extractor.state_dict(), ft_extractor_path)

                classifier_path = f'{adapt_path}/{target}/{source}/fc_layers.pth'
                torch.save(classifier.state_dict(), classifier_path)

                with open(f"{adapt_path}/{target}/{source}/accuracy.txt", "a") as file:
                    for i in range(len(ACC_ADAPT[target][source])):
                        file.write(f"Epoch: {i}, accuracy: {ACC_ADAPT[target][source][i]} \n")

                print(f"In adaptation, source dataset {source}, target dataset: {target}, accuracy: {acc_adapt}")


if __name__ == "__main__":

    METADATA_DATASET = create_dataset(maxsamples=MAXSIZE_DIST)
    
    # DIST_otdd = compute_otdd_distance()
    # dist_file_path = f'{parent_dir}/otdd_dist_exact.json'
    # with open(dist_file_path, 'w') as json_file:
    #     json.dump(DIST_otdd, json_file, indent=4)

    # DIST_sotdd = compute_sotdd_distance(num_projection=10000, METADATA_DATASET=METADATA_DATASET)
    # dist_file_path = f'{parent_dir}/sotdd_arbitrary_dist.json'
    # with open(dist_file_path, 'w') as json_file:
    #     json.dump(DIST_sotdd, json_file, indent=4)

    

    DIST_sotdd_gaussian = compute_sotdd_gaussian_distance(num_projection=10000, METADATA_DATASET=METADATA_DATASET)
    dist_file_path = f'{parent_dir}/sotdd_gaussian_dist.json'
    with open(dist_file_path, 'w') as json_file:
        json.dump(DIST_sotdd_gaussian, json_file, indent=4)

    # train_source(num_epoch_source=20, maxsamples=MAXSIZE_TRAINING, device=DEVICE)
    # training_and_adaptation(num_epochs=10, maxsamples=MAXSIZE_TRAINING, device=DEVICE)


