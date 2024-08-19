import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.method import NewDatasetDistance
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

LIST_DATASETS = ["MNIST", "FMNIST", "EMNIST", "KMNIST", "USPS"]
ACC_ADAPT = dict()
DIST = dict()

# Load data
MAXSIZE = 2000
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

dist_file_path = 'saved/dist2.json'
with open(dist_file_path, 'w') as json_file:
    json.dump(DIST, json_file, indent=4)
print(f"DIST: {DIST}")



# Training the model

MAXSIZE = None
loaders_mnist  = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=MAXSIZE)[0]
loaders_kmnist  = load_torchvision_data('KMNIST', valid_size=0, resize = 28, maxsize=MAXSIZE)[0]
loaders_emnist  = load_torchvision_data('EMNIST', valid_size=0, resize = 28, maxsize=MAXSIZE)[0]
loaders_fmnist  = load_torchvision_data('FashionMNIST', valid_size=0, resize = 28, maxsize=MAXSIZE)[0]
loaders_usps  = load_torchvision_data('USPS',  valid_size=0, resize = 28, maxsize=MAXSIZE, datadir="data/USPS")[0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

META_DATA = {
    "MNIST": {
        "data_loader": loaders_mnist,
        "input_size": 28,
        "num_classes": 10
    },
    "KMNIST": {
        "data_loader": loaders_kmnist,
        "input_size": 28,
        "num_classes": 10
    },
    "FMNIST": {
        "data_loader": loaders_fmnist,
        "input_size": 28,
        "num_classes": 10
    },
    "EMNIST": {
        "data_loader": loaders_emnist,
        "input_size": 28,
        "num_classes": 10
    },
    "USPS": {
        "data_loader": loaders_usps,
        "input_size": 28,
        "num_classes": 10
    }
}


def create_baseline(num_epochs=20):
    # NO ADAPTATION PHASE   
    ft_extractor = FeatureExtractor(input_size=28).to(DEVICE)
    frozen_module(ft_extractor)

    ACC_NO_ADAPT = dict()
    for source in LIST_DATASETS:

        classifier = Classifier(feat_dim=ft_extractor.feat_dim, num_classes=META_DATA[source.upper()]["num_classes"]).to(DEVICE)
        optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
        for epoch in range(1, num_epochs + 1):
            train(feature_extractor=ft_extractor, 
                classifier=classifier, 
                device=DEVICE, 
                train_loader=META_DATA[source.upper()]["data_loader"]['train'], 
                epoch=epoch, 
                criterion=nn.CrossEntropyLoss(), 
                ft_extractor_optimizer=None, 
                classifier_optimizer=optimizer)
        
        acc_no_adapt = test(ft_extractor, classifier, DEVICE, META_DATA[source.upper()]["data_loader"]['test'])
        
        META_DATA[source.upper()]["accuracy"] = acc_no_adapt

        ACC_NO_ADAPT[source.upper()] = acc_no_adapt

        print(f"In creating baseline, dataset {source.upper()}, accuracy: {acc_no_adapt}")

    acc_no_adapt_file = 'saved/stats/acc_no_adapt.json'
    with open(acc_no_adapt_file, 'w') as json_file:
        json.dump(ACC_NO_ADAPT, json_file, indent=4)
    print(f"ACC_NO_ADAPT: {ACC_NO_ADAPT}")


def train_source_dataset_phase(num_epochs=20):
    # TRAIN SOURCE PHASE
    ACC_NO_ADAPT = dict()
    for source in LIST_DATASETS:
        ft_extractor = FeatureExtractor(input_size=META_DATA[source.upper()]["input_size"]).to(DEVICE)
        classifier = Classifier(feat_dim=ft_extractor.feat_dim, num_classes=META_DATA[source.upper()]["num_classes"]).to(DEVICE)
        ft_extractor_optimizer = optim.Adam(ft_extractor.parameters(), lr=1e-3, weight_decay=1e-6)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
        for epoch in range(1, num_epochs + 1):
            train(feature_extractor=ft_extractor, 
                classifier=classifier, 
                device=DEVICE, 
                train_loader=META_DATA[source.upper()]["data_loader"]['train'], 
                epoch=epoch, 
                criterion=nn.CrossEntropyLoss(), 
                ft_extractor_optimizer=ft_extractor_optimizer, 
                classifier_optimizer=classifier_optimizer)
        
        acc_no_adapt = test(ft_extractor, classifier, DEVICE, META_DATA[source.upper()]["data_loader"]['test'])
        ACC_NO_ADAPT[source.upper()] = acc_no_adapt
        
        frozen_module(ft_extractor)
        model_path = f'saved/models/{source.upper()}_ft_extractor.pth'
        torch.save(ft_extractor.state_dict(), model_path)

        META_DATA[source.upper()]["ft_extractor"] = ft_extractor
        META_DATA[source.upper()]["ft_extractor_path"] = model_path

        classifier_path = f'saved/models/{source.upper()}_classifier.pth'
        torch.save(classifier.state_dict(), classifier_path)
        META_DATA[source.upper()]["classifier_path"] = classifier_path

    acc_no_adapt_file = 'saved/stats/acc_no_adapt2.json'
    with open(acc_no_adapt_file, 'w') as json_file:
        json.dump(ACC_NO_ADAPT, json_file, indent=4)
    print(f"ACC_NO_ADAPT: {ACC_NO_ADAPT}")


def adaptation_phase(num_epochs=20):
    # TRAIN ADAPTATION PHASE
    for i in range(len(LIST_DATASETS)):
        for j in range(len(LIST_DATASETS)):

            source = LIST_DATASETS[j]
            target = LIST_DATASETS[i]

            if target == source:
                continue
            
            else:

                classifier = Classifier(feat_dim=META_DATA[target.upper()]["ft_extractor"].feat_dim, num_classes=META_DATA[target.upper()]["num_classes"]).to(DEVICE)
                classifier.load_state_dict(torch.load(META_DATA[source.upper()]["classifier_path"]))
                classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)

                for epoch in range(1, num_epochs + 1):
                    train(feature_extractor=META_DATA[source.upper()]["ft_extractor"], 
                        classifier=classifier, 
                        device=DEVICE, 
                        train_loader=META_DATA[target.upper()]["data_loader"]['train'], 
                        epoch=epoch,
                        criterion=nn.CrossEntropyLoss(), 
                        ft_extractor_optimizer=None, 
                        classifier_optimizer=classifier_optimizer)

                acc_adapt = test(META_DATA[source.upper()]["ft_extractor"], classifier, DEVICE, META_DATA[target.upper()]["data_loader"]['test'])

                if source.upper() not in ACC_ADAPT:
                    ACC_ADAPT[source.upper()] = dict()
                ACC_ADAPT[source.upper()][target.upper()] = acc_adapt

                print(f"In adaptation, source dataset {source.upper()}, target dataset: {target.upper()}, accuracy: {acc_adapt}")

    acc_adapt_file = 'saved/stats/acc_adapt2.json'
    with open(acc_adapt_file, 'w') as json_file:
        json.dump(ACC_ADAPT, json_file, indent=4)
    print(f"ACC_ADAPT: {ACC_ADAPT}")


# create_baseline(NUM_EPOCHS)
train_source_dataset_phase(20)
adaptation_phase(10)