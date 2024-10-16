import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.method4 import NewDatasetDistance
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

parent_dir = f"saved/nist"
pretrained_path = parent_dir + "/pretrained_weights"
adapt_path = parent_dir + "/finetune_weights"

os.makedirs(parent_dir, exist_ok=True)
os.makedirs(pretrained_path, exist_ok=True)
os.makedirs(adapt_path, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
MAXSIZE_DIST = 2000
MAXSIZE_TRAINING = None


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
        METADATA_DATASET[dataset_name]["ft_extractor_path"] = f'{pretrained_path}/{dataset_name}_ft_extractor.pth'

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


def compute_distance_dataset(maxsamples=MAXSIZE_DIST, num_projection=1000):

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

            # dist = NewDatasetDistance(source_dataset, target_dataset, p=2, device='cpu')
            # d = dist.distance(maxsamples=maxsamples, num_projection=num_projection)

            dist = DatasetDistance(METADATA_DATASET[source_dataset]["train_loader"], 
                                    METADATA_DATASET[target_dataset]["train_loader"],
                                    inner_ot_method='exact',
                                    debiased_loss = True,
                                    p = 2, entreg = 1e-1,
                                    device='cpu')
            d = dist.distance(maxsamples = maxsamples).item()

            all_dist_dict[source_dataset][target_dataset] = d
            all_dist_dict[target_dataset][source_dataset] = d

            print(f'DIST({source_dataset}, {target_dataset})={d:8.2f}')

    return all_dist_dict


def train_source(num_epoch_source=20, maxsamples=MAXSIZE_TRAINING):

    METADATA_DATASET = create_dataset(maxsamples=maxsamples)

    # TRAIN SOURCE PHASE
    ACC_NO_ADAPT = dict()
    for source in LIST_DATASETS:
        ft_extractor = FeatureExtractor(input_size=28).to(DEVICE)
        classifier = Classifier(feat_dim=ft_extractor.feat_dim, num_classes=METADATA_DATASET[source]["num_classes"]).to(DEVICE)
        ft_extractor_optimizer = optim.Adam(ft_extractor.parameters(), lr=1e-3, weight_decay=1e-6)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
        for epoch in range(1, num_epoch_source + 1):
            train(feature_extractor=ft_extractor, 
                classifier=classifier, 
                device=DEVICE, 
                train_loader=METADATA_DATASET[source]["train_loader"], 
                epoch=epoch, 
                criterion=nn.CrossEntropyLoss(), 
                ft_extractor_optimizer=ft_extractor_optimizer, 
                classifier_optimizer=classifier_optimizer)
        
        acc_no_adapt = test(ft_extractor, classifier, DEVICE, METADATA_DATASET[source]["test_loader"])
        ACC_NO_ADAPT[source] = acc_no_adapt
        
        ft_extractor_path = f'{pretrained_path}/{source}_ft_extractor.pth'
        torch.save(ft_extractor.state_dict(), ft_extractor_path)

        classifier_path = f'{pretrained_path}/{source}_classifier.pth'
        torch.save(classifier.state_dict(), classifier_path)


    acc_no_adapt_file = f'{pretrained_path}/acc_train_source.json'
    with open(acc_no_adapt_file, 'w') as json_file:
        json.dump(ACC_NO_ADAPT, json_file, indent=4)
    print(f"ACC_NO_ADAPT: {ACC_NO_ADAPT}")


def training_and_adaptation(num_epochs=10, maxsamples=MAXSIZE_TRAINING):
    # BASELINE PHASE
    METADATA_DATASET = create_dataset(maxsamples=maxsamples)
    ACC_NO_ADAPT = dict()
    for source in LIST_DATASETS:
        ft_extractor = FeatureExtractor(input_size=28).to(DEVICE)
        # ft_extractor_optimizer = optim.Adam(ft_extractor.parameters(), lr=1e-3, weight_decay=1e-6)
        classifier = Classifier(feat_dim=ft_extractor.feat_dim, num_classes=METADATA_DATASET[source]["num_classes"]).to(DEVICE)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
        for epoch in range(1, num_epochs + 1):
            train(feature_extractor=ft_extractor, 
                classifier=classifier, 
                device=DEVICE, 
                train_loader=METADATA_DATASET[source]["train_loader"], 
                epoch=epoch,
                criterion=nn.CrossEntropyLoss(),
                ft_extractor_optimizer=None, 
                classifier_optimizer=classifier_optimizer)
        
        acc_baseline = test(ft_extractor, classifier, DEVICE, METADATA_DATASET[source]["test_loader"])

        ACC_NO_ADAPT[source] = acc_baseline

        print(f"In creating baseline, dataset {source}, accuracy: {acc_baseline}")

    acc_baseline = f'{adapt_path}/acc_baseline.json'
    with open(acc_baseline, 'w') as json_file:
        json.dump(ACC_NO_ADAPT, json_file, indent=4)
    print(f"ACC_BASELINE: {ACC_NO_ADAPT}")


    # TRAIN ADAPTATION PHASE
    for i in range(len(LIST_DATASETS)):
        
        source = LIST_DATASETS[i]

        for j in range(len(LIST_DATASETS)):

            target = LIST_DATASETS[j]

            if target == source:
                continue
            
            else:

                ft_extractor = FeatureExtractor(input_size=28).to(DEVICE)
                ft_extractor.load_state_dict(torch.load(METADATA_DATASET[source]["ft_extractor_path"]))
                ft_extractor_optimizer = optim.Adam(ft_extractor.parameters(), lr=1e-3, weight_decay=1e-6)

                classifier = Classifier(feat_dim=ft_extractor.feat_dim, num_classes=METADATA_DATASET[target]["num_classes"]).to(DEVICE)
                classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)

                for epoch in range(1, num_epochs + 1):
                    train(feature_extractor=ft_extractor, 
                        classifier=classifier, 
                        device=DEVICE, 
                        train_loader=METADATA_DATASET[target]["train_loader"], 
                        epoch=epoch,
                        criterion=nn.CrossEntropyLoss(), 
                        ft_extractor_optimizer=ft_extractor_optimizer, 
                        classifier_optimizer=classifier_optimizer)

                ft_extractor_path = f'{adapt_path}/{source}_{target}_ft_extractor.pth'
                torch.save(ft_extractor.state_dict(), ft_extractor_path)

                classifier_path = f'{adapt_path}/{source}_{target}_classifier.pth'
                torch.save(classifier.state_dict(), classifier_path)

                acc_adapt = test(ft_extractor, classifier, DEVICE, METADATA_DATASET[target]["test_loader"])

                if source not in ACC_ADAPT:
                    ACC_ADAPT[source] = dict()
                ACC_ADAPT[source][target] = acc_adapt

                print(f"In adaptation, source dataset {source}, target dataset: {target}, accuracy: {acc_adapt}")

    acc_adapt_file = f'{adapt_path}/acc_adapt.json'
    with open(acc_adapt_file, 'w') as json_file:
        json.dump(ACC_ADAPT, json_file, indent=4)
    print(f"ACC_ADAPT: {ACC_ADAPT}")


if __name__ == "__main__":
    # DIST = compute_distance_dataset()

    # dist_file_path = f'{parent_dir}/nist_dist.json'
    # with open(dist_file_path, 'w') as json_file:
    #     json.dump(DIST, json_file, indent=4)
    # print(f"DIST: {DIST}")

    # train_source(num_epoch_source=20, maxsamples=MAXSIZE_TRAINING)
    training_and_adaptation(num_epochs=10)




