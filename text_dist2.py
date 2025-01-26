import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_textclassification_data
from models.resnet import ResNet18, ResNet50
from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.method5 import compute_pairwise_distance
from trainer import *
import os
import random
import json
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
import argparse
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
print(f"Use CUDA or not: {DEVICE}")
# torch.manual_seed(42)
data_dir = "saved_text_data_2"

# ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
DATASET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
TARGET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]


class CustomTensorDataset(Dataset):
    def __init__(self, data, labels):

        # data_mean = data.mean()
        # data_std = data.std()
        # self.data = (data - data_mean) / data_std
        self.data = data
        self.labels = labels
        self.targets = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

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
    parser.add_argument('--max_size', type=int, default=2000, help='Size')
    args = parser.parse_args()

    method = args.method
    max_size = args.max_size

    METADATA_DATASET = dict()
    for dataset_name in DATASET_NAMES:
        METADATA_DATASET[dataset_name] = dict()
        METADATA_DATASET[dataset_name]["dataloader"] = get_dataloader(datadir=f"{data_dir}/{dataset_name}_text.pt", maxsize=max_size, batch_size=64)

        if dataset_name == "AG_NEWS":
            METADATA_DATASET[dataset_name]["num_classes"] = 4

        elif dataset_name == "DBpedia":
            METADATA_DATASET[dataset_name]["num_classes"] = 14

        elif dataset_name == "YelpReviewPolarity":
            METADATA_DATASET[dataset_name]["num_classes"] = 2

        elif dataset_name == "YelpReviewFull":
            METADATA_DATASET[dataset_name]["num_classes"] = 5

        elif dataset_name == "YahooAnswers":
            METADATA_DATASET[dataset_name]["num_classes"] = 10

        elif dataset_name == "AmazonReviewPolarity":
            METADATA_DATASET[dataset_name]["num_classes"] = 2

        elif dataset_name == "AmazonReviewFull":
            METADATA_DATASET[dataset_name]["num_classes"] = 5


    if method == "otdd":
        print("Computing OTDD...")
        OTDD_DIST = dict()
        for i in range(len(DATASET_NAMES)):
            for j in range(len(TARGET_NAMES)):
                data_source = DATASET_NAMES[i]
                data_target = DATASET_NAMES[j]
                if data_source not in OTDD_DIST:
                    OTDD_DIST[data_source] = dict()
                OTDD_DIST[data_source][data_target] = 0

        start = time.time()
        for i in range(len(DATASET_NAMES)):
            for j in range(i + 1, len(TARGET_NAMES)):
                data_source = DATASET_NAMES[i]
                data_target = DATASET_NAMES[j]
                if data_source == data_target:
                    continue
                # dist = DatasetDistance(METADATA_DATASET[data_source]["dataloader"], 
                #                         METADATA_DATASET[data_target]["dataloader"],
                #                         inner_ot_method='gaussian_approx',
                #                         sqrt_method='approximate',
                #                         nworkers_stats=0,
                #                         sqrt_niters=20,
                #                         debiased_loss=True,
                #                         p=2,
                #                         entreg=1e-3,
                #                         device=DEVICE)
                dist = DatasetDistance(METADATA_DATASET[data_source]["dataloader"], 
                                        METADATA_DATASET[data_target]["dataloader"],
                                        inner_ot_method='exact',
                                        debiased_loss=True,
                                        p=2,
                                        entreg=1e-3,
                                        device=DEVICE)
                d = dist.distance(maxsamples=None)
                del dist
                OTDD_DIST[data_target][data_source] = d.item()
                OTDD_DIST[data_source][data_target] = d.item()
                print(f"Data source: {data_source}, Data target: {data_target}, Distance: {d}")
        end = time.time()
        processing_time = end - start

        dist_file_path = f'{data_dir}/otdd_exact_text_dist_num_examples_{max_size}.json'
        with open(dist_file_path, 'w') as json_file:
            json.dump(OTDD_DIST, json_file, indent=4)
        print(f"Finish computing OTDD")


    elif method == "sotdd":
        print("Computing s-OTDD...")
        sOTDD_DIST = dict()
        for i in range(len(DATASET_NAMES)):
            for j in range(len(TARGET_NAMES)):
                data_source = DATASET_NAMES[i]
                data_target = DATASET_NAMES[j]
                if data_source not in sOTDD_DIST:
                    sOTDD_DIST[data_source] = dict()
                sOTDD_DIST[data_source][data_target] = 0

        list_dataset = list()
        for i in range(len(DATASET_NAMES)):
            list_dataset.append(METADATA_DATASET[DATASET_NAMES[i]]["dataloader"])

        kwargs = {
            "dimension": 768,
            "num_channels": 1,
            "num_moments": 5,
            "use_conv": False,
            "precision": "float",
            "p": 2,
            "chunk": 1000
        }

        sw_list, processing_time = compute_pairwise_distance(list_D=list_dataset, device=DEVICE, num_projections=10000, evaluate_time=True, **kwargs)

        k = 0
        for i in range(len(DATASET_NAMES)):
            for j in range(i + 1, len(TARGET_NAMES)):
                data_source = DATASET_NAMES[i]
                data_target = DATASET_NAMES[j]
                if data_source == data_target:
                    continue
                sOTDD_DIST[data_target][data_source] = sw_list[k].item()
                sOTDD_DIST[data_source][data_target] = sw_list[k].item()
                k += 1
        
        assert k == len(sw_list), "k != len(sw_list)"

        dist_file_path = f'{data_dir}/sotdd_text_dist_num_moments_{kwargs["num_moments"]}_num_examples_{max_size}.json'
        with open(dist_file_path, 'w') as json_file:
            json.dump(sOTDD_DIST, json_file, indent=4)
        print(f"Finish computing s-OTDD")

    with open(f'{data_dir}/running_time.txt', 'w') as file:
        file.write(f"Method: {method}, total time processing: {processing_time} \n")
    


if __name__ == "__main__":
    main()
