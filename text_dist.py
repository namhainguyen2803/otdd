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
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")

NUM_EXAMPLES = 2000

# ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
DATASET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
TARGET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]

parent_dir = f"saved_text_dist/text_cls/dist"
os.makedirs(parent_dir, exist_ok=True)

method = None
method2 = None
# method = "OTDD"
method2 = "sOTDD"

def main():

    METADATA_DATASET = dict()
    for dataset_name in DATASET_NAMES:
        METADATA_DATASET[dataset_name] = dict()
        METADATA_DATASET[dataset_name]["dataloader"] = load_textclassification_data(dataset_name, maxsize=NUM_EXAMPLES, load_tensor=True)[0]

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


    if method == "OTDD":
        print("Computing OTDD...")
        OTDD_DIST = dict()
        for i in range(len(DATASET_NAMES)):
            for j in range(len(TARGET_NAMES)):
                data_source = DATASET_NAMES[i]
                data_target = DATASET_NAMES[j]
                if data_source not in OTDD_DIST:
                    OTDD_DIST[data_source] = dict()
                OTDD_DIST[data_source][data_target] = 0

        for i in range(len(DATASET_NAMES)):
            for j in range(i + 1, len(TARGET_NAMES)):
                data_source = DATASET_NAMES[i]
                data_target = DATASET_NAMES[j]
                if data_source == data_target:
                    continue
                dist = DatasetDistance(METADATA_DATASET[data_source]["dataloader"], 
                                        METADATA_DATASET[data_target]["dataloader"],
                                        inner_ot_method='gaussian_approx',
                                        sqrt_method='approximate',
                                        nworkers_stats=0,
                                        sqrt_niters=20,
                                        debiased_loss=True,
                                        p=2,
                                        entreg=1e-3,
                                        device=DEVICE)
                d = dist.distance(maxsamples=None)
                del dist
                OTDD_DIST[data_target][data_source] = d.item()
                OTDD_DIST[data_source][data_target] = d.item()
                print(f"Data source: {data_source}, Data target: {data_target}, Distance: {d}")

        dist_file_path = f'{parent_dir}/{method}_20_text_dist.json'
        with open(dist_file_path, 'w') as json_file:
            json.dump(OTDD_DIST, json_file, indent=4)
        print(f"Finish computing OTDD")


    if method2 == "sOTDD":
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
            "num_moments": 10,
            "use_conv": False,
            "precision": "float",
            "p": 2,
            "chunk": 1000
        }

        sw_list = compute_pairwise_distance(list_D=list_dataset, device='cpu', num_projections=10000, evaluate_time=False, **kwargs)

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

        dist_file_path = f'{parent_dir}/{method2}_text_dist.json'
        with open(dist_file_path, 'w') as json_file:
            json.dump(sOTDD_DIST, json_file, indent=4)
        print(f"Finish computing s-OTDD")


if __name__ == "__main__":
    main()

