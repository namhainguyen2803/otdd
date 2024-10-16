import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_textclassification_data
from models.resnet import ResNet18, ResNet50
from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.method4 import NewDatasetDistance, compute_pairwise_distance
from trainer import *
import os
import random
import json
from datetime import datetime, timedelta
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")

NUM_EXAMPLES = 500

# ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
DATASET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
TARGET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]

parent_dir = f"saved/text_cls_new2/dist"
os.makedirs(parent_dir, exist_ok=True)

method = "sOTDD"

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
        
    DATA_DIST = dict()
    for i in range(len(DATASET_NAMES)):
        for j in range(len(TARGET_NAMES)):

            data_source = DATASET_NAMES[i]
            data_target = DATASET_NAMES[j]

            if data_source not in DATA_DIST:
                DATA_DIST[data_source] = dict()

            DATA_DIST[data_source][data_target] = 0

    
    for i in range(len(DATASET_NAMES)):

        for j in range(i + 1, len(TARGET_NAMES)):

            data_source = DATASET_NAMES[i]
            data_target = DATASET_NAMES[j]
            
            if data_source == data_target:
                continue

            if method == "sOTDD":
                dist = NewDatasetDistance(METADATA_DATASET[data_source]["dataloader"], 
                                        METADATA_DATASET[data_target]["dataloader"], 
                                        p=2, 
                                        device='cpu')
                d = dist.distance(maxsamples=None, num_projection=10000, use_conv=False)

            else:
                dist = DatasetDistance(METADATA_DATASET[data_source]["dataloader"], 
                                        METADATA_DATASET[data_target]["dataloader"],
                                        inner_ot_method = 'gaussian_approx',
                                        debiased_loss = True,
                                        p = 2, entreg = 1e-2,
                                        device='cpu')
                d = dist.distance(maxsamples=None)

            del dist

            DATA_DIST[data_target][data_source] = d.item()
            DATA_DIST[data_source][data_target] = d.item()
            
            print(f"Data source: {data_source}, Data target: {data_target}, Distance: {d}")

    dist_file_path = f'{parent_dir}/{method}_text_dist3.json'
    with open(dist_file_path, 'w') as json_file:
        json.dump(DATA_DIST, json_file, indent=4)
    print(f"DIST: {DATA_DIST}")


if __name__ == "__main__":
    main()

