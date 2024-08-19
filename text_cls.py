import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_textclassification_data
from models.resnet import ResNet18, ResNet50
from otdd.pytorch.distance import DatasetDistance
from trainer import *
import os
import random
import json
from datetime import datetime, timedelta


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")

NUM_EXAMPLES = 1000

DATASET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]


# METADATA_DATASET = dict()
# for dataset_name in DATASET_NAMES:
#     METADATA_DATASET[dataset_name] = dict()
#     METADATA_DATASET[dataset_name]["dataloader"] = load_textclassification_data(dataset_name, maxsize=NUM_EXAMPLES)[0]

#     if dataset_name == "AG_NEWS":
#         METADATA_DATASET[dataset_name]["num_classes"] = 4

#     elif dataset_name == "DBpedia":
#         METADATA_DATASET[dataset_name]["num_classes"] = 14

#     elif dataset_name == "YelpReviewPolarity":
#         METADATA_DATASET[dataset_name]["num_classes"] = 2

#     elif dataset_name == "YelpReviewFull":
#         METADATA_DATASET[dataset_name]["num_classes"] = 5

#     elif dataset_name == "YahooAnswers":
#         METADATA_DATASET[dataset_name]["num_classes"] = 10

#     elif dataset_name == "AmazonReviewPolarity":
#         METADATA_DATASET[dataset_name]["num_classes"] = 2

#     elif dataset_name == "AmazonReviewFull":
#         METADATA_DATASET[dataset_name]["num_classes"] = 5
    

dt_loader = load_textclassification_data("AG_NEWS", maxsize=NUM_EXAMPLES)[0]

for x, y in dt_loader:
    print(x, y)


