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

from otdd.pytorch.datasets import load_torchvision_data, load_imagenet
from models.resnet import ResNet18, ResNet50
from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.method5 import compute_pairwise_distance
from trainer import *
import os
import random
from datetime import datetime, timedelta
import time
from torch.utils.data import Dataset, DataLoader
import argparse


class CustomTensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def save_data(data_set, saved_tensor_path):
    list_images = list()
    list_labels = list()
    for img, label in data_set:
        list_images.append(img)
        list_labels.append(label)
    tensor_images = torch.stack(list_images)
    tensor_labels = torch.tensor(list_labels)
    torch.save((tensor_images, tensor_labels), saved_tensor_path)
    print(f"Number of data: {len(list_images)}, Save data into {saved_tensor_path}")


DATASET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]

parent_dir = f"saved_text_data"
os.makedirs(parent_dir, exist_ok=True)


for dataset_name in DATASET_NAMES:
    train_dataset = load_textclassification_data(dataset_name, maxsize=None, load_tensor=True)[0]
    save_data(data_set=train_dataset, saved_tensor_path=f"{parent_dir}/{dataset_name}_text.pt")