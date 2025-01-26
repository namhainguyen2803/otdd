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


def save_data(data_set, saved_tensor_path):
    os.makedirs(os.path.dirname(saved_tensor_path), exist_ok=True)
    list_images = list()
    list_labels = list()
    for img, label in data_set:
        list_images.append(img.squeeze(1))
        list_labels.append(label)
    tensor_images = torch.cat(list_images, dim=0)
    tensor_labels = torch.cat(list_labels, dim=0)
    torch.save((tensor_images, tensor_labels), saved_tensor_path)
    print(f"Data shape: {tensor_images.shape, tensor_labels.shape}. Save data into {saved_tensor_path}")


DATASET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parent_dir = f"saved_text_data_2"
os.makedirs(parent_dir, exist_ok=True)


for dataset_name in DATASET_NAMES:
    train_dataset = load_textclassification_data(dataset_name, maxsize=None, load_tensor=True, device=DEVICE)[0]
    save_data(data_set=train_dataset, saved_tensor_path=f"{parent_dir}/{dataset_name}_text.pt")