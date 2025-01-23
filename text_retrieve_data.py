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
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
print(f"Use CUDA or not: {DEVICE}")

# NUM_EXAMPLES = 2000

# ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]

DATASET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
TARGET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]

parent_dir = f"saved_text_dist_final"
os.makedirs(parent_dir, exist_ok=True)

# method = None
# method2 = None
# method = "OTDD"
# method2 = "sOTDD"

def save_data(data_set, saved_tensor_path):
    os.makedirs(os.path.dirname(saved_tensor_path), exist_ok=True)
    list_images = list()
    list_labels = list()
    for img, label in data_set:
        img = img.squeeze(1)
        list_images.append(img)
        list_labels.append(label)

    tensor_images = torch.cat(list_images, dim=0)
    tensor_labels = torch.cat(list_labels, dim=0)
    torch.save((tensor_images, tensor_labels), saved_tensor_path)
    print(f"Shape of data: {tensor_images.shape, tensor_labels.shape}. Save data into {saved_tensor_path}")


def main():

    parser = argparse.ArgumentParser(description='Arguments for sOTDD and OTDD computations')
    parser.add_argument('--dataset_name', type=str, default="sotdd", help="Method name")
    parser.add_argument('--max_size', type=int, default=10000, help='Sie')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    max_size = args.max_size 

    dataset = load_textclassification_data(dataset_name, maxsize=10000, load_tensor=True, batch_size=128, device=DEVICE)[0]
    print("Cac")
    save_data(data_set=dataset, saved_tensor_path=f"saved_text_dataset/{dataset_name}.pt")
    


if __name__ == "__main__":
    main()
