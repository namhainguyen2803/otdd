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
# DEVICE = "cpu"
print(f"Use CUDA or not: {DEVICE}")
torch.manual_seed(42)

# ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
# DATASET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
# TARGET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]


def main():

    parser = argparse.ArgumentParser(description='Arguments for sOTDD and OTDD computations')
    parser.add_argument('--saved_path', type=str, default="saved_text_dist", help='Name of method')
    parser.add_argument('--source', type=str, default="AG_NEWS", help='Source dataset')
    parser.add_argument('--target', type=str, default="DBpedia", help='Target dataset')
    parser.add_argument('--method', type=str, default="sotdd", help='Name of method')
    parser.add_argument('--num_examples', type=int, default=1000, help='number of examples')
    args = parser.parse_args()

    os.makedirs(args.saved_path, exist_ok=True)

    DATASET_NAMES = [args.source, args.target]

    METADATA_DATASET = dict()
    for dataset_name in DATASET_NAMES:
        METADATA_DATASET[dataset_name] = dict()
        METADATA_DATASET[dataset_name]["dataloader"] = load_textclassification_data(dataset_name, maxsize=args.num_examples, load_tensor=True)[0]

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


    if args.method == "otdd":
        print("Computing OTDD...")
        # dist = DatasetDistance(METADATA_DATASET[args.source]["dataloader"], 
        #                         METADATA_DATASET[args.target]["dataloader"],
        #                         inner_ot_method='gaussian_approx',
        #                         sqrt_method='approximate',
        #                         nworkers_stats=0,
        #                         sqrt_niters=20,
        #                         debiased_loss=True,
        #                         p=2,
        #                         entreg=1e-3,
        #                         device=DEVICE)
        dist = DatasetDistance(METADATA_DATASET[args.source]["dataloader"], 
                                METADATA_DATASET[args.target]["dataloader"],
                                inner_ot_method='exact',
                                debiased_loss=True,
                                p=2,
                                entreg=1e-3,
                                device=DEVICE)
        d = dist.distance(maxsamples=None)
        del dist
        print(f"Data source: {args.source}, Data target: {args.target}, Distance: {d}")
        with open(f'{args.saved_path}/otdd_exact_distance.txt', 'a') as file:
            file.write(f"Data source: {args.source}, Data target: {args.target}, Distance: {d} \n")


    if args.method == "sotdd":
        print("Computing s-OTDD...")
        NUM_MOMENTS = 5
        PROJ = 10000

        list_dataset = [METADATA_DATASET[args.source]["dataloader"], METADATA_DATASET[args.target]["dataloader"]]
        kwargs = {
            "dimension": 768,
            "num_channels": 1,
            "num_moments": NUM_MOMENTS,
            "use_conv": False,
            "precision": "float",
            "p": 2,
            "chunk": 1000
        }

        sw_list = compute_pairwise_distance(list_D=list_dataset, device=DEVICE, num_projections=PROJ, evaluate_time=False, **kwargs)[0]

        print(f"Data source: {args.source}, Data target: {args.target}, Distance: {d}")
        with open(f'{args.saved_path}/sotdd_num_moments_{NUM_MOMENTS}_projections_{PROJ}_distance.txt', 'a') as file:
            file.write(f"Data source: {args.source}, Data target: {args.target}, Distance: {d} \n")


if __name__ == "__main__":
    main()
