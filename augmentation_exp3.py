import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data, load_imagenet
from models.resnet import ResNet50
from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.method5 import compute_pairwise_distance
from trainer import *
import os
import random
from datetime import datetime, timedelta
import time
from torch.utils.data import Dataset, DataLoader
import argparse


OTDD_MAXSIZE_IMAGENET = None
OTDD_MAXSIZE_CIFAR10 = None

sOTDD_MAXSIZE_IMAGENET = None
sOTDD_MAXSIZE_CIFAR10 = None


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")


def main():
    parser = argparse.ArgumentParser(description='Arguments for Augmentation')
    parser.add_argument('--parent_dir', type=str, default="saved_augmentation", help='Parent directory')
    parser.add_argument('--seed', type=int, default=1, help='Seed')

    args = parser.parse_args()
        
    parent_dir = f"{args.parent_dir}/aug_{str(args.seed)}"
    os.makedirs(parent_dir, exist_ok=True)
    result_file = f"{parent_dir}/result.txt"

    def create_data():
        brightness = random.uniform(0.1, 0.9)
        contrast = random.uniform(0.1, 0.9)
        saturation = random.uniform(0.1, 0.9)
        hue = random.uniform(0, 0.5)
        print(f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue}")
        with open(result_file, 'a') as file:
            file.write(f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue} \n")
        datadir_tiny_imagenet = "data/tiny-ImageNet/tiny-imagenet-200"
        imagenet = load_imagenet(datadir=datadir_tiny_imagenet, 
                                resize=32, 
                                tiny=True, 
                                augmentations=True, 
                                brightness=brightness, 
                                contrast=contrast, 
                                saturation=saturation, 
                                maxsize=1000,
                                hue=hue)
        imagenet_trainset = imagenet[1]["train"]
        imagenet_testset = imagenet[1]["test"]
        imagenet_trainloader = imagenet[0]["train"]
        imagenet_testloader = imagenet[0]["test"]
        
        datadir_cifar10 = "data/CIFAR10"
        cifar10 = load_torchvision_data("CIFAR10",
                                        valid_size=0, 
                                        download=False, 
                                        maxsize=1000, 
                                        datadir=datadir_cifar10)
        cifar10_trainset = cifar10[1]["train"]
        cifar10_testset = cifar10[1]["test"]
        cifar10_trainloader = cifar10[0]["train"]
        cifar10_testloader = cifar10[0]["test"]

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

        save_data(data_set=imagenet_trainset, saved_tensor_path=f"{parent_dir}/transformed_train_imagenet.pt")
        save_data(data_set=imagenet_testset, saved_tensor_path=f"{parent_dir}/transformed_test_imagenet.pt")
        save_data(data_set=cifar10_trainset, saved_tensor_path=f"{parent_dir}/transformed_train_cifar10.pt")
        save_data(data_set=cifar10_testset, saved_tensor_path=f"{parent_dir}/transformed_test_cifar10.pt")

        return {
            "cifar10": {
                    "trainset": cifar10_trainset, 
                    "testset": cifar10_testset,
                    "trainloader": cifar10_trainloader, 
                    "testloader": cifar10_testloader
                    },
            "imagenet": {
                    "trainset": imagenet_trainset,
                    "testset": imagenet_testset,
                    "trainloader": imagenet_trainloader, 
                    "testloader": imagenet_testloader
                    }
        }


    class CustomTensorDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]


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
        return dataloader



    DATA_DICT = create_data()
    print("Finish creating data")

    imagenet_trainloader = DATA_DICT["imagenet"]["trainloader"]
    for img, label in imagenet_trainloader:
        print("ImageNet 1")
        print(torch.min(img[0]), torch.max(img[0]))
        print(label[0])
        break

    cifar10_trainloader = DATA_DICT["cifar10"]["trainloader"]
    for img, label in cifar10_trainloader:
        print("CIFAR10 1")
        print(torch.min(img[0]), torch.max(img[0]))
        print(label[0])
        break


    # Compute s-OTDD
    num_projection = 10000
    kwargs = {
        "dimension": 32,
        "num_channels": 3,
        "num_moments": 10,
        "use_conv": True,
        "precision": "float",
        "p": 2,
        "chunk": 1000
    }
    list_dataset = [imagenet_trainloader, cifar10_trainloader]
    sotdd_dist = compute_pairwise_distance(list_D=list_dataset, device=DEVICE, num_projections=num_projection, evaluate_time=False, **kwargs)[0].item()
    print(sotdd_dist)



    cifar10_dataloader = get_dataloader(datadir=f'{parent_dir}/transformed_train_cifar10.pt', maxsize=1000, batch_size=64)
    imagenet_dataloader = get_dataloader(datadir=f'{parent_dir}/transformed_train_imagenet.pt', maxsize=1000, batch_size=64)

    for img, label in imagenet_dataloader:
        print("ImageNet 2")
        print(torch.min(img[0]), torch.max(img[0]))
        print(label[0])
        break

    for img, label in cifar10_dataloader:
        print("CIFAR10 2")
        print(torch.min(img[0]), torch.max(img[0]))
        print(label[0])
        break

    
    # Compute s-OTDD
    num_projection = 10000
    kwargs = {
        "dimension": 32,
        "num_channels": 3,
        "num_moments": 10,
        "use_conv": True,
        "precision": "float",
        "p": 2,
        "chunk": 1000
    }
    list_dataset = [cifar10_dataloader, imagenet_dataloader]
    sotdd_dist = compute_pairwise_distance(list_D=list_dataset, device=DEVICE, num_projections=num_projection, evaluate_time=False, **kwargs)[0].item()
    print(sotdd_dist)



if __name__ == "__main__":
    main()




    