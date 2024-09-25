import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data, load_imagenet
from models.resnet import ResNet18, ResNet50
from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.method3 import NewDatasetDistance
from trainer import *
import os
import random
from datetime import datetime, timedelta
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")


datadir_tiny_imagenet = "data/tiny-ImageNet/tiny-imagenet-200"
brightness = random.uniform(0.1, 0.9)
contrast = random.uniform(0.1, 0.9)
saturation = random.uniform(0.1, 0.9)
hue = random.uniform(0, 0.5)


print(f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue}")
with open(result_file, 'a') as file:
    file.write(f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue} \n")


MAXSIZE = 2000
imagenet_loader = load_imagenet(datadir=datadir_tiny_imagenet, 
                                resize=32, 
                                tiny=True, 
                                augmentations=True, 
                                brightness=brightness, 
                                contrast=contrast, 
                                saturation=saturation, 
                                maxsize=MAXSIZE,
                                hue=hue)[0]