import torch
import math
# import torch.optim as optim
# import torch.nn as nn
# from otdd.pytorch.datasets import load_torchvision_data, load_imagenet
# from models.resnet import ResNet18, ResNet50
# from otdd.pytorch.distance import DatasetDistance
# from otdd.pytorch.method3 import NewDatasetDistance
# from trainer import *
# import os
# import random
# from datetime import datetime, timedelta
# import time


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Use CUDA or not: {DEVICE}")


# datadir_tiny_imagenet = "data/tiny-ImageNet/tiny-imagenet-200"
# brightness = random.uniform(0.1, 0.9)
# contrast = random.uniform(0.1, 0.9)
# saturation = random.uniform(0.1, 0.9)
# hue = random.uniform(0, 0.5)


# print(f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue}")
# with open(result_file, 'a') as file:
#     file.write(f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue} \n")


# MAXSIZE = 2000
# imagenet_loader = load_imagenet(datadir=datadir_tiny_imagenet, 
#                                 resize=32, 
#                                 tiny=True, 
#                                 augmentations=True, 
#                                 brightness=brightness, 
#                                 contrast=contrast, 
#                                 saturation=saturation, 
#                                 maxsize=MAXSIZE,
#                                 hue=hue)[0]


# def generate_moments(num_moments, min_moment=1, max_moment=None, gen_type="uniform"):
#     assert gen_type in ("uniform", "poisson", "fixed")

#     if gen_type == "fixed":
#         return torch.arange(num_moments) + 1

#     elif gen_type == "uniform":
#         return torch.sort(torch.randperm(max_moment)[:num_moments])[0] + min_moment

#     elif gen_type == "poisson":

#         if max_moment is not None:
#             mean_moment = (max_moment + 2 * min_moment) / 3
#         else:
#             mean_moment = 5

#         print(mean_moment)
#         moment = torch.sort(torch.poisson(torch.ones(num_moments) * mean_moment))[0]

#         if max_moment is not None:
#             moment[moment > max_moment] = max_moment
#         moment[moment < min_moment] = min_moment

#         return moment


# num_moments = 8
# moment = torch.stack([generate_moments(num_moments=num_moments, min_moment=1, max_moment=None, gen_type="poisson") for _ in range(5)])

# x = torch.randn(5, num_moments)

# print(torch.sign(x) * torch.pow(torch.abs(x), 1/moment))





x = torch.randint(10, (5, 5)) + 1

y = torch.unique(x)

factorial_y = list()
for i in range(len(y)):
    factorial_y.append(math.factorial(y[i]))

print(y)
print(factorial_y)

print(x)
for i in range(len(y)):
    x[x == y[i]] = factorial_y[i]

print(x)