import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data, load_imagenet
from models.resnet import ResNet18, ResNet50
from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.method5 import compute_pairwise_distance
from trainer import *
import os
import random
from datetime import datetime, timedelta
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")


time_difference = timedelta(hours=7)
current_utc = datetime.utcnow()
current_vietnam_time = current_utc + time_difference
current_datetime_vn = current_vietnam_time.strftime('%Y-%m-%d_%H-%M-%S')
parent_dir = f"saved/augmentation3/{current_datetime_vn}"
os.makedirs(parent_dir, exist_ok=True)
result_file = f"{parent_dir}/result.txt"


datadir_tiny_imagenet = "data/tiny-ImageNet/tiny-imagenet-200"
brightness = random.uniform(0.1, 0.9)
contrast = random.uniform(0.1, 0.9)
saturation = random.uniform(0.1, 0.9)
hue = random.uniform(0, 0.5)


print(f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue}")
with open(result_file, 'a') as file:
    file.write(f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue} \n")


MAXSIZE = 1000
imagenet_loader = load_imagenet(datadir=datadir_tiny_imagenet, 
                                resize=32, 
                                tiny=True, 
                                augmentations=True, 
                                brightness=brightness, 
                                contrast=contrast, 
                                saturation=saturation, 
                                maxsize=2000,
                                hue=hue)[0]
datadir_cifar10 = "data/CIFAR10"
cifar10_loader  = load_torchvision_data("CIFAR10",
                                        valid_size=0, 
                                        download=False, 
                                        maxsize=MAXSIZE, 
                                        datadir=datadir_cifar10)[0]

dist = DatasetDistance(imagenet_loader['train'], cifar10_loader['train'],
                        inner_ot_method='gaussian_approx',
                        sqrt_method='approximate',
                        nworkers_stats=0,
                        sqrt_niters=20,
                        debiased_loss=True,
                        p = 2, 
                        entreg = 1e-3,
                        device='cpu')

start_time = time.time()
d = dist.distance(maxsamples=MAXSIZE)
end_time = time.time()
time_taken = end_time - start_time
print(d)

with open(result_file, 'a') as file:
    file.write(f"OTDD, Distance: {d}, time taken: {time_taken} \n")



MAXSIZE = 50000
imagenet_loader = load_imagenet(datadir=datadir_tiny_imagenet, 
                                resize=32, 
                                tiny=True, 
                                augmentations=True, 
                                brightness=brightness, 
                                contrast=contrast, 
                                saturation=saturation, 
                                maxsize=MAXSIZE,
                                hue=hue)[0]
datadir_cifar10 = "data/CIFAR10"
cifar10_loader  = load_torchvision_data("CIFAR10",
                                        valid_size=0, 
                                        download=False, 
                                        maxsize=MAXSIZE, 
                                        datadir=datadir_cifar10)[0]

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

list_dataset.append(imagenet_loader)
list_dataset.append(cifar10_loader)

start_time = time.time()
sotdd_dist = compute_pairwise_distance(list_D=list_dataset, device=DEVICE, num_projections=num_projection, evaluate_time=False, **kwargs)[0].item()
end_time = time.time()
time_taken = end_time - start_time
print(sotdd_dist)

with open(result_file, 'a') as file:
    file.write(f"s-OTDD, Distance: {sotdd_dist}, time taken: {time_taken} \n")

del imagenet_loader
del cifar10_loader


MAXSIZE = None

imagenet_loader = load_imagenet(datadir=datadir_tiny_imagenet, 
                                resize=32, 
                                tiny=True, 
                                augmentations=True, 
                                brightness=brightness, 
                                contrast=contrast, 
                                saturation=saturation, 
                                maxsize=MAXSIZE,
                                hue=hue)[0]

datadir_cifar10 = "data/CIFAR10"
cifar10_loader  = load_torchvision_data("CIFAR10", 
                                        valid_size=0, 
                                        download=False, 
                                        maxsize=MAXSIZE, 
                                        datadir=datadir_cifar10)[0]

num_epochs = 300

imagenet_feature_extractor = ResNet50().to(DEVICE)
imagenet_classifier = nn.Linear(imagenet_feature_extractor.latent_dims, 200).to(DEVICE)

feature_extractor_optimizer = optim.SGD(imagenet_feature_extractor.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
classifier_optimizer = optim.SGD(imagenet_classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

for epoch in range(1, num_epochs + 1):
    train(feature_extractor=imagenet_feature_extractor,
        classifier=imagenet_classifier,
        device=DEVICE,
        train_loader=imagenet_loader['train'],
        epoch=epoch,
        criterion=nn.CrossEntropyLoss(),
        ft_extractor_optimizer=feature_extractor_optimizer,
        classifier_optimizer=classifier_optimizer)

imagenet_acc_no_adapt = test(imagenet_feature_extractor, imagenet_classifier, DEVICE, imagenet_loader['test'])
print(f"Accuracy when no pretrainng {imagenet_acc_no_adapt}")

with open(result_file, 'a') as file:
    file.write(f"Accuracy when no pretrainng {imagenet_acc_no_adapt} \n")

frozen_module(imagenet_feature_extractor)

ft_extractor_path = f'{parent_dir}/imagenet_ft_extractor.pth'
torch.save(imagenet_feature_extractor.state_dict(), ft_extractor_path)


cifar10_classifier = nn.Linear(imagenet_feature_extractor.latent_dims, 10).to(DEVICE)

cifar10_classifier_optimizer = optim.SGD(cifar10_classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

num_epochs = 30

for epoch in range(1, num_epochs + 1):
    train(feature_extractor=imagenet_feature_extractor,
        classifier=cifar10_classifier,
        device=DEVICE,
        train_loader=cifar10_loader['train'],
        epoch=epoch,
        criterion=nn.CrossEntropyLoss(),
        ft_extractor_optimizer=None,
        classifier_optimizer=cifar10_classifier_optimizer)


cifar10_acc_adapt = test(imagenet_feature_extractor, cifar10_classifier, DEVICE, cifar10_loader['test'])
print(f"Accuracy when having pretraned feature extractor, evaluated on CIFAR10 dataset: {cifar10_acc_adapt}")

with open(result_file, 'a') as file:
    file.write(f"Accuracy when having pretraned feature extractor, evaluated on CIFAR10 dataset: {cifar10_acc_adapt} \n")

