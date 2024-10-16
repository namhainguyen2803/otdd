import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
from otdd.pytorch.method4 import compute_pairwise_distance
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
from torch.utils.data.sampler import SubsetRandomSampler
import time
from trainer import train, test_func, frozen_module
from models.resnet import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


save_dir = 'saved/compare_time'
os.makedirs(save_dir, exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
print(f"Use CUDA or not: {DEVICE}")

NUM_EPOCHS_ADAPT = 300
NUM_EPOCHS_BASELINE = 30


# class Subset(Dataset):
#     def __init__(self, dataset, original_indices, transform):

#         self._dataset = dataset
#         self._original_indices = original_indices

#         self.transform = transform
#         self.indices = torch.arange(start=0, end=len(self._original_indices), step=1)
#         self.data = self._dataset.data[self._original_indices]
#         self.targets = torch.tensor(self._dataset.targets)[self._original_indices]
#         self.classes = sorted(torch.unique(torch.tensor(self._dataset.targets)).tolist())

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         # original_idx = self.indices[idx]
#         # return self.data[idx], self.targets[idx]
#         return self.transform(self.data[idx]), self.targets[idx]


# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# ])


# dataset = CIFAR100(root='data2/CIFAR100', train=True, download=False)
# test_dataset = CIFAR100(root='data2/CIFAR100', train=False, download=False, transform=transform)

# num_splits = 10
# split_size = len(dataset) // num_splits
# print(split_size, len(dataset))
# indices = np.arange(len(dataset))


# data_index_cls = dict()
# classes = torch.unique(torch.tensor(dataset.targets))
# for cls_id in classes:
#     data_index_cls[cls_id] = indices[torch.tensor(dataset.targets) == cls_id]

# for cls_id in data_index_cls.keys():
#     np.random.shuffle(data_index_cls[cls_id])

# subsets = []
# for i in range(num_splits):

#     subset_indices = list()
#     for cls_id in data_index_cls.keys():
#         num_dataset_cls = len(data_index_cls[cls_id]) // num_splits
#         start_idx = i * num_dataset_cls
#         end_idx = min(start_idx + num_dataset_cls, len(data_index_cls[cls_id]))
#         subset_indices.extend(data_index_cls[cls_id][start_idx:end_idx])

#     np.random.shuffle(subset_indices)
#     sub = Subset(dataset=dataset, original_indices=subset_indices, transform=transform)
#     subsets.append(sub)


# dataloaders = []
# for subset in subsets:
#     dataloader = DataLoader(subset, batch_size=32, shuffle=True)
#     dataloaders.append(dataloader)


# # NEW METHOD
# print("Compute new method...")
# pairwise_dist, sotdd_list_processing_time = compute_pairwise_distance(list_dataset=dataloaders, 
#                                                                             maxsamples=None, 
#                                                                             num_projection=1000, 
#                                                                             chunk=100, num_moments=4, 
#                                                                             image_size=32, 
#                                                                             dimension=None, 
#                                                                             num_channels=3, 
#                                                                             device='cpu', 
#                                                                             dtype=torch.FloatTensor)
# pairwise_dist = torch.tensor(pairwise_dist)
# print(pairwise_dist)
# torch.save(pairwise_dist, f'{save_dir}/sotdd_dist.pt')


# # OTDD
# dict_OTDD = torch.zeros(num_splits, num_splits)
# print("Compute OTDD...")
# otdd_list_processing_time = list()
# for i in range(len(dataloaders)):
#     for j in range(i+1, len(dataloaders)):
        
#         start_time_otdd = time.time()
#         dist = DatasetDistance(dataloaders[i], 
#                                 dataloaders[j],
#                                 inner_ot_method='exact',
#                                 debiased_loss=True,
#                                 p=2, 
#                                 entreg=1e-1,
#                                 device='cpu')
#         d = dist.distance(maxsamples = None).item()
#         end_time_otdd = time.time()
#         otdd_time_taken = end_time_otdd - start_time_otdd
#         otdd_list_processing_time.append(otdd_time_taken)

#         dict_OTDD[i][j] = d
#         dict_OTDD[j][i] = d

# torch.save(dict_OTDD, f'{save_dir}/otdd_dist.pt')



# np.save(f'{save_dir}/sotdd_times.npy', np.array(sotdd_list_processing_time))
# np.save(f'{save_dir}/otdd_times.npy', np.array(otdd_list_processing_time))

sotdd_list_processing_time = np.load(f'{save_dir}/sotdd_times.npy')
otdd_list_processing_time = np.load(f'{save_dir}/otdd_times.npy')

def accumulate_list(lst):
    accu_lst = list()
    accu_lst.append(lst[0])
    for i in range(1, len(lst)):
        accu_lst.append(lst[i] + accu_lst[i - 1])
    return accu_lst

accu_sotdd_pt = accumulate_list(sotdd_list_processing_time)
accu_otdd_pt = accumulate_list(otdd_list_processing_time)

works = list(range(1, len(accu_sotdd_pt) + 1))
plt.figure(figsize=(10, 6))
plt.plot(works, accu_sotdd_pt, label='sOTDD', color='blue', marker='o', linewidth=2)
plt.plot(works, accu_otdd_pt, label='OTDD', color='orange', marker='s', linewidth=2)
plt.title('Processing Time Comparison', fontsize=20)
plt.xlabel('Number of Works', fontsize=16)
plt.ylabel('Processing Time (seconds)', fontsize=16)
plt.grid(True)
plt.legend()
plt.savefig(f"{save_dir}/comparison.pdf")
plt.savefig(f"{save_dir}/comparison.png")


