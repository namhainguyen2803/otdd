import torch
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

class Subset(Dataset):
    def __init__(self, dataset, original_indices, transform):

        self._dataset = dataset
        self._original_indices = original_indices

        self.transform = transform
        self.indices = torch.arange(start=0, end=len(self._original_indices), step=1)
        self.data = self._dataset.data[self._original_indices]
        self.targets = torch.tensor(self._dataset.targets)[self._original_indices]
        self.classes = sorted(torch.unique(torch.tensor(self._dataset.targets)).tolist())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # original_idx = self.indices[idx]
        # return self.data[idx], self.targets[idx]
        return self.transform(self.data[idx]), self.targets[idx]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

dataset = CIFAR100(root='data2/CIFAR100', train=True, download=False)
test_dataset = CIFAR100(root='data2/CIFAR100', train=False, download=False, transform=transform)

num_splits = 10
split_size = len(dataset) // num_splits
print(split_size, len(dataset))
indices = np.arange(len(dataset))



data_index_cls = dict()
classes = torch.unique(torch.tensor(dataset.targets))
for cls_id in classes:
    data_index_cls[cls_id] = indices[torch.tensor(dataset.targets) == cls_id]

for cls_id in data_index_cls.keys():
    print(cls_id, len(data_index_cls[cls_id]))
    np.random.shuffle(data_index_cls[cls_id])

subsets = []
for i in range(num_splits):

    subset_indices = list()
    for cls_id in data_index_cls.keys():
        num_dataset_cls = len(data_index_cls[cls_id]) // num_splits
        start_idx = i * num_dataset_cls
        end_idx = min(start_idx + num_dataset_cls, len(data_index_cls[cls_id]))
        subset_indices.extend(data_index_cls[cls_id][start_idx:end_idx])

    np.random.shuffle(subset_indices)
    sub = Subset(dataset=dataset, original_indices=subset_indices, transform=transform)
    subsets.append(sub)

dataloaders = []
for subset in subsets:
    dataloader = DataLoader(subset, batch_size=32, shuffle=True)
    dataloaders.append(dataloader)

# for img, label in dataloaders[0]:
#     break
# print(img[0], label[0])

# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
# for img, label in test_loader:
#     break
# print(img[0], label[0])

save_dir = 'saved/split_cifar100'
os.makedirs(save_dir, exist_ok=True)

for i, subset in enumerate(subsets):
    with open(os.path.join(save_dir, f'subset_{i}.pkl'), 'wb') as f:
        pickle.dump(subset, f)

print(f'Saved {num_splits} subsets in {save_dir}')


loaded_subsets = []
for i in range(num_splits):
    with open(os.path.join(save_dir, f'subset_{i}.pkl'), 'rb') as f:
        loaded_subset = pickle.load(f)
        loaded_subsets.append(loaded_subset)

# Create DataLoaders for the loaded subsets
loaded_dataloaders = [DataLoader(subset, batch_size=32, shuffle=True) for subset in loaded_subsets]


# NEW METHOD
print("Compute new method...")
start_time_new_method = time.time()
pairwise_dist = compute_pairwise_distance(list_dataset=dataloaders, 
                                            maxsamples=None, 
                                            num_projection=1000, 
                                            chunk=100, num_moments=4, 
                                            image_size=32, 
                                            dimension=None, 
                                            num_channels=3, 
                                            device='cpu', 
                                            dtype=torch.FloatTensor)
end_time_new_method = time.time()
new_method_time_taken = end_time_new_method - start_time_new_method
print(f"Finish computing new method. Time taken: {new_method_time_taken:.2f} seconds")
pairwise_dist = torch.tensor(pairwise_dist)
print(pairwise_dist)
torch.save(pairwise_dist, f'{save_dir}/new_method_dist.pt')


# OTDD
dict_OTDD = torch.zeros(num_splits, num_splits)
print("Compute new method...")
start_time_otdd = time.time()
for i in range(len(dataloaders)):
    for j in range(i+1, len(dataloaders)):

        dist = DatasetDistance(dataloaders[i], 
                                dataloaders[j],
                                inner_ot_method='exact',
                                debiased_loss=True,
                                p=2, 
                                entreg=1e-1,
                                device='cpu')
        d = dist.distance(maxsamples = None).item()

        dict_OTDD[i][j] = d
        dict_OTDD[j][i] = d
end_time_otdd = time.time()
otdd_time_taken = end_time_otdd - start_time_otdd
print(f"Finish computing new method. Time taken: {otdd_time_taken:.2f} seconds")
torch.save(dict_OTDD, f'{save_dir}/otdd_dist.pt')


