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
from trainer import train, test
from models.resnet import *

save_dir = 'saved/split_cifar100'
os.makedirs(save_dir, exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")


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
test_loader = DataLoader(test_dataset, batch_size=32)

for i, subset in enumerate(subsets):
    cac = os.path.join(save_dir, f'{i}')
    os.makedirs(cac, exist_ok=True)
    with open(cac + f'/data.pkl', 'wb') as f:
        pickle.dump(subset, f)

print(f'Saved {num_splits} subsets in {save_dir}')


# loaded_subsets = []
# for i in range(num_splits):
#     with open(os.path.join(save_dir, f'{i}/data.pkl'), 'rb') as f:
#         loaded_subset = pickle.load(f)
#         loaded_subsets.append(loaded_subset)


# Create DataLoaders for the loaded subsets
# loaded_dataloaders = [DataLoader(subset, batch_size=32, shuffle=True) for subset in loaded_subsets]
# test_loader = DataLoader(test_dataset, batch_size=32)


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


def train_baseline(train_loader, test_loader, num_epochs=300, device=DEVICE):

    cifar10_feature_extractor = ResNet34().to(device)
    cifar10_classifier = nn.Linear(cifar10_feature_extractor.latent_dims, 10).to(device)

    feature_extractor_optimizer = optim.SGD(cifar10_feature_extractor.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    classifier_optimizer = optim.SGD(cifar10_classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    for epoch in range(1, num_epochs + 1):
        train(feature_extractor=cifar10_feature_extractor,
            classifier=cifar10_classifier,
            device=device,
            train_loader=train_loader,
            epoch=epoch,
            criterion=nn.CrossEntropyLoss(),
            ft_extractor_optimizer=feature_extractor_optimizer,
            classifier_optimizer=classifier_optimizer)

    cifar10_acc_no_adapt = test(cifar10_feature_extractor, cifar10_classifier, device, test_loader)
    print(f"Accuracy when no pretrainng {cifar10_acc_no_adapt}")

    return cifar10_feature_extractor, cifar10_classifier, cifar10_acc_no_adapt




meta_dict = dict()

for i in range(len(dataloaders)):
    train_loader = dataloaders[i]

    cifar10_feature_extractor, cifar10_classifier, cifar10_acc_no_adapt = train_baseline(train_loader=train_loader, 
                                                                                        test_loader=test_loader, 
                                                                                        num_epochs=300, 
                                                                                        device=DEVICE)

    frozen_module(cifar10_feature_extractor)

    ft_extractor_path = f'{save_dir}/{i}/ft_extractor.pth'
    torch.save(cifar10_feature_extractor.state_dict(), ft_extractor_path)

    classifier_path = f'{save_dir}/{i}/classifier.pth'
    torch.save(cifar10_classifier.state_dict(), classifier_path)

    meta_dict[i] = {
        "ft_extractor": ft_extractor_path,
        "classifier": classifier_path,
        "baseline": cifar10_acc_no_adapt
    }


NUM_EPOCHS_ADAPT = 30
acc_adapt_dict = dict()
for i in range(len(dataloaders)):
    for j in range(len(dataloaders)):
        if i == j:
            if i not in acc_adapt_dict:
                acc_adapt_dict[i] = dict()
                acc_adapt_dict[i][j] = 0
            continue
        else:
            source_ft_path = meta_dict[i]["ft_extractor"]
            source_classifier_path = meta_dict[i]["classifier"]
            
            source_ft = ResNet34().to(DEVICE)
            source_ft.load_state_dict(torch.load(source_ft_path))

            source_classifier = nn.Linear(source_ft.latent_dims, 10).to(DEVICE)
            source_classifier.load_state_dict(torch.load(source_classifier_path))

            classifier_optimizer = optim.SGD(source_classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

            for epoch in range(1, NUM_EPOCHS_ADAPT + 1):
                train(feature_extractor=source_ft,
                    classifier=source_classifier,
                    device=device,
                    train_loader=dataloaders[j],
                    epoch=epoch,
                    criterion=nn.CrossEntropyLoss(),
                    ft_extractor_optimizer=None,
                    classifier_optimizer=classifier_optimizer)

            cifar10_acc = test(source_ft, source_classifier, device, test_loader)

            print(f"From {i} to {j}, accuracy: {cifar10_acc}")

            if i not in acc_adapt_dict:
                acc_adapt_dict[i] = dict()
            acc_adapt_dict[i][j] = cifar10_acc


baseline_dict = dict()
for i in range(len(dataloaders)):
    baseline_dict[i] = meta_dict[i]["baseline"]


adapt_file_path = f"{save_dir}/adapt_result.json"
with open(adapt_file_path, 'w') as json_file:
    json.dump(acc_adapt_dict, json_file, indent=4)

no_adapt_file_path = f"{save_dir}/baseline_result.json"
with open(no_adapt_file_path, 'w') as json_file:
    json.dump(baseline_dict, json_file, indent=4)