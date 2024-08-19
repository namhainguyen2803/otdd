import torch
import torch.optim as optim
import torch.nn as nn
from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.method import NewDatasetDistance
from otdd.pytorch.distance import DatasetDistance

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from scipy import stats
import json

LIST_DATASETS = ["mnist", "fmnist", "emnist", "kmnist", "usps"]

# Load data
MAXSIZE = 60000
loaders_mnist  = load_torchvision_data('MNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)[0]
loaders_kmnist  = load_torchvision_data('KMNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)[0]
loaders_emnist  = load_torchvision_data('EMNIST', valid_size=0, resize = 28, download=True, maxsize=MAXSIZE)[0]
loaders_fmnist  = load_torchvision_data('FashionMNIST', valid_size=0, resize = 28, download=False, maxsize=MAXSIZE)[0]
loaders_usps  = load_torchvision_data('USPS',  valid_size=0, resize = 28, download=False, maxsize=MAXSIZE, datadir="data/USPS")[0]


def compute_distance_dataset(name_src, name_tgt, maxsamples=MAXSIZE, num_projection=5000):
    if name_src == "mnist":
        loaders_src = loaders_mnist
    elif name_src == "kmnist":
        loaders_src = loaders_kmnist
    elif name_src == "emnist":
        loaders_src = loaders_emnist
    elif name_src == "fmnist":
        loaders_src = loaders_fmnist
    elif name_src == "usps":
        loaders_src = loaders_usps
    else:
        raise("Unknown src dataset")

    if name_tgt == "mnist":
        loaders_tgt = loaders_mnist
    elif name_tgt == "kmnist":
        loaders_tgt = loaders_kmnist
    elif name_tgt == "emnist":
        loaders_tgt = loaders_emnist
    elif name_tgt == "fmnist":
        loaders_tgt = loaders_fmnist
    elif name_tgt == "usps":
        loaders_tgt = loaders_usps
    else:
        raise("Unknown tgt dataset")

    dist = NewDatasetDistance(loaders_src['train'], loaders_tgt['train'], p=2, device='cpu')
    d = dist.distance(maxsamples=maxsamples, num_projection=num_projection)

    # dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'],
    #                         inner_ot_method = 'exact',
    #                         debiased_loss = True,
    #                         p = 2, entreg = 1e-1,
    #                         device='cpu')
    # d = dist.distance(maxsamples = maxsamples)

    print(f'DIST({name_src}, {name_tgt})={d:8.2f}')
    return d


def compute_pairwise_distance():
    all_dist_dict = dict()
    for i in range(len(LIST_DATASETS)):
        all_dist_dict[LIST_DATASETS[i]] = dict()
        for j in range(len(LIST_DATASETS)):
            all_dist_dict[LIST_DATASETS[i]][LIST_DATASETS[j]] = 0

    for i in range(len(LIST_DATASETS)):
        for j in range(i + 1, len(LIST_DATASETS)):
            dist = compute_distance_dataset(LIST_DATASETS[i], LIST_DATASETS[j]).item()
            all_dist_dict[LIST_DATASETS[i]][LIST_DATASETS[j]] = dist
            all_dist_dict[LIST_DATASETS[j]][LIST_DATASETS[i]] = dist
    return all_dist_dict

dist = compute_pairwise_distance()

dist_file_path = 'saved/dist.json'
with open(dist_file_path, 'w') as json_file:
    json.dump(dist, json_file, indent=4)

print(f"DIST: {dist}")


class FeatureExtractor(nn.Module):
    def __init__(self, input_size=28):
        super(FeatureExtractor, self).__init__()
        assert input_size in [28, 32], "LeNet supports only 28x28 or 32x32 input sizes."
        
        if input_size == 32:
            self.layers = nn.Sequential(
                nn.Conv2d(1, 6, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.feat_dim = 16 * 5 * 5
        elif input_size == 28:
            self.layers = nn.Sequential(
                nn.Conv2d(1, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.feat_dim = 16 * 4 * 4

    def forward(self, x):
        return self.layers(x)

class Classifier(nn.Module):
    def __init__(self, feat_dim, num_classes=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

def frozen_module(module):
    for name, param in module.named_parameters():
        param.requires_grad = False

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def train(feature_extractor, classifier, device, train_loader, epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=1e-3, weight_decay=1e-6)
    feature_extractor.train()
    classifier.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        feature = feature_extractor(data).to(device)
        feature = feature.view(-1, num_flat_features(feature))
        output = classifier(feature).to(device)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % 1000 == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
        #           f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def adaptation(feature_extractor, classifier, device, train_loader, epoch):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)

    feature_extractor.eval()
    classifier.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        feature = feature_extractor(data).to(device)
        feature = feature.view(-1, num_flat_features(feature))
        output = classifier(feature).to(device)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % 1000 == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
        #           f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(feature_extractor, classifier, device, test_loader):
    feature_extractor.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            feature = feature_extractor(data).to(device)
            feature = feature.view(-1, num_flat_features(feature))
            output = classifier(feature).to(device)

            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    # print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
    #       f' ({100. * correct / len(test_loader.dataset):.0f}%)')
    
    return correct / len(test_loader.dataset)



# Training the model

MAXSIZE = None
loaders_mnist  = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=MAXSIZE)[0]
loaders_kmnist  = load_torchvision_data('KMNIST', valid_size=0, resize = 28, maxsize=MAXSIZE)[0]
loaders_emnist  = load_torchvision_data('EMNIST', valid_size=0, resize = 28, maxsize=MAXSIZE)[0]
loaders_fmnist  = load_torchvision_data('FashionMNIST', valid_size=0, resize = 28, maxsize=MAXSIZE)[0]
loaders_usps  = load_torchvision_data('USPS',  valid_size=0, resize = 28, maxsize=MAXSIZE, datadir="data/USPS")[0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 20
mnist_ft_extractor = FeatureExtractor(input_size=28).to(DEVICE)
mnist_classifier = Classifier(feat_dim=mnist_ft_extractor.feat_dim, num_classes=10).to(DEVICE)
kmnist_ft_extractor = FeatureExtractor(input_size=28).to(DEVICE)
kmnist_classifier = Classifier(feat_dim=kmnist_ft_extractor.feat_dim, num_classes=10).to(DEVICE)
fmnist_ft_extractor = FeatureExtractor(input_size=28).to(DEVICE)
fmnist_classifier = Classifier(feat_dim=fmnist_ft_extractor.feat_dim, num_classes=10).to(DEVICE)
emnist_ft_extractor = FeatureExtractor(input_size=28).to(DEVICE)
emnist_classifier = Classifier(feat_dim=emnist_ft_extractor.feat_dim, num_classes=26).to(DEVICE)
usps_ft_extractor = FeatureExtractor(input_size=28).to(DEVICE)
usps_classifier = Classifier(feat_dim=usps_ft_extractor.feat_dim, num_classes=10).to(DEVICE)

for epoch in range(1, num_epochs + 1):
    train(mnist_ft_extractor, mnist_classifier, DEVICE, loaders_mnist['train'], epoch)
for epoch in range(1, num_epochs + 1):
    train(kmnist_ft_extractor, kmnist_classifier, DEVICE, loaders_kmnist['train'], epoch)
for epoch in range(1, num_epochs + 1):
    train(fmnist_ft_extractor, fmnist_classifier, DEVICE, loaders_fmnist['train'], epoch)
for epoch in range(1, num_epochs + 1):
    train(emnist_ft_extractor, emnist_classifier, DEVICE, loaders_emnist['train'], epoch)
for epoch in range(1, num_epochs + 1):
    train(usps_ft_extractor, usps_classifier, DEVICE, loaders_usps['train'], epoch)

acc_mnist = test(mnist_ft_extractor, mnist_classifier, DEVICE, loaders_mnist['test'])
acc_kmnist = test(kmnist_ft_extractor, kmnist_classifier, DEVICE, loaders_kmnist['test'])
acc_fmnist = test(fmnist_ft_extractor, fmnist_classifier, DEVICE, loaders_fmnist['test'])
acc_emnist = test(emnist_ft_extractor, emnist_classifier, DEVICE, loaders_emnist['test'])
acc_usps = test(usps_ft_extractor, usps_classifier, DEVICE, loaders_usps['test'])

err_mnist = 1 - acc_mnist
err_kmnist = 1 - acc_kmnist
err_fmnist = 1 - acc_fmnist
err_emnist = 1 - acc_emnist
err_usps = 1 - acc_usps

err_from_scratch = {
    "mnist": err_mnist,
    "kmnist": err_kmnist,
    "emnist": err_emnist,
    "fmnist": err_fmnist,
    "usps": err_usps
}

model_path = 'saved/mnist_ft_extractor.pth'
torch.save(mnist_ft_extractor.state_dict(), model_path)
model_path = 'saved/fmnist_ft_extractor.pth'
torch.save(fmnist_ft_extractor.state_dict(), model_path)
model_path = 'saved/emnist_ft_extractor.pth'
torch.save(emnist_ft_extractor.state_dict(), model_path)
model_path = 'saved/kmnist_ft_extractor.pth'
torch.save(kmnist_ft_extractor.state_dict(), model_path)
model_path = 'saved/usps_ft_extractor.pth'
torch.save(usps_ft_extractor.state_dict(), model_path)

frozen_module(mnist_ft_extractor)
frozen_module(kmnist_ft_extractor)
frozen_module(emnist_ft_extractor)
frozen_module(fmnist_ft_extractor)
frozen_module(usps_ft_extractor)

list_frozen_ft_extractors = {
    "mnist": mnist_ft_extractor,
    "kmnist": kmnist_ft_extractor,
    "emnist": emnist_ft_extractor,
    "fmnist": fmnist_ft_extractor,
    "usps": usps_ft_extractor
}

def compare_adaptation(tgt_dataset, 
                        train_dataloader, test_dataloader,
                        feat_dim, num_classes, 
                        list_frozen_ft_extractors=list_frozen_ft_extractors, num_epochs=10, device=DEVICE):

    list_dataset = ["mnist", "fmnist", "emnist", "kmnist", "usps"]

    list_acc_adaptation = dict()
    for dt_name, ft_extractor in list_frozen_ft_extractors.items():
        if dt_name != tgt_dataset:
            classifier = Classifier(feat_dim=feat_dim, num_classes=num_classes).to(device)
            for epoch in range(1, num_epochs + 1):
                adaptation(ft_extractor, classifier, device, train_dataloader, epoch)
            acc_adaptation = test(ft_extractor, classifier, device, test_dataloader)
            list_acc_adaptation[dt_name] = acc_adaptation

    return list_acc_adaptation



def compute_rho_p_value(target_data):

    if target_data == "mnist":
        train_loader = loaders_mnist['train']
        test_loader = loaders_mnist['test']
        num_classes = 10
    elif target_data == "kmnist":
        train_loader = loaders_kmnist['train']
        test_loader = loaders_kmnist['test']
        num_classes = 10
    elif target_data == "emnist":
        train_loader = loaders_emnist['train']
        test_loader = loaders_emnist['test']
        num_classes = 26
    elif target_data == "fmnist":
        train_loader = loaders_fmnist['train']
        test_loader = loaders_fmnist['test']
        num_classes = 10
    elif target_data == "usps":
        train_loader = loaders_usps['train']
        test_loader = loaders_usps['test']
        num_classes = 10
    else:
        raise("Unknown src dataset")
    
    num_epochs = 10
    feat_dim = list_frozen_ft_extractors[target_data].feat_dim
    err = err_from_scratch[target_data]


    acc_adapt = compare_adaptation(
        tgt_dataset=target_data,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        feat_dim=feat_dim, num_classes=num_classes,
        list_frozen_ft_extractors=list_frozen_ft_extractors, 
        num_epochs=num_epochs, device=DEVICE
    )

    rel_error = dict()
    x = list()
    y = list()
    for dt_name in LIST_DATASETS:
        if dt_name != target_data:
            rel_error[dt_name] = (1 - acc_adapt[dt_name] - err) / err
            x.append(dist[target_data][dt_name])
            y.append(rel_error[dt_name])

    rho, p_value = stats.pearsonr(x, y)

    with open('output.txt', 'a') as f:
        f.write(f"--- TARGET DATA: {target_data} ---\n")
        f.write(f"Accuracy in adaptation: {acc_adapt}, Accuracy in training from scratch: {1 - err}\n")
        f.write(f"Relative error in adaptation: {rel_error}\n")
        f.write(f"Distance between sources and target: {dist[target_data]}\n")
        f.write(f"Pearson correlation coefficient: {rho}\n")
        f.write(f"P-value: {p_value}\n")
        f.write("\n")

    return rho, p_value, x, y

avg_rho = 0
list_x = list()
list_y = list()
rho, p_value, x, y = compute_rho_p_value("mnist")
avg_rho += rho
list_x.extend(x)
list_y.extend(y)
rho, p_value, x, y = compute_rho_p_value("kmnist")
avg_rho += rho
list_x.extend(x)
list_y.extend(y)
rho, p_value, x, y = compute_rho_p_value("emnist")
avg_rho += rho
list_x.extend(x)
list_y.extend(y)
rho, p_value, x, y = compute_rho_p_value("fmnist")
avg_rho += rho
list_x.extend(x)
list_y.extend(y)
rho, p_value, x, y = compute_rho_p_value("usps")
avg_rho += rho
list_x.extend(x)
list_y.extend(y)


ovr_rho, ovr_p_value = stats.pearsonr(list_x, list_y)
print(f"Overall Pearson correlation coefficient: {ovr_rho}")
print(f"Overall P-value: {ovr_p_value}")


list_X = np.array(list_x).reshape(-1, 1)
list_y = np.array(list_y)
model = LinearRegression().fit(list_X, list_y)
list_y_pred = model.predict(list_X)

plt.figure(figsize=(10, 8))
# sns.regplot(x=x, y=y, ci=95, scatter_kws={'s': 100}, line_kws={'color': 'blue'})
plt.scatter(list_x, list_y, s=100, color='blue', label='Data points')
plt.plot(list_x, list_y_pred, color='red', linewidth=2, label='Fitted line')

plt.xlabel('OT Dataset Distance')
plt.ylabel('Relative Drop in Test Error (%)')
plt.title(f'Distance vs Adaptation: *NIST Datasets {ovr_rho:.4f}, {ovr_p_value:.4f}')

plt.legend()
plt.savefig('distance_vs_adaptation.png')

avg_rho /= 5

print(f"Avg rho: {avg_rho}")
