import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

class FullyConnectedNetwork(nn.Module):
    def __init__(self, feat_dim, num_classes=10):
        super(FullyConnectedNetwork, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(feat_dim, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout()
        )
        self.classifier = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.module(x)
        return self.classifier(x)
    
    def change_head(self, new_num_classes):
        self.classifier = nn.Linear(84, new_num_classes)

def frozen_module(module):
    for name, param in module.named_parameters():
        param.requires_grad = False

def check_frozen_module(module):
    for name, param in module.named_parameters():
        if param.requires_grad == True:
            return False

    return True

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

# Training loop
def train(feature_extractor, classifier, device, train_loader, epoch=None, criterion=nn.CrossEntropyLoss(), ft_extractor_optimizer=None, classifier_optimizer=None):

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if classifier_optimizer is None:
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
        classifier.train()
    else:
        classifier.train()
    
    assert check_frozen_module(classifier) == False, "Classifier must be trained"

    if ft_extractor_optimizer is None:
        frozen_module(feature_extractor)
        feature_extractor.eval()
        assert check_frozen_module(feature_extractor) == True, "Feature Extractor is set to be frozen."
    else:
        feature_extractor.train()
        assert check_frozen_module(feature_extractor) == False, "Feature Extractor is set to be trained."

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if ft_extractor_optimizer is not None:
            ft_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        feature = feature_extractor(data).to(device)
        feature = feature.view(-1, num_flat_features(feature))
        output = classifier(feature).to(device)

        loss = criterion(output, target)
        loss.backward()

        if ft_extractor_optimizer is not None:
            ft_extractor_optimizer.step()
        classifier_optimizer.step()

    return loss.item()
        # if batch_idx % 1000 == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
        #           f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test_func(feature_extractor, classifier, device, test_loader):
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
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)')
    
    return correct / len(test_loader.dataset), test_loss

