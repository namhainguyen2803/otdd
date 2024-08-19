import torch
from torchtext.datasets import DBpedia

# Download and load the train and test datasets
train_iter, test_iter = DBpedia(split=('train', 'test'))

# Example: Accessing the first item in the training dataset
for label, text in train_iter:
    print(f"Label: {label}")
    print(f"Text: {text}")
    break  # Remove this to iterate through the whole dataset
