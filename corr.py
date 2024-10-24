import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import os

LIST_DATASETS = ["MNIST", "FashionMNIST", "EMNIST", "KMNIST", "USPS"]

# Load ACC_ADAPT
file_path = 'saved/nist/sotdd_dist_no_conv_15.json'
with open(file_path, 'r') as file:
    otdd_dist = json.load(file)


with open("saved/nist/sotdd15_rankings.txt", "w") as file:
    for source_name in otdd_dist.keys():
        lst = list()
        for target_name in otdd_dist[source_name].keys():
            if target_name == source_name:
                continue
            lst.append([target_name, otdd_dist[source_name][target_name]])
        lst.sort(key= lambda x: x[1])
        file.write(f"{source_name} \n")
        for i in range(len(lst)):
            file.write(f"   +) {i+1}. {lst[i][0]} \n")
    
    