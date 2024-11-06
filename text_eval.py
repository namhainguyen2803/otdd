import json
import numpy as np 
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats


method = "OTDD"
if method == "sOTDD":
    display_method = "s-OTDD"
else:
    display_method = method.upper()


parent_dir = "saved/text_cls_new"
baseline_result_path = f"{parent_dir}/baseline_new/accuracy.txt"
adapt_result_path = f"{parent_dir}/adapt_weights/adapt_result.txt"
text_dist_path = f"{parent_dir}/dist/{method}_text_dist.json"


# read text distance
with open(text_dist_path, "r") as file:
    text_dist = json.load(file)

# read adapt result
adapt_acc = {}
with open(adapt_result_path, 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        source_dataset = parts[0].split(': ')[1]
        target_dataset = parts[1].split(': ')[1]
        accuracy = float(parts[2].split(': ')[1])

        if source_dataset not in adapt_acc:
            adapt_acc[source_dataset] = {}
        adapt_acc[source_dataset][target_dataset] = accuracy


# read baseline result
baseline_acc = {}
with open(baseline_result_path, 'r') as file:
    for line in file:
        parts = line.strip().split(': ')
        # print(parts)
        source_dataset = parts[1].split(', ')[0]
        accuracy = float(parts[3])
        baseline_acc[source_dataset] = accuracy

print(baseline_acc)


mean = 0.0
std_dev = 0.001

perf_list = list()
dist_list = list()
DATASET_NAME = list(baseline_acc.keys())
print(DATASET_NAME)
for i in range(len(DATASET_NAME)):
    for j in range(len(DATASET_NAME)):
        source = DATASET_NAME[i]
        target = DATASET_NAME[j]

        if source == target:
            continue
        if source == "AmazonReviewPolarity" or target == "AmazonReviewPolarity":
            continue

        # gaussian_numbers = torch.normal(mean=mean, std=std_dev, size=(1,))
        perf = ((adapt_acc[source][target]) - (baseline_acc[target])) / baseline_acc[target]
        dist = text_dist[source][target]

        perf_list.append(perf)
        dist_list.append(dist)

list_X = np.array(dist_list).reshape(-1, 1)
list_y = np.array(perf_list)
model = LinearRegression().fit(list_X, list_y)
list_y_pred = model.predict(list_X)


if method == "OTDD":
    x_min, x_max = min(dist_list) - 50, max(dist_list) + 50
else:
    x_min, x_max = min(dist_list) - 0.01, max(dist_list) + 0.01

x_extended = np.linspace(x_min, x_max, 100).reshape(-1, 1)

# Predict y values for the extended range
y_extended_pred = model.predict(x_extended)

plt.figure(figsize=(10, 8))

plt.scatter(dist_list, perf_list, s=20, color='blue')
plt.plot(x_extended, y_extended_pred, color='red', linewidth=3)

def compute_rss(observed, predicted):
    if len(observed) != len(predicted):
        raise ValueError("Both lists must have the same length.")
    rss = sum((obs - pred) ** 2 for obs, pred in zip(observed, predicted))
    return rss

rss = compute_rss(list_y, list_y_pred) * 100
rho, p_value = stats.pearsonr(dist_list, perf_list)


FONT_SIZE = 25
plt.title(f'{display_method} $\\rho={rho:.3f}, p={p_value:.3f}, \\mathrm{{RSS}}={rss:.3f} \\times 10^{{-2}}$', fontsize=FONT_SIZE)
plt.xlabel(f'{display_method} Distance', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)

# plt.legend()
plt.savefig(f'text_cls_{display_method}.png')

