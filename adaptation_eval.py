import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import json

method = "sotdd"
finetune_weights_path = "saved/nist/finetune_weights"
baseline_weight_path = "saved/nist/pretrained_weights"
dist_path = f"saved/nist/{method}_dist.json"

acc_adapt = dict()
for target_name in os.listdir(finetune_weights_path):

    if target_name not in acc_adapt:
        acc_adapt[target_name] = dict()

    target_dir = f"{finetune_weights_path}/{target_name}"

    for source_name in os.listdir(target_dir):
        source_target_accuracy_file = f"{target_dir}/{source_name}/accuracy.txt"
        with open(source_target_accuracy_file, "r") as file:
            for line in file:
                parts = line.split(": ")
                num_epoch = int(parts[1].split(",")[0])
                acc_loss = parts[2].strip()[1:-1].split(", ")
                acc = float(acc_loss[0])
                loss = float(acc_loss[1])
                if num_epoch == 9:
                    acc_adapt[target_name][source_name] = acc

acc_baseline = dict()
for dt_name in os.listdir(baseline_weight_path):
    acc_path = f"{baseline_weight_path}/{dt_name}/accuracy.txt"

    with open(acc_path, "r") as file:
        for line in file:
            parts = line.split(": ")
            num_epoch = int(parts[1].split(",")[0])
            acc_loss = parts[2].strip()[1:-1].split(", ")
            acc = float(acc_loss[0])
            loss = float(acc_loss[1])
            if num_epoch == 9:
                acc_baseline[dt_name] = acc


with open(dist_path, 'r') as file:
    dict_dist = json.load(file)


perf_list = list()
dist_list = list()
for target_name in acc_baseline.keys():
    for source_name in acc_adapt[target_name].keys():
        perf = acc_baseline[target_name] - acc_adapt[target_name][source_name]
        perf_list.append(perf)
        dist_list.append(dict_dist[source_name][target_name])


list_X = np.array(dist_list).reshape(-1, 1)
list_y = np.array(perf_list)
model = LinearRegression().fit(list_X, list_y)
list_y_pred = model.predict(list_X)

plt.figure(figsize=(10, 8))

plt.scatter(dist_list, perf_list, s=100, color='blue', label='Data points')
plt.plot(dist_list, list_y_pred, color='red', linewidth=2, label='Fitted line')

rho, p_value = stats.pearsonr(dist_list, perf_list)


FONT_SIZE = 25
plt.title(f'{method} $\\rho={rho:.3f}, p={p_value:.3f}$', fontsize=FONT_SIZE)
plt.xlabel(f'{method} Distance', fontsize=FONT_SIZE)
plt.ylabel('Performance', fontsize=FONT_SIZE)

plt.legend()
plt.savefig(f'saved/nist/{method}.png')