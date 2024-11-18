import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import json

method = "sotdd"
if method == "sotdd":
    display_method = "s-OTDD"
else:
    display_method = method.upper()

finetune_weights_path = "saved/nist/finetune_weights"
baseline_weight_path = "saved/nist/pretrained_weights"
dist_path = f"saved/nist/{method}_dist_no_conv_8_normalizing_moments_3.json"
dist_path = f"saved_nist/dist/{method}_dist.json"
dist_path = f"saved_nist/dist/sotdd_dist_use_conv_False_num_moments_10.json"


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
        # if target_name == "USPS" or source_name == "USPS":
        perf = acc_baseline[target_name] - acc_adapt[target_name][source_name]
        # if perf < 0: 
        #     continue
        perf_list.append(perf * 100)
        dist_list.append(dict_dist[source_name][target_name])


list_X = np.array(dist_list).reshape(-1, 1)
list_y = np.array(perf_list)
model = LinearRegression().fit(list_X, list_y)
list_y_pred = model.predict(list_X)

if method == "sotdd":
    x_min, x_max = min(dist_list) - 0.1, max(dist_list) + 0.1
else:
    x_min, x_max = min(dist_list) - 50, max(dist_list) + 50
x_extended = np.linspace(x_min, x_max, 100).reshape(-1, 1)

# Predict y values for the extended range
y_extended_pred = model.predict(x_extended)

def compute_rss(observed, predicted):
    if len(observed) != len(predicted):
        raise ValueError("Both lists must have the same length.")
    rss = sum((obs - pred) ** 2 for obs, pred in zip(observed, predicted))
    return rss


rss = compute_rss(list_y, list_y_pred)
rho, p_value = stats.pearsonr(dist_list, perf_list)


plt.figure(figsize=(10, 8))

plt.scatter(dist_list, perf_list, s=15, color='blue')
# plt.plot(x_extended, y_extended_pred, color='red', linewidth=3, label=f'$ \\rho={rho:.3f} \\ p-value={p_value:.3f} \\mathrm{{RSS}}={rss:.3f} \\times 10^{{-3}}$')
# plt.plot(x_extended, y_extended_pred, color='red', linewidth=3,
#          label=f'$\\rho={rho:.3f} \n p-value={p_value:.3f} \n \\mathrm{{RSS}}={rss:.3f} \\times 10^{{-3}}$')
plt.plot(x_extended, y_extended_pred, color='red', linewidth=3,
         label=(f'$ \\rho={rho:.3f}$\n'
                f'p-value={p_value:.2f}'
                ))

FONT_SIZE = 20
plt.title('Transfer Learning: *NIST Datasets', fontsize=FONT_SIZE)
plt.xlabel(f'{display_method} Distance', fontsize=FONT_SIZE)
plt.ylabel('Performance Gap (%)', fontsize=FONT_SIZE)

plt.legend(fontsize=15)
plt.savefig(f'saved/nist/{display_method}_nist.png')
plt.savefig(f'saved/nist/{display_method}_nist.pdf')