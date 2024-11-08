import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import json
import torch


saved_path = "saved_mnist/time_comparison/MNIST/split_size"

sotdd_list = list()
ga_otdd_list = list()
exact_otdd_list = list()
for file_name in os.listdir(saved_path):
    if "SS" in file_name and "NS" in file_name and "NP" in file_name:
        parts = file_name.split("_")
        split_size = int(parts[0][2:])
        num_split = int(parts[1][2:])
        num_projections = int(parts[2][2:])
        print(file_name, split_size, num_split, num_projections)
        each_run_file_name = f"{saved_path}/{file_name}"
        sotdd_dist = torch.load(f"{each_run_file_name}/sotdd_dist.pt")[0][1].item()
        ga_otdd_dist = torch.load(f"{each_run_file_name}/ga_otdd_dist.pt")[0][1].item()
        exact_otdd_dist = torch.load(f"{each_run_file_name}/exact_otdd_dist.pt")[0][1].item()
        
        sotdd_list.append(sotdd_dist)
        ga_otdd_list.append(ga_otdd_dist)
        exact_otdd_list.append(exact_otdd_dist)

method = "ga"
if method == "ga":
    otdd_list = ga_otdd_list
else:
    otdd_list = exact_otdd_list

list_X = np.array(sotdd_list).reshape(-1, 1)
list_y = np.array(otdd_list)
model = LinearRegression().fit(list_X, list_y)

x_min, x_max = min(sotdd_list) - 0.001, max(sotdd_list) + 0.001
x_extended = np.linspace(x_min, x_max, 100).reshape(-1, 1)
y_extended_pred = model.predict(x_extended)

rho, p_value = stats.pearsonr(sotdd_list, otdd_list)
print(p_value)


plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")

if method == "ga":
    label = f"$\\rho$: {rho:.2f}\np-value: {p_value * 10**17:.2f}$\\times 10^{{-17}}$"
else:
    label = f"$\\rho$: {rho:.2f}\np-value: {p_value * 10**18:.2f}$\\times 10^{{-18}}$"

sns.regplot(
    x=sotdd_list,
    y=otdd_list,
    scatter=True, 
    ci=95, 
    color="c", 
    scatter_kws={"s": 10, "color": "tab:blue"},  # Set dot color to blue
    label=label
)

FONT_SIZE = 20
plt.title("Distance Correlation", fontsize=FONT_SIZE, fontweight='bold')
plt.xlabel(f's-OTDD (10,000 projections)', fontsize=FONT_SIZE - 2)

if method == "ga":
    plt.ylabel('OTDD (Gaussian approximation)', fontsize=FONT_SIZE - 2)
else:
    plt.ylabel('OTDD (Exact)', fontsize=FONT_SIZE - 2)

plt.grid(False)
plt.legend(loc="upper left", frameon=True, fontsize=15)
plt.savefig(f'{saved_path}/correlation_dist_{method}.png', dpi=1000)
plt.savefig(f'{saved_path}/correlation_dist_{method}.pdf', dpi=1000)