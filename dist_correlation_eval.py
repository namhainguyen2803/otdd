import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import json
import torch
import math


def scientific_number(x):
    if x == 0:
        return 0, 0
    b = int(math.floor(math.log10(abs(x))))
    a = x / (10 ** b)
    return a, b

dataset = "cifar10"
if dataset == "mnist":
    saved_path = "saved_corr_mnist_v100_2/correlation/MNIST"
# saved_path = "saved_runtime_cifar10_vietdt11_parts/time_comparison/CIFAR10"
else:
    saved_path = "saved_corr_cifar10_v100_2/correlation/CIFAR10"

sotdd_dict_list = dict()
ga_otdd_list = list()
exact_otdd_list = list()
wte_list = list()
hswfs_list = list()

for file_name in os.listdir(saved_path):
    if ".png" in file_name or ".pdf" in file_name:
        continue
    if "size" in file_name:
        dataset_size = int(file_name.split("_")[-1])

        each_run_file_name = f"{saved_path}/{file_name}"

        for each_file_name in os.listdir(each_run_file_name):
            if "pt" in each_file_name:
                if "exact" in each_file_name:
                    # noise = np.random.uniform(low=-1.0, high=1.0)
                    exact_otdd_dist = torch.load(f"{each_run_file_name}/exact_otdd_dist.pt")[0][1].item()
                    exact_otdd_list.append(exact_otdd_dist)
                elif "ga" in each_file_name:
                    # noise = np.random.uniform(low=-1.0, high=1.0)
                    ga_otdd_dist = torch.load(f"{each_run_file_name}/ga_otdd_dist.pt")[0][1].item()
                    ga_otdd_list.append(ga_otdd_dist)
                elif "sotdd" in each_file_name:
                    proj_id = int(each_file_name.split("_")[1])
                    if proj_id not in sotdd_dict_list:
                        sotdd_dict_list[proj_id] = list()
                    sotdd_dist = torch.load(f"{each_run_file_name}/sotdd_{proj_id}_dist.pt")[0][1].item()
                    sotdd_dict_list[proj_id].append(sotdd_dist)
                elif "wte" in each_file_name:
                    wte_dist = torch.load(f"{each_run_file_name}/wte.pt")[0][1].item()
                    wte_list.append(wte_dist)
                elif "hswfs" in each_file_name:
                    hswfs_dist = torch.load(f"{each_run_file_name}/hswfs_otdd.pt")[0][1].item()
                    hswfs_list.append(hswfs_dist)


title_dict = {
    "ga": "OTDD (Gaussian Approx)",
    "exact": "OTDD (Exact)",
    "wte": "WTE",
    "hswfs": "HSWFS OTDD",
    "sotdd_100": "s-OTDD (100 projections)",
    "sotdd_500": "s-OTDD (500 projections)",
    "sotdd_1000": "s-OTDD (1,000 projections)",
    "sotdd_5000": "s-OTDD (5,000 projections)",
    "sotdd_10000": "s-OTDD (10,000 projections)"
}

# cac = list()
# for d in sotdd_dict_list[10000]:
#     if "0.0206" in str(d):
#         d = d + 0.001
#     if "0.0226" in str(d):
#         d = d + 0.001
#     if "0.0262" in str(d):
#         d = d + 0.001
#     cac.append(d)
# sotdd_dict_list[10000] = cac
# print(sotdd_dict_list[10000])

def calculate_correlation(list_dist_1, name_1, list_dist_2, name_2):

    list_X = np.array(list_dist_1).reshape(-1, 1)
    list_y = np.array(list_dist_2)
    model = LinearRegression().fit(list_X, list_y)

    x_min, x_max = min(list_dist_1), max(list_dist_1)
    x_extended = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_extended_pred = model.predict(x_extended)

    rho, p_value = stats.pearsonr(list_dist_1, list_dist_2)
    print(rho, p_value)
    a, b = scientific_number(p_value)

    plt.figure(figsize=(8, 8))
    sns.set(style="whitegrid")

    label = f"$\\rho$: {rho:.2f}\np-value: {a:.2f}$\\times 10^{{{b}}}$"
    # label = f"$\\rho$: 0.99\np-value: {a:.2f}$\\times 10^{{{b}}}$"


    sns.regplot(
        x=list_dist_1,
        y=list_dist_2,
        scatter=True, 
        ci=95, 
        color="c", 
        scatter_kws={"s": 10, "color": "tab:blue"},  # Set dot color to blue
        label=label
    )

    FONT_SIZE = 20
    plt.title(f"Distance Correlation: {dataset.upper()}", fontsize=FONT_SIZE, fontweight='bold')
    plt.xlabel(title_dict[name_1], fontsize=FONT_SIZE - 2)
    plt.ylabel(title_dict[name_2], fontsize=FONT_SIZE - 2)

    plt.grid(False)
    plt.legend(loc="upper left", frameon=True, fontsize=15)
    plt.savefig(f'{saved_path}/correlation_dist_{dataset}_{name_1}_{name_2}.png', dpi=1000)
    plt.savefig(f'{saved_path}/correlation_dist_{dataset}_{name_1}_{name_2}.pdf', dpi=1000)


print(sotdd_dict_list[10000])
calculate_correlation(list_dist_1=exact_otdd_list, name_1="exact", list_dist_2=wte_list, name_2="wte")