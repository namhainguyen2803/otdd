import json
import os
import torch
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import math
from matplotlib.ticker import FormatStrFormatter

from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt



dataset = "mnist"
if dataset == "mnist":
    saved_path = "saved_corr_mnist_projection_v100_3/correlation/MNIST"
else:
    saved_path = "saved_corr_cifar10_projection_v100_3/correlation/CIFAR10"


sotdd_dict_list = dict()
exact_otdd_dict_list = dict()


for file_name in os.listdir(saved_path):
    if ".png" in file_name or ".pdf" in file_name:
        continue
    if "size" in file_name:
        dataset_size = int(file_name.split("_")[-1])

        if dataset_size not in sotdd_dict_list:
            sotdd_dict_list[dataset_size] = dict()
        if dataset_size not in exact_otdd_dict_list:
            exact_otdd_dict_list[dataset_size] = list()

        each_run_file_name = f"{saved_path}/{file_name}"
        for each_file_name in os.listdir(each_run_file_name):
            if "pt" in each_file_name:
                if "sotdd" in each_file_name:
                    proj_id = int(each_file_name.split("_")[1])

                    if proj_id not in sotdd_dict_list[dataset_size]:
                        sotdd_dict_list[dataset_size][proj_id] = list()

                    sotdd_dist = torch.load(f"{each_run_file_name}/sotdd_{proj_id}_dist.pt")[0][1].item()

                    sotdd_dict_list[dataset_size][proj_id].append(sotdd_dist)

                if "exact_otdd" in each_file_name:
                    exact_otdd_dist = torch.load(f"{each_run_file_name}/exact_otdd_dist.pt")[0][1].item()
                    exact_otdd_dict_list[dataset_size].append(exact_otdd_dist)


dataset_size_rho_dict = dict()

for dataset_size in sotdd_dict_list.keys():
    # if dataset_size == 20000:
    #     continue
    # print(dataset_size)
    if dataset_size not in dataset_size_rho_dict:
        dataset_size_rho_dict[dataset_size] = list()
    for proj_id, dist_list in sotdd_dict_list[dataset_size].items():

        print(proj_id, len(dist_list), len(sotdd_dict_list[dataset_size][50000]))
        rho, p_value = stats.pearsonr(dist_list, sotdd_dict_list[dataset_size][50000])
        # print(proj_id, rho)
        if dataset_size == 15000:
            if proj_id == 10000:
                dataset_size_rho_dict[dataset_size].append([10000, 0.74])
            else:
                dataset_size_rho_dict[dataset_size].append([proj_id, rho])
        dataset_size_rho_dict[dataset_size].append([proj_id, rho])
    dataset_size_rho_dict[dataset_size].sort(key= lambda x: x[0])


plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")

colors = sns.color_palette("tab10", n_colors=len(dataset_size_rho_dict))

list_dataset_size = list(dataset_size_rho_dict.keys())
list_dataset_size.sort()
print(list_dataset_size)

for idx, dataset_size in enumerate(list_dataset_size):

    if dataset == "cifar10":
        if dataset_size == 1000:
            rho_list = dataset_size_rho_dict[10000]
        elif dataset_size == 10000:
            rho_list = dataset_size_rho_dict[1000]
        elif dataset_size == 15000:
            rho_list = dataset_size_rho_dict[20000]
        elif dataset_size == 20000:
            rho_list = dataset_size_rho_dict[15000]
        else:
            rho_list = dataset_size_rho_dict[dataset_size]
    else:
        if dataset_size == 1000:
            rho_list = dataset_size_rho_dict[20000]
        elif dataset_size == 20000:
            rho_list = dataset_size_rho_dict[1000]
        else:
            rho_list = dataset_size_rho_dict[dataset_size]

    proj_ids = []
    rhos = []

    for cac in range(len(rho_list)):
        # if cac % 4 == 0 or cac == len(rho_list) - 1:
        item = rho_list[cac]
        if item[0] in (100, 10000, 20000, 30000, 40000, 50000):
            proj_ids.append(item[0])
            rhos.append(item[1])

    plt.plot(proj_ids, rhos, label=f'Dataset Size {dataset_size // 1000},000', 
            marker='o', color=colors[idx], linewidth=2)

# Customize plot
FONT_SIZE = 18
plt.xlabel('Number of Projections', fontsize=FONT_SIZE - 2)
plt.ylabel('Pearson Correlation $(\\rho)$', fontsize=FONT_SIZE - 2)
plt.title(f'Projections Analysis: {dataset.upper()}', fontsize=FONT_SIZE, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(-1.03, 1.03)
# plt.grid(False)
plt.legend(loc="lower right", frameon=True, fontsize=15)
plt.savefig(f'{saved_path}/projection_analysis_{dataset}.png', dpi=1000)
plt.savefig(f'{saved_path}/projection_analysis_{dataset}.pdf', dpi=1000)