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
from matplotlib.ticker import FormatStrFormatter


def scientific_number(x):
    if x == 0:
        return 0, 0
    b = int(math.floor(math.log10(abs(x))))
    a = x / (10 ** b)
    return a, b

dataset = "cifar10"
if dataset == "mnist":
    saved_path = "saved_corr_mnist_v100_4/correlation/MNIST"
    saved_path = "saved_corr_mnist_projection_v100_19_01_2025_2/correlation/MNIST"
# saved_path = "saved_runtime_cifar10_vietdt11_parts/time_comparison/CIFAR10"
else:
    # saved_path = "saved_corr_cifar10_v100_2/correlation/CIFAR10"
    saved_path = "saved_corr_cifar10_v100_4/correlation/CIFAR10"

sotdd_dict_list = dict()
ga_otdd_list = list()
exact_otdd_list = list()
wte_list = list()
hswfs_dict_list = dict()

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
                    # if dataset == "cifar10":
                    #     if sotdd_dist > 0.050:
                    #         sotdd_dist = 0.043
                    # else:
                    #     if sotdd_dist > 0.028:
                    #         sotdd_dist = 0.0238
                    sotdd_dict_list[proj_id].append(sotdd_dist)
                elif "wte" in each_file_name:
                    wte_dist = torch.load(f"{each_run_file_name}/wte.pt")[0][1].item()
                    wte_list.append(wte_dist)
                    
                elif "hswfs" in each_file_name:
                    proj_id = int(each_file_name.split("_")[1])
                    if proj_id not in hswfs_dict_list:
                        hswfs_dict_list[proj_id] = list()
                    hswfs_dist = torch.load(f"{each_run_file_name}/hswfs_{proj_id}_dist.pt")[0][1].item() * 10**3
                    hswfs_dict_list[proj_id].append(hswfs_dist)


title_dict = {
    "ga": "OTDD (Gaussian approx)",
    "exact": "OTDD (Exact)",
    "wte": "WTE",
    "hswfs_1000": "CHSW (1,000 projections) $\\times 10^{-3}$",
    "hswfs_5000": "CHSW (5,000 projections) $\\times 10^{-3}$",
    "hswfs_10000": "s-CHSW (10,000 projections) $\\times 10^{-3}$",
    "sotdd_1000": "s-OTDD (1,000 projections)",
    "sotdd_5000": "s-OTDD (5,000 projections)",
    "sotdd_10000": "s-OTDD (10,000 projections)"
}

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


    sns.regplot(
        x=list_dist_1,
        y=list_dist_2,
        scatter=True, 
        ci=95, 
        color="c", 
        scatter_kws={"s": 10, "color": "tab:blue"},  # Set dot color to blue
        label=label
    )

    FONT_SIZE = 25
    plt.title(f"Distance Correlation: {dataset.upper()}", fontsize=FONT_SIZE, fontweight='bold')
    plt.xlabel(title_dict[name_1], fontsize=FONT_SIZE - 2)
    plt.ylabel(title_dict[name_2], fontsize=FONT_SIZE - 2)
    # plt.ylim(min(list_dist_2) - 0.0005, max(list_dist_2) + 0.002)
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.grid(False)
    plt.legend(loc="upper left", frameon=True, fontsize=15)
    plt.tight_layout()
    plt.savefig(f'{saved_path}/correlation_dist_{dataset}_{name_1}_{name_2}.png', dpi=1000)
    plt.savefig(f'{saved_path}/correlation_dist_{dataset}_{name_1}_{name_2}.pdf', dpi=1000)


def retrieve_dist_list(method_name):
    if method_name == "exact":
        return exact_otdd_list, method_name
    elif method_name == "ga":
        return ga_otdd_list, method_name
    elif method_name == "wte":
        return wte_list, method_name
    elif method_name == "sotdd_100":
        return sotdd_dict_list[100], method_name
    elif method_name == "sotdd_500":
        return sotdd_dict_list[500], method_name
    elif method_name == "sotdd_1000":
        return sotdd_dict_list[1000], method_name
    elif method_name == "sotdd_5000":
        return sotdd_dict_list[5000], method_name
    elif method_name == "sotdd_10000":
        return sotdd_dict_list[10000], method_name
    elif method_name == "hswfs_100":
        return hswfs_dict_list[100], method_name
    elif method_name == "hswfs_500":
        return hswfs_dict_list[500], method_name
    elif method_name == "hswfs_1000":
        return hswfs_dict_list[1000], method_name
    elif method_name == "hswfs_5000":
        return hswfs_dict_list[5000], method_name
    elif method_name == "hswfs_10000":
        return hswfs_dict_list[10000], method_name

def retrieve_pair(method1, method2):
    method1_list, name1 = retrieve_dist_list(method1)
    method2_list, name2 = retrieve_dist_list(method2)
    min_method_size = min(len(method1_list), len(method2_list))
    print(f"len method 1: {len(method1_list)}, len method 2: {len(method2_list)}")
    print(f"Method 1: {method1}, method 2: {method2}, size method 1: {len(method1_list[:min_method_size])}, size method 2: {len(method2_list[:min_method_size])}")
    return method1_list[:min_method_size], name1, method2_list[:min_method_size], name2

list_methods = ["exact", "ga", "wte", "hswfs_100", "hswfs_500", "hswfs_1000", "hswfs_5000", "hswfs_10000", "sotdd_100", "sotdd_500", "sotdd_1000", "sotdd_5000", "sotdd_10000"]

abc = retrieve_pair("ga", "exact")
calculate_correlation(list_dist_1=abc[0], name_1=abc[1], list_dist_2=abc[2], name_2=abc[3])