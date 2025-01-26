import csv

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

parent_path = "saved_augmentation_2"

method = "sotdd"
maxsize = 50000
displayed_method = "s-OTDD (10,000 projections)"

# method = "otdd_exact"
# maxsize = 5000
# displayed_method = "OTDD (Exact)"

file_path = f"{parent_path}/acc_dist_method_{method}_maxsize_{maxsize}_10.txt"
# file_path = "saved_augmentation_2/acc_dist_method_sotdd_maxsize_50000.txt"


list_acc = list()
list_dist = list()
with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        print(parts)
        seed_id = int(parts[0].split(': ')[1])
        accuracy = float(parts[1].split(': ')[1]) * 100
        distance = float(parts[2].split(': ')[1])
        print(accuracy, distance)
        if accuracy < 87.6:
            list_acc.append(accuracy)
            list_dist.append(distance)


print(list_acc)
print(list_dist)


def calculate_correlation(list_dist_1, list_dist_2):
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
        scatter_kws={"s": 10, "color": "tab:blue"},
        label=label
    )
    FONT_SIZE = 18
    plt.title(f"Distance vs Adaptation: ImageNet$\\rightarrow$CIFAR10", fontsize=FONT_SIZE, fontweight='bold')
    plt.xlabel(f"{displayed_method}", fontsize=FONT_SIZE - 2)
    plt.ylabel("Accuracy (%)", fontsize=FONT_SIZE - 2)
    # plt.ylim(86, 87.8)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.grid(False)
    plt.legend(loc="upper right", frameon=True, fontsize=15)
    plt.savefig(f'{parent_path}/aug_{method}_{maxsize}.png', dpi=1000)
    plt.savefig(f'{parent_path}/aug_{method}_{maxsize}.pdf', dpi=1000)

calculate_correlation(list_dist, list_acc)
