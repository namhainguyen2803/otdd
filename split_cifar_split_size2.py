import os
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# parent_path = "CIFAR100/split_size"
parent_path = "saved_mnist/time_comparison/MNIST/split_size"
# parent_path = "saved_cpu_2/time_comparison/CIFAR100/split_size"

sotdd_time_dict = dict()
exact_otdd_time_dict = dict()
gaussian_otdd_time_dict = dict()

for file_name in os.listdir(parent_path):

    parts = file_name.split("_")
    split_size = int(parts[0][2:])
    num_split = int(parts[1][2:])
    num_projections = int(parts[2][2:])
    print(file_name, split_size, num_split, num_projections)

    with open(f"{parent_path}/{file_name}/time_running.txt", "r") as file:
        for line in file:
            pattern = r"sOTDD \((\d+) projections\): ([\d.]+)"
            match = re.search(pattern, line)

            if match:
                if int(match.group(1)) not in sotdd_time_dict:
                    sotdd_time_dict[int(match.group(1))] = dict()

                sotdd_time_dict[int(match.group(1))][split_size] = float(match.group(2))
            else:
                if "exact" in line:
                    print(line)
                    parts = float(line.split(": ")[-1])
                    exact_otdd_time_dict[split_size] = parts
                elif "gaussian" in line:
                    if "iter 20" in line:
                        print(line)
                        parts = float(line.split(": ")[-1])
                        gaussian_otdd_time_dict[split_size] = parts



def make_xy_coordinate(dict_data):
    lst_data = list()
    for ss, pt in dict_data.items():
        if ss >= 2000:
            lst_data.append([ss, pt])
    lst_data.sort(key= lambda x: x[0])

    list_x = list()
    list_y = list()
    for x, y in lst_data:
        list_x.append(x)
        list_y.append(y)

    return list_x, list_y

list_ss, list_pt = make_xy_coordinate(exact_otdd_time_dict)
list_gss, list_gpt = make_xy_coordinate(gaussian_otdd_time_dict)
list_ss1000, list_pt1000 = make_xy_coordinate(sotdd_time_dict[1000])
list_ss3000, list_pt3000 = make_xy_coordinate(sotdd_time_dict[3000])
list_ss5000, list_pt5000 = make_xy_coordinate(sotdd_time_dict[5000])
list_ss8000, list_pt8000 = make_xy_coordinate(sotdd_time_dict[8000])
list_ss10000, list_pt10000 = make_xy_coordinate(sotdd_time_dict[10000])


Y_LIMITS = 100

# Set Seaborn style
sns.set(style="whitegrid")

# Define colors
colors = sns.color_palette("tab10")
MARKERSIZE = 6
LINEWIDTH = 2
FONT_SIZE = 20

# Create the main plot
fig, ax_main = plt.subplots(figsize=(8, 6))

# Main plot
ax_main.plot(list_ss, list_pt, color=colors[0], label='OTDD (exact)', marker='o', linestyle='-', linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax_main.plot(list_gss, list_gpt, color=colors[1], label='OTDD (Gaussian approx)', marker='s', linestyle='--', linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax_main.plot(list_ss1000, list_pt1000, color=colors[2], label='sOTDD (1,000 projections)', marker='D', linestyle='-.', linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax_main.plot(list_ss10000, list_pt10000, color=colors[3], label='sOTDD (10,000 projections)', marker='*', linestyle=':', linewidth=LINEWIDTH, markersize=MARKERSIZE)

# Set limits, labels, and title for the main plot
ax_main.set_ylim(0, Y_LIMITS)  # Limit the main plot's y-axis to 200
ax_main.set_xlabel("Dataset Size", fontsize=FONT_SIZE - 2)
ax_main.set_ylabel("Processing Time", fontsize=FONT_SIZE - 2)
ax_main.set_title("Time Comparison by Dataset Size", fontsize=FONT_SIZE, fontweight='bold')
ax_main.legend(loc="center right", frameon=True)

# Create an inset plot
ax_inset = inset_axes(ax_main, width="30%", height="20%", loc="upper right", borderpad=0.2)  # Adjust width, height, and location
ax_inset.set_xticks([5000, 10000, 14000])
ax_inset.set_xticklabels(['5K', '10K', '14K'], fontsize=10)  # Shortened format

# Inset plot: focusing on the y-axis range above 200
ax_inset.plot(list_ss, list_pt, color=colors[0], marker='o', linestyle='-', linewidth=LINEWIDTH, markersize=MARKERSIZE)
ax_inset.plot(list_gss, list_gpt, color=colors[1], marker='s', linestyle='--', linewidth=LINEWIDTH, markersize=MARKERSIZE)

# Set the zoomed-in y-axis limit for the inset plot
ax_inset.set_ylim(Y_LIMITS, max(list_pt) + 50)
# ax_inset.set_xlim(5000, 15000)
# ax_inset.set_title(, fontsize=FONT_SIZE - 6)  # Smaller title for inset

# Hide the inset plot's legend if redundant
ax_inset.legend().set_visible(False)

# Save and display
# ax_main.grid(False)
ax_inset.grid(False)
plt.savefig('split_size_comparison_with_inset.png', dpi=1000)
plt.savefig('split_size_comparison_with_inset.pdf', dpi=1000)
