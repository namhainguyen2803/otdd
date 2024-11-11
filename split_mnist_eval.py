import re 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

parent_path = "saved_runtime_mnist_new/time_comparison/MNIST"


otdd_gaussian = list()
otdd_exact = list()
wte = list()
sotdd = dict()

for file_name in os.listdir(parent_path):
    dataset_size = int(file_name.split("_")[-1])
    runtime_path = f"{parent_path}/{file_name}/time_running.txt"
    with open(runtime_path, "r") as file:
        for line in file:
            pattern = r"sOTDD \((\d+) projections\): ([\d.]+)"
            match = re.search(pattern, line)
            if "sOTDD" in line:
                if match:
                    proj_id = int(match.group(1))
                    if proj_id not in sotdd:
                        sotdd[proj_id] = list()
                    sotdd[proj_id].append([dataset_size, float(match.group(2))])
            elif "OTDD" in line:
                parts = float(line.split(": ")[-1])
                if "exact" in line:
                    otdd_exact.append([dataset_size, parts])
                elif "gaussian" in line:
                    otdd_gaussian.append([dataset_size, parts])
            elif "WTE" in line:
                parts = float(line.split(": ")[-1])
                wte.append([dataset_size, parts])


def make_xy_coordinate(lst_data):
    lst_data.sort(key= lambda x: x[0])
    list_x = list()
    list_y = list()
    for x, y in lst_data:
        list_x.append(x)
        list_y.append(y)
    return list_x, list_y

list_dataset_size, list_otdd_exact = make_xy_coordinate(otdd_exact)
list_dataset_size, list_otdd_gaussian = make_xy_coordinate(otdd_gaussian)
list_dataset_size, list_wte = make_xy_coordinate(wte)
list_dataset_size, list_sotdd_100 = make_xy_coordinate(sotdd[100])
list_dataset_size, list_sotdd_1000 = make_xy_coordinate(sotdd[1000])
list_dataset_size, list_sotdd_10000 = make_xy_coordinate(sotdd[10000])

max_dataset_size = 30000

print(list_dataset_size, len(list_dataset_size))


sns.set(style="whitegrid")
colors = sns.color_palette("tab10")
MARKERSIZE = 6
LINEWIDTH = 2
FONT_SIZE = 20

plt.figure(figsize=(8, 8))
plt.plot(list_dataset_size, list_otdd_exact, color=colors[0], label='OTDD (Exact)', marker='o', linestyle='-', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size[:-1], list_wte, color=colors[5], label='WTE', marker='D', linestyle='--', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size, list_otdd_gaussian, color=colors[1], label='OTDD (Gaussian Approx)', marker='s', linestyle='--', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size, list_sotdd_100, color=colors[2], label='sOTDD (100 projections)', marker='D', linestyle='-.', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size, list_sotdd_1000, color=colors[3], label='sOTDD (1,000 projections)', marker='*', linestyle=':', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size, list_sotdd_10000, color=colors[4], label='sOTDD (10,000 projections)', marker='*', linestyle=':', linewidth=LINEWIDTH, markersize=MARKERSIZE)

plt.xlabel("Dataset Size", fontsize=FONT_SIZE - 2)
plt.ylabel("Processing Time", fontsize=FONT_SIZE - 2)
plt.title("Time Comparison by Dataset Size", fontsize=FONT_SIZE, fontweight='bold')
plt.legend(loc="upper left", frameon=True)

plt.grid(True)
plt.legend()
plt.savefig('split_size_comparison2.pdf', dpi=1000)
plt.savefig('split_size_comparison2.png', dpi=1000)

