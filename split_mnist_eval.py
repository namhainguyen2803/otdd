import re 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

dataset = "mnist"

if dataset == "mnist":
    parent_path = "saved_runtime_mnist_vietdt11_parts/time_comparison/MNIST"
else:
    parent_path = "saved_runtime_cifar10_new/time_comparison/CIFAR10"


otdd_gaussian = list()
otdd_exact = list()
wte = list()
sotdd = dict()
hswfs = list()

for file_name in os.listdir(parent_path):
    if ".pdf" in file_name or ".png" in file_name or "npy" in file_name:
        continue
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
            elif "OTDD" in line and (("exact" in line) or ("gaussian" in line)):
                parts = float(line.split(": ")[-1])
                if "exact" in line:
                    otdd_exact.append([dataset_size, parts])
                elif "gaussian" in line:
                    otdd_gaussian.append([dataset_size, parts])
            elif "WTE" in line:
                parts = float(line.split(": ")[-1])
                wte.append([dataset_size, parts])
            elif "HSWFS_OTDD" in line:
                parts = float(line.split(": ")[-1])
                hswfs.append([dataset_size, parts])


def make_xy_coordinate(lst_data):
    lst_data.sort(key= lambda x: x[0])
    list_x = list()
    list_y = list()
    exclude = []
    for x, y in lst_data:
        # if x in exclude:
        #     continue
        # if (x // 2000) % 2 == 0:
        #     continue
        list_x.append(x)
        list_y.append(y)
    return list_x, list_y

list_dataset_size_otdd_exact, list_otdd_exact = make_xy_coordinate(otdd_exact)
list_dataset_size_otdd_gaussian, list_otdd_gaussian = make_xy_coordinate(otdd_gaussian)
list_dataset_size_wte, list_wte = make_xy_coordinate(wte)
list_dataset_size_hswfs, list_hswfs = make_xy_coordinate(hswfs)
list_dataset_size_sotdd_100, list_sotdd_100 = make_xy_coordinate(sotdd[100])
list_dataset_size_sotdd_500, list_sotdd_500 = make_xy_coordinate(sotdd[500])
list_dataset_size_sotdd_1000, list_sotdd_1000 = make_xy_coordinate(sotdd[1000])
list_dataset_size_sotdd_5000, list_sotdd_5000 = make_xy_coordinate(sotdd[5000])
list_dataset_size_sotdd_10000, list_sotdd_10000 = make_xy_coordinate(sotdd[10000])

# print(list_dataset_size, len(list_dataset_size))


sns.set(style="whitegrid")
colors = sns.color_palette("tab10")
MARKERSIZE = 6
LINEWIDTH = 2
FONT_SIZE = 18

plt.figure(figsize=(8, 8))
plt.plot(list_dataset_size_otdd_exact, list_otdd_exact, color=colors[0], label='OTDD (Exact)', marker='o', linestyle='-', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size_wte, list_wte, color=colors[5], label='WTE', marker='D', linestyle='--', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size_otdd_gaussian, list_otdd_gaussian, color=colors[1], label='OTDD (Gaussian Approx)', marker='s', linestyle='-', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size_sotdd_1000, list_sotdd_1000, color=colors[3], label='sOTDD (1,000 projections)', marker='*', linestyle='-.', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size_sotdd_5000, list_sotdd_5000, color=colors[6], label='sOTDD (5,000 projections)', marker='*', linestyle='-.', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size_sotdd_10000, list_sotdd_10000, color=colors[7], label='sOTDD (10,000 projections)', marker='*', linestyle='-.', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size_hswfs, list_hswfs, color=colors[4], label='HSWFS OTDD (500 projections)', marker='*', linestyle='--', linewidth=LINEWIDTH, markersize=MARKERSIZE)

plt.xlabel("Dataset Size", fontsize=FONT_SIZE - 2)
plt.ylabel("Processing Time", fontsize=FONT_SIZE - 2)
plt.title(f"Time Comparison by Dataset Size", fontsize=FONT_SIZE, fontweight='bold')
plt.legend(loc="upper left", frameon=True)

plt.grid(True)
plt.legend()
plt.savefig(f'{parent_path}/split_size_comparison_{dataset}.pdf', dpi=1000)
plt.savefig(f'{parent_path}/split_size_comparison_{dataset}.png', dpi=1000)

