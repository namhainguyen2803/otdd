import os
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parent_path = "CIFAR100/split_size"
# parent_path = "saved_cpu_2/time_comparison/CIFAR100/split_size"

sotdd_time_dict = dict()
otdd_time_dict = dict()

for file_name in os.listdir(parent_path):

    parts = file_name.split("_")
    split_size = int(parts[0][2:])
    num_split = int(parts[1][2:])
    num_projections = int(parts[2][2:])
    print(file_name, split_size, num_split, num_projections)

    sotdd_time_dict[split_size] = dict()

    with open(f"{parent_path}/{file_name}/time_running.txt", "r") as file:
        for line in file:
            pattern = r"sOTDD \((\d+) projections\): ([\d.]+)"
            match = re.search(pattern, line)

            if match:
                sotdd_time_dict[split_size][int(match.group(1))] = float(match.group(2))
            else:
                parts = float(line.split(": ")[-1])
                otdd_time_dict[split_size] = parts

# print(sotdd_time_dict)

print(otdd_time_dict)


def make_xy_coordinate(dict_data):
    lst_data = list()
    for ss, pt in dict_data.items():
        if ss >= 4000:
            lst_data.append([ss, pt])
    lst_data.sort(key= lambda x: x[0])

    list_x = list()
    list_y = list()
    for x, y in lst_data:
        list_x.append(x)
        list_y.append(y)

    return list_x, list_y

list_ss, list_pt = make_xy_coordinate(otdd_time_dict)
list_ss1000, list_pt1000 = make_xy_coordinate(sotdd_time_dict[1000])
list_ss3000, list_pt3000 = make_xy_coordinate(sotdd_time_dict[3000])
list_ss5000, list_pt5000 = make_xy_coordinate(sotdd_time_dict[5000])
list_ss8000, list_pt8000 = make_xy_coordinate(sotdd_time_dict[8000])
list_ss10000, list_pt10000 = make_xy_coordinate(sotdd_time_dict[10000])


plt.figure(figsize=(10, 8))
plt.plot(list_ss, list_pt, color='b', label='OTDD Running Time', marker='o', linewidth=2)
plt.plot(list_ss1000, list_pt1000, color='y', label='sOTDD (1.000 projections)', marker='o', linewidth=2)
plt.plot(list_ss3000, list_pt3000, color='m', label='sOTDD (3.000 projections)', marker='o', linewidth=2)
plt.plot(list_ss5000, list_pt5000, color='c', label='sOTDD (5.000 projections)', marker='o', linewidth=2)
plt.plot(list_ss8000, list_pt8000, color='r', label='sOTDD (8.000 projections)', marker='o', linewidth=2)
plt.plot(list_ss10000, list_pt10000, color='g', label='sOTDD (10.000 projections)', marker='o', linewidth=2)

plt.title('Processing Time Comparison', fontsize=20)
plt.xlabel('Dataset Size', fontsize=20)
plt.ylabel('Time Processing', fontsize=20)
plt.title(f'Time Comparison When Varying Size of Dataset', fontsize=16)
plt.grid(True)
plt.legend()
plt.savefig('split_size_comparison.pdf')
plt.savefig('split_size_comparison.png')

