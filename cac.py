import os
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parent_path = "CIFAR100/split_size"

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


list_ss = list()
list_pt = list()
for ss, pt in otdd_time_dict.items():
    print(ss, pt)
    list_ss.append(ss)
    list_pt.append(pt)

plt.figure(figsize=(10, 8))
plt.scatter(list_ss, list_pt, s=100, color='blue', label='Data points')

plt.xlabel('Size')
plt.ylabel('Time')
plt.title(f'cac')

plt.legend()
plt.savefig('otdd_split_size.png')