import os
import json
import re


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

print(sotdd_time_dict)

print(otdd_time_dict)