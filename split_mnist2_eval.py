import re 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

runtime_path = "saved_runtime_mnist/time_comparison/MNIST/time_running.txt"

otdd_gaussian = list()
otdd_exact = list()
sotdd = dict()
with open(runtime_path, "r") as file:
    for line in file:
        if "sOTDD" in line:
            pattern = r"sOTDD \((\d+) projections\): ([\d.]+)"
            match = re.search(pattern, line)
            if match:
                proj_id = int(match.group(1))
                if proj_id not in sotdd:
                    sotdd[proj_id] = list()
                sotdd[proj_id].append(float(match.group(2)))
        elif "OTDD" in line:
            parts = float(line.split(": ")[-1])
            if "exact" in line:
                otdd_exact.append(parts)
            elif "gaussian" in line:
                otdd_gaussian.append(parts)

print(len(otdd_exact), len(otdd_gaussian))
for k, v in sotdd.items():
    print(k, len(v))

max_dataset_size = 30000
list_dataset_size = [2000 * (i + 1) for i in range(int(max_dataset_size // 2000))]

print(list_dataset_size, len(list_dataset_size))


sns.set(style="whitegrid")
colors = sns.color_palette("tab10")
MARKERSIZE = 6
LINEWIDTH = 2
FONT_SIZE = 20

plt.figure(figsize=(8, 8))

plt.plot(list_dataset_size, otdd_exact, color=colors[0], label='OTDD (exact)', marker='o', linestyle='-', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size, otdd_gaussian, color=colors[1], label='OTDD (Gaussian approx)', marker='s', linestyle='--', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size, sotdd[1000], color=colors[2], label='sOTDD (1,000 projections)', marker='D', linestyle='-.', linewidth=LINEWIDTH, markersize=MARKERSIZE)
# plt.plot(list_dataset_size, sotdd[5000], color=colors[3], label='sOTDD (5,000 projections)', marker='*', linestyle=':', linewidth=LINEWIDTH, markersize=MARKERSIZE)
plt.plot(list_dataset_size, sotdd[10000], color=colors[4], label='sOTDD (10,000 projections)', marker='*', linestyle=':', linewidth=LINEWIDTH, markersize=MARKERSIZE)

plt.xlabel("Dataset Size", fontsize=FONT_SIZE - 2)
plt.ylabel("Processing Time", fontsize=FONT_SIZE - 2)
plt.title("Time Comparison by Dataset Size", fontsize=FONT_SIZE, fontweight='bold')
plt.legend(loc="upper left", frameon=True)

plt.grid(True)
plt.legend()
plt.savefig('split_size_comparison.pdf', dpi=1000)
plt.savefig('split_size_comparison.png', dpi=1000)

