import json
import numpy as np 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats

display_method = "sotdd"
parent_dir = "saved_text_dataset"
baseline_result_path = f"{parent_dir}/accuracy.txt"
adapt_result_path = f"{parent_dir}/adapt_result.txt"
text_dist_path = f"{parent_dir}/sotdd_text_dist_num_moments_5_num_examples_2000.json"


# read text distance
with open(text_dist_path, "r") as file:
    text_dist = json.load(file)

# read adapt result
adapt_acc = {}
with open(adapt_result_path, 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        source_dataset = parts[0].split(': ')[1]
        target_dataset = parts[1].split(': ')[1]
        accuracy = float(parts[2].split(': ')[1])

        if source_dataset not in adapt_acc:
            adapt_acc[source_dataset] = {}
        adapt_acc[source_dataset][target_dataset] = accuracy


# read baseline result
baseline_acc = {}
with open(baseline_result_path, 'r') as file:
    for line in file:
        parts = line.strip().split(': ')
        # print(parts)
        source_dataset = parts[1].split(', ')[0]
        accuracy = float(parts[3])
        baseline_acc[source_dataset] = accuracy

perf_list = list()
dist_list = list()
DATASET_NAME = list(baseline_acc.keys())
print(DATASET_NAME)


# Initialize global min and max values for dist_list
global_min = float('inf')
global_max = float('-inf')

# Loop to find the global min and max for all dist_lists
for i in range(len(DATASET_NAME)):
    dist_list = []
    perf_list = []

    if DATASET_NAME[i] == "AmazonReviewPolarity":
        continue
    
    for j in range(len(DATASET_NAME)):

        target = DATASET_NAME[i]
        source = DATASET_NAME[j]

        if source == target:
            continue
        if source == "AmazonReviewPolarity" or target == "AmazonReviewPolarity":
            continue

        perf = ((adapt_acc[source][target]) - (baseline_acc[target]))
        dist = text_dist[source][target]

        perf_list.append(perf)
        dist_list.append(dist)
    
    # Update global min and max for dist_list
    global_min = min(global_min, np.min(dist_list))
    global_max = max(global_max, np.max(dist_list))

# Set up a figure for all plots
plt.figure(figsize=(8, 8))

# Loop over datasets and accumulate the data
for i in range(len(DATASET_NAME)):
    perf_list = []
    dist_list = []

    if DATASET_NAME[i] == "AmazonReviewPolarity":
        continue

    for j in range(len(DATASET_NAME)):
        target = DATASET_NAME[i]
        source = DATASET_NAME[j]

        if source == target:
            continue
        if source == "AmazonReviewPolarity" or target == "AmazonReviewPolarity":
            continue

        perf = ((adapt_acc[source][target]) - (baseline_acc[target]))
        dist = text_dist[source][target]

        perf_list.append(perf)
        dist_list.append(dist)

    # Convert lists to numpy arrays
    list_X = np.array(dist_list).reshape(-1, 1)
    list_y = np.array(perf_list)

    # Fit linear regression model
    model = LinearRegression().fit(list_X, list_y)
    
    # Create new x-values spanning the global range for the fitted line
    global_X_range = np.linspace(global_min, global_max, 100).reshape(-1, 1)
    list_y_pred_global = model.predict(global_X_range)

    # Scatter plot of current dataset points
    plt.scatter(dist_list, perf_list, s=100, label=f'{DATASET_NAME[i]} data points')

    # Calculate Pearson correlation
    rho, p_value = stats.pearsonr(dist_list, perf_list)
    print(f"{DATASET_NAME[i]}: rho={rho}, p-value={p_value}")

    if rho < 0:
        # Plot fitted line over the global range
        plt.plot(global_X_range, list_y_pred_global, linewidth=2, label=f'{DATASET_NAME[i]} fitted line')


# Add labels and title after loop
FONT_SIZE = 25
plt.title(f'{display_method} - Multiple Datasets', fontsize=FONT_SIZE)
plt.xlabel(f'{display_method}', fontsize=FONT_SIZE)
plt.ylabel('Accuracy', fontsize=FONT_SIZE)

# Show legend
plt.legend()

# Save the final plot
plt.savefig(f'{parent_dir}/combined_text_{display_method}.png')
plt.show()