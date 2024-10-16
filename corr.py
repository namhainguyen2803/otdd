import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import os

LIST_DATASETS = ["MNIST", "FashionMNIST", "EMNIST", "KMNIST", "USPS"]

parent_dir = f"nist_corr2"
os.makedirs(parent_dir, exist_ok=True)

# Load ACC_ADAPT
file_path = 'saved/nist/finetune_weights/acc_adapt.json'
with open(file_path, 'r') as file:
    ACC_ADAPT = json.load(file)

# Load ACC_NO_ADAPT
file_path = 'saved/nist/finetune_weights/acc_baseline.json'
with open(file_path, 'r') as file:
    ACC_BASELINE = json.load(file)

# Load DIST
file_path = 'saved/nist/new_dist.json'
with open(file_path, 'r') as file:
    DIST = json.load(file)

for i in range(len(LIST_DATASETS)):
    target = LIST_DATASETS[i]

    list_performance_gains = []
    list_dists = []
    list_labels = []

    for j in range(len(LIST_DATASETS)):
        source = LIST_DATASETS[j]

        if source == target:
            continue
        else:
            perf_gain = (ACC_ADAPT[source][target] - ACC_BASELINE[target])
            list_performance_gains.append(perf_gain * 100)
            list_dists.append(DIST[source][target])
            list_labels.append(f'{source[0]}->{target[0]}')

    # Calculate the Pearson correlation
    ovr_rho, ovr_p_value = stats.pearsonr(list_dists, list_performance_gains)
    print(f"Overall Pearson correlation coefficient: {ovr_rho}")
    print(f"Overall P-value: {ovr_p_value}")

    # Fit the linear regression model
    list_X = np.array(list_dists).reshape(-1, 1)
    list_y = np.array(list_performance_gains)
    model = LinearRegression().fit(list_X, list_y)
    list_y_pred = model.predict(list_X)

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(list_dists, list_performance_gains, s=100, color='blue', label='Data points')
    plt.plot(list_dists, list_y_pred, color='blue', linewidth=2, label='Fitted line')

    # Annotate each point with its label
    for i in range(len(list_dists)):
        plt.text(list_dists[i], list_performance_gains[i], list_labels[i], fontsize=12, ha='right')

    plt.xlabel('OTDD Distance')
    plt.ylabel('Performance Gap (%)')
    plt.title(f'$r = {ovr_rho:.2f}$ $p = {ovr_p_value:.2f}$')

    plt.legend()
    plt.savefig(f'{parent_dir}/{target}_corr.png')
    plt.show()


