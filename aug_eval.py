import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats


# Define the base directory
base_dir = 'saved/augmentation2'

# List all folders inside the base directory
folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

list_dist = list()
list_acc = list()

method = "New method"
method = "OTDD"

display_method = "sOTDD" if method == "New method" else "OTDD"


def compute_rss(observed, predicted):
    if len(observed) != len(predicted):
        raise ValueError("Both lists must have the same length.")
    rss = sum((obs - pred) ** 2 for obs, pred in zip(observed, predicted))
    return rss


# Print the folder names
for date_run in folders:
    folder_name = base_dir + f"/{date_run}/result.txt"

    if os.path.isfile(folder_name):
        with open(folder_name, 'r') as file:
            lines = file.readlines()
            
            # Initialize variables to hold the extracted values
            distance = None
            accuracy_pretrain = None
            
            # Extract the necessary information
            for line in lines:
                if line.startswith(f"{method}, Distance:"):
                    print(line.split(":")[1].split(",")[0].strip())
                    distance = float(line.split(":")[1].split(",")[0].strip())
                elif "Accuracy when having pretraned" in line:
                    accuracy_pretrain = float(line.split()[-1])

            list_dist.append(distance)
            list_acc.append(accuracy_pretrain)

            print(distance, accuracy_pretrain)


list_X = np.array(list_dist).reshape(-1, 1)
list_y = np.array(list_acc)
model = LinearRegression().fit(list_X, list_y)
list_y_pred = model.predict(list_X)
print(list_X)
print(list_y)
plt.figure(figsize=(10, 8))
# sns.regplot(x=x, y=y, ci=95, scatter_kws={'s': 100}, line_kws={'color': 'blue'})
plt.scatter(list_dist, list_acc, s=100, color='blue', label='Data points')
plt.plot(list_dist, list_y_pred, color='red', linewidth=2, label='Fitted line')

rho, p_value = stats.pearsonr(list_dist, list_acc)
rss = compute_rss(list_y, list_y_pred)
print(rss)
rss = rss * 1000
# plt.title(f'{method} corr={rho:.4f}, p_value={p_value:.4f}, rss={rss:.4f}')

FONT_SIZE = 25
plt.title(f'{display_method} $\\rho={rho:.3f}, p={p_value:.3f}, \\mathrm{{RSS}}={rss:.3f} \\times 10^{{-3}}$', fontsize=FONT_SIZE)  # Increase title size
plt.xlabel(f'{display_method}', fontsize=FONT_SIZE)  # Increase x-axis label size
plt.ylabel('Accuracy', fontsize=FONT_SIZE)  # Increase y-axis label size


plt.legend()
plt.savefig(f'aug_{display_method}.pdf')

