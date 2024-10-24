import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import json

otdd_dist_path = f"saved/nist/sotdd_dist_no_conv_8_normalizing_moments_3.json"
sotdd_dist_path = f"saved/nist/otdd_dist.json"




with open(otdd_dist_path, 'r') as file:
    otdd_dist = json.load(file)

with open(sotdd_dist_path, 'r') as file:
    sotdd_dist = json.load(file)


otdd_list = list()
sotdd_list = list()
for target_name in otdd_dist.keys():
    for source_name in otdd_dist[target_name].keys():
        otdd_list.append(otdd_dist[target_name][source_name])
        sotdd_list.append(sotdd_dist[target_name][source_name])


list_X = np.array(otdd_list).reshape(-1, 1)
list_y = np.array(sotdd_list)
model = LinearRegression().fit(list_X, list_y)
list_y_pred = model.predict(list_X)

# x_min, x_max = min(dist_list) - 300, max(dist_list) + 300
# x_extended = np.linspace(x_min, x_max, 100).reshape(-1, 1)
# y_extended_pred = model.predict(x_extended)

plt.figure(figsize=(10, 8))

plt.scatter(otdd_list, sotdd_list, s=20, color='blue')
plt.plot(otdd_list, list_y_pred, color='red', linewidth=4)
# plt.plot(x_extended, y_extended_pred, color='red', linewidth=3)

rho, p_value = stats.pearsonr(otdd_list, sotdd_list)


FONT_SIZE = 25
plt.title(f'$\\rho={rho:.3f}, p={p_value:.3f}$', fontsize=FONT_SIZE)
plt.xlabel(f'OTDD', fontsize=FONT_SIZE)
plt.ylabel('s-OTDD', fontsize=FONT_SIZE)

# plt.legend()
plt.savefig(f'correlation.png')