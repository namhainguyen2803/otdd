import json
import numpy as np 
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import math
from matplotlib.ticker import FormatStrFormatter


def scientific_number(x):
    if x == 0:
        return 0, 0
    b = int(math.floor(math.log10(abs(x))))
    a = x / (10 ** b)
    return a, b

method = "sOTDD"
# method = "OTDD"
if method == "sOTDD":
    display_method = "s-OTDD"
else:
    display_method = method.upper()


dataset_nicknames = {
"AG_NEWS": "ag",
"DBpedia": "db",
"YelpReviewPolarity": "y1+",
"YelpReviewFull": "y15",
"YahooAnswers": "yh",
"AmazonReviewPolarity": "am+",
"AmazonReviewFull": "am5"
}


parent_dir = "saved/text_cls_new"
baseline_result_path = f"{parent_dir}/baseline_new/accuracy.txt"
adapt_result_path = f"{parent_dir}/adapt_weights/adapt_result.txt"
# text_dist_path = f"{parent_dir}/dist/{method}_text_dist.json"
if method == "OTDD":
    text_dist_path = "saved_text_dist/text_cls/dist/OTDD_20_text_dist.json"
    text_dist_path = "saved/text_cls_new2/dist/OTDD_text_dist.json"
    text_dist_path = "saved_text_dist/text_cls/otdd_dist/otdd_exact_text_dist.json"
else:
    text_dist_path = "saved_text_dist/text_cls/dist/sOTDD_text_dist.json"
    # text_dist_path = "saved/text_cls_new2/dist/sOTDD_text_dist3.json"
    # text_dist_path = "saved_text_dist/text_cls/otdd_dist/sotdd_text_dist_num_moments_5_num_examples_2000.json"

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

        if target_dataset not in adapt_acc:
            adapt_acc[target_dataset] = {}
        adapt_acc[target_dataset][source_dataset] = accuracy


# read baseline result
baseline_acc = {}
with open(baseline_result_path, 'r') as file:
    for line in file:
        parts = line.strip().split(': ')
        # print(parts)
        source_dataset = parts[1].split(', ')[0]
        accuracy = float(parts[3])
        baseline_acc[source_dataset] = accuracy

# print(baseline_acc)




DATASET_NAME = list(baseline_acc.keys())
perf_data = []
for i in range(len(DATASET_NAME)):
    for j in range(len(DATASET_NAME)):
        source_name = DATASET_NAME[i]
        target_name = DATASET_NAME[j]
        if source_name == target_name:
            continue
        # if source_name == "AmazonReviewPolarity" or target_name == "AmazonReviewPolarity":
        #     continue
        
        perf = ((baseline_acc[target_name]) - (adapt_acc[target_name][source_name]))
        # if perf > 0.4:
        #     continue
        # error = torch.abs(torch.normal(mean=0.0, std=0.05, size=(1,)))
        # print(error)
        dist = text_dist[target_name][source_name]

        if dist is not None or dist != 0:
            perf_data.append({
                "Source -> Target": f"{dataset_nicknames[source_name]}->{dataset_nicknames[target_name]}",
                "distance": dist,
                "performance": perf * 100,
                "Error": 0
            })


# Create DataFrame
df = pd.DataFrame(perf_data)

# Calculate Pearson correlation
rho, p_value = stats.pearsonr(df["distance"], df["performance"])
print(f"rho: {rho}, p_value: {p_value}")
# Plotting
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")


a, b = scientific_number(p_value)
label = f"$\\rho$: {rho:.2f}\np-value: {a:.2f}$\\times 10^{{{b}}}$"
sns.regplot(
    x="distance", 
    y="performance", 
    data=df, 
    scatter=True, 
    ci=95, 
    color="c", 
    scatter_kws={"s": 10, "color": "tab:blue"},
    label=label
)

# Add error bars
plt.errorbar(
    df["distance"], 
    df["performance"], 
    yerr=df["Error"], 
    fmt='o', 
    color='gray', 
    capsize=1.5, 
    capthick=0, 
    elinewidth=1,
    markersize=0
)

# Fit linear regression manually to extend line beyond the data range
X = df["distance"].values.reshape(-1, 1)
y = df["performance"].values
reg = LinearRegression().fit(X, y)


x_range = np.linspace(df["distance"].min(), df["distance"].max(), 500)
y_pred = reg.predict(x_range.reshape(-1, 1))


plt.legend(loc="upper right", frameon=True)

# Customize title and labels
FONT_SIZE = 18
plt.title(f"Distance vs Adaptation: Text Classification", fontsize=FONT_SIZE, fontweight='bold')

# for i, row in df.iterrows():
#     plt.text(perf_data[i]["distance"], perf_data[i]["performance"], 
#              perf_data[i]["Source -> Target"], ha='right', fontsize=10)

if method == "sOTDD":
    plt.xlabel(f's-OTDD (10,000 projections) Distance', fontsize=FONT_SIZE - 2)
else:
    plt.xlabel(f'OTDD (Exact) Distance', fontsize=FONT_SIZE - 2)
plt.ylabel('Performance Gap (%)', fontsize=FONT_SIZE - 2)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Display plot
plt.legend(fontsize=15)
plt.grid(False)
plt.savefig(f'text_cls_{display_method}.png', dpi=1000)
plt.savefig(f'text_cls_{display_method}.pdf', dpi=1000)

