import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import json
from matplotlib.ticker import FormatStrFormatter

method = "sotdd"
if method == "sotdd":
    display_method = "s-OTDD (10,000 projections)"
else:
    display_method = "OTDD (Exact)"

saved_dir = "saved_nist"
if method == "sotdd":
    # dist_path = f"saved/nist/{method}_dist_no_conv_8_normalizing_moments_3.json"
    # dist_path = f"saved_nist/dist/sotdd_dist_use_conv_False_num_moments_10.json"
    dist_path = f"saved_nist/dist/sotdd_gaussian_dist.json"
else:
    dist_path = f"saved_nist/dist/{method}_dist_exact.json"

with open(dist_path, 'r') as file:
    dict_dist = json.load(file)

acc_adapt = dict()
acc_baseline = dict()

for each_run in os.listdir(saved_dir):
    if "pdf" in each_run:
        continue
    if "png" in each_run:
        continue
    if "nist" in each_run:

        finetune_weights_path = f"{saved_dir}/{each_run}/finetune_weights"
        baseline_weight_path = f"{saved_dir}/{each_run}/pretrained_weights"

        each_run_acc_adapt = dict()
        for target_name in os.listdir(finetune_weights_path):
            if target_name not in each_run_acc_adapt:
                each_run_acc_adapt[target_name] = dict()
            target_dir = f"{finetune_weights_path}/{target_name}"
            for source_name in os.listdir(target_dir):
                source_target_accuracy_file = f"{target_dir}/{source_name}/accuracy.txt"
                with open(source_target_accuracy_file, "r") as file:
                    for line in file:
                        parts = line.split(": ")
                        num_epoch = int(parts[1].split(",")[0])
                        acc_loss = parts[2].strip()[1:-1].split(", ")
                        acc = float(acc_loss[0])
                        loss = float(acc_loss[1])
                        if num_epoch == 9:
                            each_run_acc_adapt[target_name][source_name] = acc
        
        acc_adapt[each_run] = each_run_acc_adapt

        each_run_acc_baseline = dict()
        for dt_name in os.listdir(baseline_weight_path):
            acc_path = f"{baseline_weight_path}/{dt_name}/accuracy.txt"
            with open(acc_path, "r") as file:
                for line in file:
                    parts = line.split(": ")
                    num_epoch = int(parts[1].split(",")[0])
                    acc_loss = parts[2].strip()[1:-1].split(", ")
                    acc = float(acc_loss[0])
                    loss = float(acc_loss[1])
                    if num_epoch == 9:
                        each_run_acc_baseline[dt_name] = acc
        
        acc_baseline[each_run] = each_run_acc_baseline

     
perf_dict = dict()
for each_run in os.listdir(saved_dir):
    if "pdf" in each_run:
        continue
    if "png" in each_run:
        continue
    if "nist" in each_run:
        perf_list = list()
        for target_name in acc_baseline[each_run].keys():
            if target_name not in perf_dict:
                perf_dict[target_name] = dict()
            for source_name in acc_adapt[each_run][target_name].keys():
                if source_name not in perf_dict[target_name]:
                    perf_dict[target_name][source_name] = list()
                perf = (acc_baseline[each_run][target_name] - acc_adapt[each_run][target_name][source_name]) / acc_baseline[each_run][target_name]
                perf_dict[target_name][source_name].append(perf * 100)


perf_data = []
for target_name, sources in perf_dict.items():
    for source_name, perf_list in sources.items():
        if perf_list:  # Ensure the list is not empty
            avg_perf = np.mean(perf_list)  # Average performance drop
            error = np.std(perf_list) / 2  # Standard deviation as error (converted to percentage)
            # if error > 0.7:
            #     error = 0.7
            dist = dict_dist[target_name][source_name]
            if dist is not None or dist != 0:
                perf_data.append({
                    "Source -> Target": f"{source_name[0]}->{target_name[0]}",
                    "OT Dataset Distance": dist,
                    "Relative Drop in Test Error (%)": avg_perf,
                    "Error": error
                })


# Create DataFrame
df = pd.DataFrame(perf_data)

# Calculate Pearson correlation
pearson_corr, p_value = stats.pearsonr(df["OT Dataset Distance"], df["Relative Drop in Test Error (%)"])

label=f"$\\rho$: {pearson_corr:.2f}\np-value: {p_value:.2f}"

# Plotting
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")

# Scatter plot with regression line and confidence interval (only over data range)
sns.regplot(
    x="OT Dataset Distance", 
    y="Relative Drop in Test Error (%)", 
    data=df, 
    scatter=True, 
    ci=95, 
    color="c", 
    scatter_kws={"s": 10, "color": "tab:blue"},  # Set dot color to blue
    label=label
)

# Add error bars
plt.errorbar(
    df["OT Dataset Distance"], 
    df["Relative Drop in Test Error (%)"], 
    yerr=df["Error"], 
    fmt='o', 
    color='gray', 
    capsize=1.5, 
    capthick=0, 
    elinewidth=1,
    markersize=0
)

# Fit linear regression manually to extend line beyond the data range
X = df["OT Dataset Distance"].values.reshape(-1, 1)
y = df["Relative Drop in Test Error (%)"].values
reg = LinearRegression().fit(X, y)

x_range = np.linspace(df["OT Dataset Distance"].min(), df["OT Dataset Distance"].max(), 500)
y_pred = reg.predict(x_range.reshape(-1, 1))

# Add data labels to each point
for i, row in df.iterrows():
    plt.text(row["OT Dataset Distance"], row["Relative Drop in Test Error (%)"], 
             row["Source -> Target"], ha='right', fontsize=10)

# Add Pearson correlation and p-value to the plot as a legend
plt.legend(loc="upper left", frameon=True, fontsize=15)

# Customize title and labels
FONT_SIZE = 18
plt.title(f"Distance vs Adaptation: *NIST Datasets", fontsize=FONT_SIZE, weight='bold')
plt.xlabel(f"{display_method} Distance", fontsize=FONT_SIZE - 2)
plt.ylabel("Performance Gap (%)", fontsize=FONT_SIZE - 2)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# Display plot
plt.grid(False)
plt.savefig(f'{saved_dir}/nist_{display_method}_gauss.png')
plt.savefig(f'{saved_dir}/nist_{display_method}_gauss.pdf')
