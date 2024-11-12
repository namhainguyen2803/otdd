import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
from matplotlib.ticker import FormatStrFormatter

# Define the base directory and output directory for saved plots
base_dir = 'saved/augmentation2'
saved_dir = 'saved/plots'  # Define a directory to save plots if not already defined

# List all folders inside the base directory
folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

list_dist = []
list_acc = []

method = "New method"
method = "OTDD"

display_method = "s-OTDD" if method == "New method" else "OTDD"

# Collect distances and accuracies from files
for date_run in folders:
    file_path = os.path.join(base_dir, date_run, 'result.txt')
    
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            # Initialize variables to hold the extracted values
            distance = None
            accuracy_pretrain = None
            
            # Extract the necessary information
            for line in lines:
                if line.startswith(f"{method}, Distance:"):
                    distance = float(line.split(":")[1].split(",")[0].strip())
                elif "Accuracy when having pretraned" in line:
                    accuracy_pretrain = float(line.split()[-1])

            # Only add to lists if values were extracted successfully
            if distance is not None and accuracy_pretrain is not None:
                list_dist.append(distance)
                list_acc.append(accuracy_pretrain)


# Calculate Pearson correlation
pearson_corr, p_value = stats.pearsonr(list_dist, list_acc)

# Prepare data for plotting
df = pd.DataFrame({'OT Dataset Distance': list_dist, 'Accuracy (%)': list_acc})

# Plotting
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")

# Scatter plot with regression line and confidence interval (only over data range)
# sns.regplot(
#     x="OT Dataset Distance", 
#     y="Accuracy (%)", 
#     data=df, 
#     scatter=True, 
#     ci=95, 
#     color="c", 
#     scatter_kws={"s": 5, "color": "tab:blue"}  # Set dot color to blue
# )

plt.scatter(df["OT Dataset Distance"], df["Accuracy (%)"], s=5, color="tab:blue")

# Fit linear regression manually to extend line beyond the data range
X = np.array(list_dist).reshape(-1, 1)
y = np.array(list_acc)
reg = LinearRegression().fit(X, y)

# Generate x values for the extended line
x_range = np.linspace(min(list_dist), max(list_dist), 500)
y_pred = reg.predict(x_range.reshape(-1, 1))

# Plot the extended regression line
plt.plot(x_range, y_pred, linewidth=1.5, color="tab:blue", label=f"$\\rho$: {pearson_corr:.2f}\np-value: {p_value:.2f}")

# Add Pearson correlation and p-value to the plot as a legend
plt.legend(loc="upper right", frameon=True)

# Customize title and labels
FONT_SIZE = 20
plt.title(f"Distance vs Adaptation: Augmentation", fontsize=FONT_SIZE, fontweight='bold')
if 
plt.xlabel(f"s-OTDD (1,000 projections)", fontsize=FONT_SIZE - 2)
# plt.xlabel(f"{display_method} Distance", fontsize=FONT_SIZE - 2)
plt.ylabel("Accuracy (%)", fontsize=FONT_SIZE - 2)

plt.grid(False)
os.makedirs(saved_dir, exist_ok=True)
plt.savefig(f'{saved_dir}/aug_{display_method}.png', dpi=1000)
plt.savefig(f'{saved_dir}/aug_{display_method}.pdf', dpi=1000)