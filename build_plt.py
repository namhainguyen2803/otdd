import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

# Example data
ot_distance = np.array([1000, 1200, 1300, 1400, 1500, 1600, 1700])
relative_drop = np.array([70, 60, 50, 45, 40, 30, 25])
error = np.array([5, 7, 6, 4, 8, 3, 6])

# Calculate correlation
corr, p_value = pearsonr(ot_distance, relative_drop)

# Set up the plot
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")

# Scatter plot with error bars
plt.errorbar(ot_distance, relative_drop, yerr=error, fmt='o', color='gray', ecolor='lightgray', capsize=3)

# Regression plot with Seaborn
sns.regplot(x=ot_distance, y=relative_drop, scatter=False, color='blue', line_kws={"label": f"$\\rho$ : {corr:.2f}\n$p$-value: {p_value:.2f}"})

# Add legend
plt.legend(loc="upper right", frameon=True, fancybox=True)

# Annotations for points
annotations = ["E→M", "K→M", "E→U", "M→U", "F→E", "K→F", "F→M"]
for i, label in enumerate(annotations):
    plt.annotate(label, (ot_distance[i], relative_drop[i]), textcoords="offset points", xytext=(0, 5), ha='center')

# Labels and title
plt.xlabel("OT Dataset Distance")
plt.ylabel("Relative Drop in Test Error (%)")
plt.title("Distance vs Adaptation: *NIST Datasets")

plt.savefig("build_plt.png")