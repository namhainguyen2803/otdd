import json

baseline_result_path = "saved/text_cls_spp/text_baseline.txt"
adapt_result_path = "saved/text_cls_spp/text_adapt.txt"
text_dist_path = "saved/text_cls_new/dist/text_dist.json"


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

print("cac 1")
print(adapt_acc["AmazonReviewPolarity"])


# read baseline result
baseline_acc = {}
with open(baseline_result_path, 'r') as file:
    for line in file:
        parts = line.strip().split(': ')
        source_dataset = parts[1].split(', ')[0]
        accuracy = float(parts[2])
        baseline_acc[source_dataset] = accuracy

print("cac 2")
print(baseline_acc)


perf_list = list()
dist_list = list()
DATASET_NAME = list(baseline_acc.keys())
print(DATASET_NAME)
for i in range(len(DATASET_NAME)):
    for j in range(i+1, len(DATASET_NAME)):
        source = DATASET_NAME[i]
        target = DATASET_NAME[j]
        print(source, target)
        perf = adapt_acc[source][target] - baseline_acc[source]
        dist = text_dist[source][target]

        perf_list.append(perf)
        dist_list.append(dist)


list_X = np.array(dist_list).reshape(-1, 1)
list_y = np.array(perf_list)
model = LinearRegression().fit(list_X, list_y)
list_y_pred = model.predict(list_X)
print(list_X)
print(list_y)
plt.figure(figsize=(10, 8))
# sns.regplot(x=x, y=y, ci=95, scatter_kws={'s': 100}, line_kws={'color': 'blue'})
plt.scatter(list_dist, perf_list, s=100, color='blue', label='Data points')
plt.plot(list_dist, list_y_pred, color='red', linewidth=2, label='Fitted line')

rho, p_value = stats.pearsonr(list_dist, perf_list)

display_method = "OTDD"
FONT_SIZE = 25
plt.title(f'{display_method} $\\rho={rho:.3f}, p={p_value:.3f}$', fontsize=FONT_SIZE)  # Increase title size
plt.xlabel(f'{display_method}', fontsize=FONT_SIZE)  # Increase x-axis label size
plt.ylabel('Accuracy', fontsize=FONT_SIZE)  # Increase y-axis label size


plt.legend()
plt.savefig(f'text_{display_method}.pdf')

