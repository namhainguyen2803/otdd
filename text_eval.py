baseline_result_path = "saved/text_cls_spp/text_baseline.txt"
adapt_result_path = "saved/text_cls_spp/text_adapt.txt"

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
print(adapt_acc)


# read baseline result
baseline_acc = {}
with open(baseline_result_path, 'r') as file:
    for line in file:
        parts = line.strip().split(': ')
        source_dataset = parts[1].split(', ')[0]
        accuracy = float(parts[2])
        baseline_acc[source_dataset] = accuracy

print(baseline_acc)

DATASET_NAME = baseline_acc.keys()
for i in range(len(DATASET_NAME)):
    for j in range(i+1, len(DATASET_NAME)):
        source = DATASET_NAME[i]
        target = DATASET_NAME[j]
        perf = adapt_acc[source][target] - baseline_acc[source]
