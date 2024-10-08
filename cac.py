import os
import json


saved_dist_path = "saved/text_cls_new/dist"


DATASET_NAMES = ["AG_NEWS", "DBpedia", "YelpReviewPolarity", "YelpReviewFull", "YahooAnswers", "AmazonReviewPolarity", "AmazonReviewFull"]
dataset_dist = dict()
for source_name in DATASET_NAMES:
    for target_name in DATASET_NAMES:
        if source_name not in dataset_dist:
            dataset_dist[source_name] = dict()
        dataset_dist[source_name][target_name] = 0

for filename in os.listdir(saved_dist_path):
    if filename == "text_dist.json":
        continue
    print(filename)
    json_path = saved_dist_path + "/"+ filename
    with open(json_path, "r") as file:
        data = json.load(file)

        target_dt = filename.split("_text")[0]

        for source_dt in data.keys():
            dataset_dist[source_dt][target_dt] = data[source_dt][target_dt]
            dataset_dist[target_dt][source_dt] = data[source_dt][target_dt]

print(dataset_dist)


dist_file_path = f'{saved_dist_path}/text_dist.json'
with open(dist_file_path, 'w') as json_file:
    json.dump(dataset_dist, json_file, indent=4)