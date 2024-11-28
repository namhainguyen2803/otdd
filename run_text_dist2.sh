#!/bin/bash

# List of dataset names
datasets=("AG_NEWS" "DBpedia" "YelpReviewPolarity" "YelpReviewFull" "YahooAnswers" "AmazonReviewPolarity" "AmazonReviewFull")

# Loop through all dataset pairs where source != target and avoid redundant calculations
for i in "${!datasets[@]}"; do
    for j in $(seq $((i + 1)) $((${#datasets[@]} - 1))); do
        source="${datasets[i]}"
        target="${datasets[j]}"
        echo "Running for Source: $source, Target: $target"
        python3 text_dist2.py --saved_path saved_text_dist2 --source "$source" --target "$target" --method sotdd --num_examples 10000
    done
done
