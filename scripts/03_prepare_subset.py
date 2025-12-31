import json

INPUT = "data/raw/final_dataset_instruct.jsonl"
OUTPUT = "data/raw/train_subset_200.jsonl"

with open(INPUT, "r", encoding="utf-8") as f:
    lines = f.readlines()

subset = lines[:200]

with open(OUTPUT, "w", encoding="utf-8") as f:
    for line in subset:
        f.write(line)

print("Subset size:", len(subset))
print("Saved to:", OUTPUT)
