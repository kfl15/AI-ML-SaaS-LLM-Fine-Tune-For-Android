import json
from collections import Counter

DATA_PATH = "data/raw/final_dataset_instruct.jsonl"

count = 0
lengths = []
schema_errors = 0
sample_records = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        count += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            schema_errors += 1
            continue

        if not all(k in obj for k in ["instruction", "input", "output"]):
            schema_errors += 1
            continue

        text = obj["instruction"] + obj["input"] + obj["output"]
        lengths.append(len(text))

        if len(sample_records) < 3:
            sample_records.append(obj)

print("Total records:", count)
print("Schema errors:", schema_errors)

if lengths:
    print("Min length (chars):", min(lengths))
    print("Max length (chars):", max(lengths))
    print("Average length (chars):", sum(lengths) // len(lengths))

print("\n--- SAMPLE RECORDS ---")
for i, rec in enumerate(sample_records, 1):
    print(f"\nSample {i}:")
    print("Instruction:", rec["instruction"][:200])
    print("Output:", rec["output"][:200])
