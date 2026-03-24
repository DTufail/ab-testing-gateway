"""
Standalone script to create the golden test set for Phase 4 CI/CD validation.
Saves 200 shuffled examples from the BANKING77 test split to JSONL format.
"""
import json
import os
from collections import Counter
from datasets import load_dataset

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "golden_test_set.jsonl")

print("Loading PolyAI/banking77 dataset...")
dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)

label_names = dataset["test"].features["label"].names

print("Shuffling test split with seed=42 and selecting 200 examples...")
golden = dataset["test"].shuffle(seed=42).select(range(200))

print(f"Saving golden set to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w") as f:
    for example in golden:
        record = {
            "text": example["text"],
            "label": example["label"],
            "label_name": label_names[example["label"]],
        }
        f.write(json.dumps(record) + "\n")

label_counts = Counter(label_names[ex["label"]] for ex in golden)
print(f"\nGolden set saved: {len(golden)} examples across {len(label_counts)} unique labels.")
print("Label distribution (top 10):")
for label, count in label_counts.most_common(10):
    print(f"  {label}: {count}")
