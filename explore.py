# explore_ner.py

from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt

ds = load_dataset("eriktks/conll2003")

print(ds)
print("\nFeatures:")
print(ds["train"].features)

label_names = ds["train"].features["ner_tags"].feature.names

print("\nNER labels:")
for i, name in enumerate(label_names):
    print(i, name)

# --------------------------------------------------
# Dataset sizes
# --------------------------------------------------
print("\nDataset sizes")
for split in ds:
    print(f"{split:12s}: {len(ds[split]):,}")

# --------------------------------------------------
# Tag distribution
# --------------------------------------------------
counter = Counter()

for row in ds["train"]:
    counter.update(row["ner_tags"])

print("\nTag distribution")
for idx, count in counter.most_common():
    print(f"{label_names[idx]:10s}: {count:,}")

# --------------------------------------------------
# Entity-level distribution
# --------------------------------------------------
entity_counter = Counter()

for idx, count in counter.items():
    tag = label_names[idx]

    if tag != "O":
        entity = tag.split("-")[-1]
        entity_counter[entity] += count

print("\nEntity token distribution")
for entity, count in entity_counter.most_common():
    print(f"{entity:8s}: {count:,}")

# --------------------------------------------------
# Sentence length
# --------------------------------------------------
lengths = [len(x["tokens"]) for x in ds["train"]]

print("\nSentence statistics")
print("avg:", sum(lengths) / len(lengths))
print("max:", max(lengths))
print("min:", min(lengths))

# --------------------------------------------------
# Actual examples
# --------------------------------------------------
print("\nExamples")

for row in ds["train"].select(range(10)):
    print("\nSentence:")
    print(" ".join(row["tokens"]))

    print("Entities:")
    for token, tag_id in zip(row["tokens"], row["ner_tags"]):
        tag = label_names[tag_id]

        if tag != "O":
            print(f"{token:20s} {tag}")

# --------------------------------------------------
# Plot
# --------------------------------------------------
names = [label_names[i] for i in counter.keys()]
values = [counter[i] for i in counter.keys()]

plt.figure(figsize=(10, 5))
plt.bar(names, values)
plt.xticks(rotation=45)
plt.title("CoNLL-2003 NER Tag Distribution")
plt.tight_layout()
plt.savefig("ner_tag_distribution.png")

print("\nSaved: ner_tag_distribution.png")
