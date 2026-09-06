# ner_generalization.py
#
# Experiment:
#   TRAIN:
#       CoNLL2003 sentences containing exactly ONE entity span
#
#   EVAL:
#       1. single_entity
#       2. multi_entity
#       3. multi_type
#       4. many_entity_3plus
#
# Model:
#       bert-base-cased token classifier
#
# Metrics:
#       entity-level precision / recall / F1
#       exact sentence match
#
# Run:
#   python ner_generalization.py
#
# Optional:
#   MODEL_NAME=roberta-base python ner_generalization.py

import os
import random
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "bert-base-cased"
)

OUTPUT_DIR = "./ner_singlehop_generalization"

MAX_LENGTH = 256
TRAIN_EPOCHS = 3
TRAIN_BATCH = 32
EVAL_BATCH = 64
LR = 2e-5

# Set this to e.g. 5000 if you want a faster experiment.
MAX_TRAIN = None


# ============================================================
# SEED
# ============================================================

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATA
# ============================================================

print("=" * 100)
print("LOADING CoNLL-2003")
print("=" * 100)

ds = load_dataset("eriktks/conll2003")

label_names = ds["train"].features["ner_tags"].feature.names

print("labels:")
print(label_names)

id2label = {
    i: label
    for i, label in enumerate(label_names)
}

label2id = {
    label: i
    for i, label in enumerate(label_names)
}


# ============================================================
# ENTITY SPAN UTILS
# ============================================================

def bio_to_spans(tokens, tag_ids):
    """
    Convert BIO token labels to:
        [
            {
                "start": 2,
                "end": 4,
                "type": "ORG",
                "text": "Bank of America"
            },
            ...
        ]
    """

    tags = [
        id2label[int(x)]
        for x in tag_ids
    ]

    spans = []

    cur_type = None
    cur_start = None

    for i, tag in enumerate(tags + ["O"]):

        if tag == "O":

            if cur_type is not None:
                spans.append({
                    "start": cur_start,
                    "end": i,
                    "type": cur_type,
                    "text": " ".join(
                        tokens[cur_start:i]
                    )
                })

                cur_type = None
                cur_start = None

            continue

        prefix, entity_type = tag.split("-", 1)

        if prefix == "B":

            if cur_type is not None:
                spans.append({
                    "start": cur_start,
                    "end": i,
                    "type": cur_type,
                    "text": " ".join(
                        tokens[cur_start:i]
                    )
                })

            cur_type = entity_type
            cur_start = i

        elif prefix == "I":

            if cur_type != entity_type:

                if cur_type is not None:
                    spans.append({
                        "start": cur_start,
                        "end": i,
                        "type": cur_type,
                        "text": " ".join(
                            tokens[cur_start:i]
                        )
                    })

                cur_type = entity_type
                cur_start = i

    return spans


def get_example_stats(example):

    spans = bio_to_spans(
        example["tokens"],
        example["ner_tags"]
    )

    types = set(
        span["type"]
        for span in spans
    )

    return {
        "n_entities": len(spans),
        "n_types": len(types),
        "types": sorted(types),
    }


# ============================================================
# SPLIT BY COMPLEXITY
# ============================================================

def is_single_entity(example):

    stats = get_example_stats(example)

    return (
        stats["n_entities"] == 1
    )


def is_multi_entity(example):

    stats = get_example_stats(example)

    return (
        stats["n_entities"] >= 2
    )


def is_multi_type(example):

    stats = get_example_stats(example)

    return (
        stats["n_entities"] >= 2
        and stats["n_types"] >= 2
    )


def is_three_plus(example):

    stats = get_example_stats(example)

    return (
        stats["n_entities"] >= 3
    )


train_single = ds["train"].filter(
    is_single_entity
)

val_single = ds["validation"].filter(
    is_single_entity
)

test_single = ds["test"].filter(
    is_single_entity
)

test_multi_entity = ds["test"].filter(
    is_multi_entity
)

test_multi_type = ds["test"].filter(
    is_multi_type
)

test_3plus = ds["test"].filter(
    is_three_plus
)


if MAX_TRAIN is not None:

    train_single = train_single.shuffle(
        seed=SEED
    ).select(
        range(
            min(MAX_TRAIN, len(train_single))
        )
    )


print()
print("=" * 100)
print("DATASET SIZE")
print("=" * 100)

print(
    f"train single entity : {len(train_single):,}"
)

print(
    f"valid single entity : {len(val_single):,}"
)

print(
    f"test single entity  : {len(test_single):,}"
)

print(
    f"test multi entity   : {len(test_multi_entity):,}"
)

print(
    f"test multi type     : {len(test_multi_type):,}"
)

print(
    f"test 3+ entity      : {len(test_3plus):,}"
)


# ============================================================
# CHECK EXAMPLES
# ============================================================

print()
print("=" * 100)
print("MULTI-TYPE EXAMPLES")
print("=" * 100)

for ex in test_multi_type.select(
    range(
        min(5, len(test_multi_type))
    )
):

    spans = bio_to_spans(
        ex["tokens"],
        ex["ner_tags"]
    )

    print()
    print(
        "TEXT:",
        " ".join(ex["tokens"])
    )

    print(
        "ENTITIES:",
        spans
    )


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


def tokenize_and_align_labels(examples):

    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
    )

    labels = []

    for batch_idx, ner_tags in enumerate(
        examples["ner_tags"]
    ):

        word_ids = tokenized.word_ids(
            batch_index=batch_idx
        )

        label_ids = []

        previous_word_idx = None

        for word_idx in word_ids:

            if word_idx is None:

                label_ids.append(-100)

            elif word_idx != previous_word_idx:

                label_ids.append(
                    ner_tags[word_idx]
                )

            else:

                # Ignore subsequent word pieces.
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized["labels"] = labels

    return tokenized


def prepare(dataset):

    return dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset.column_names
    )


tokenized_train = prepare(train_single)
tokenized_val = prepare(val_single)

eval_sets = {
    "single_entity": prepare(test_single),
    "multi_entity": prepare(test_multi_entity),
    "multi_type": prepare(test_multi_type),
    "three_plus_entities": prepare(test_3plus),
}


# ============================================================
# MODEL
# ============================================================

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
)


data_collator = DataCollatorForTokenClassification(
    tokenizer
)


# ============================================================
# METRIC
# ============================================================

def decode_predictions(
    predictions,
    labels
):

    pred_ids = np.argmax(
        predictions,
        axis=-1
    )

    true_predictions = []
    true_labels = []

    for prediction, label in zip(
        pred_ids,
        labels
    ):

        one_pred = []
        one_gold = []

        for p, l in zip(
            prediction,
            label
        ):

            if l == -100:
                continue

            one_pred.append(
                id2label[int(p)]
            )

            one_gold.append(
                id2label[int(l)]
            )

        true_predictions.append(
            one_pred
        )

        true_labels.append(
            one_gold
        )

    return (
        true_predictions,
        true_labels
    )


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    preds, golds = decode_predictions(
        predictions,
        labels
    )

    p = precision_score(
        golds,
        preds
    )

    r = recall_score(
        golds,
        preds
    )

    f1 = f1_score(
        golds,
        preds
    )

    exact = np.mean([
        pred == gold
        for pred, gold in zip(
            preds,
            golds
        )
    ])

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "sentence_exact_match": exact,
    }


# ============================================================
# TRAIN
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    learning_rate=LR,

    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=EVAL_BATCH,

    num_train_epochs=TRAIN_EPOCHS,

    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1",
    greater_is_better=True,

    fp16=torch.cuda.is_available(),

    logging_steps=20,

    report_to="none",

    seed=SEED,
)


trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,

    tokenizer=tokenizer,
    data_collator=data_collator,

    compute_metrics=compute_metrics,
)


print()
print("=" * 100)
print("TRAIN")
print("=" * 100)

trainer.train()


# ============================================================
# EVALUATION
# ============================================================

rows = []

print()
print("=" * 100)
print("GENERALIZATION TEST")
print("=" * 100)


for name, dataset in eval_sets.items():

    print()
    print("-" * 100)
    print(name)
    print("-" * 100)

    result = trainer.evaluate(
        eval_dataset=dataset,
        metric_key_prefix=name
    )

    row = {
        "split": name,
        "n": len(dataset),
        "precision": result[
            f"{name}_precision"
        ],
        "recall": result[
            f"{name}_recall"
        ],
        "f1": result[
            f"{name}_f1"
        ],
        "sentence_exact_match": result[
            f"{name}_sentence_exact_match"
        ],
    }

    rows.append(row)

    print(row)


df = pd.DataFrame(rows)

print()
print("=" * 100)
print("FINAL RESULT")
print("=" * 100)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# GENERALIZATION GAP
# ============================================================

single_f1 = df.loc[
    df["split"] == "single_entity",
    "f1"
].iloc[0]

print()
print("=" * 100)
print("GENERALIZATION GAP")
print("=" * 100)

for _, row in df.iterrows():

    gap = (
        row["f1"]
        - single_f1
    )

    relative = (
        gap / single_f1
        if single_f1 > 0
        else 0
    )

    print(
        f"{row['split']:25s}"
        f" F1={row['f1']:.4f}"
        f" delta={gap:+.4f}"
        f" relative={relative:+.2%}"
    )


# ============================================================
# ENTITY COUNT BUCKET EVAL
# ============================================================

def entity_count(example):

    return get_example_stats(
        example
    )["n_entities"]


buckets = {
    "1_entity": lambda x: entity_count(x) == 1,
    "2_entities": lambda x: entity_count(x) == 2,
    "3_entities": lambda x: entity_count(x) == 3,
    "4plus_entities": lambda x: entity_count(x) >= 4,
}


bucket_rows = []

print()
print("=" * 100)
print("ENTITY COUNT ANALYSIS")
print("=" * 100)

for bucket_name, func in buckets.items():

    subset = ds["test"].filter(
        func
    )

    if len(subset) == 0:
        continue

    tokenized = prepare(
        subset
    )

    result = trainer.evaluate(
        tokenized
    )

    bucket_rows.append({
        "bucket": bucket_name,
        "n": len(subset),
        "precision": result[
            "eval_precision"
        ],
        "recall": result[
            "eval_recall"
        ],
        "f1": result[
            "eval_f1"
        ],
        "exact_match": result[
            "eval_sentence_exact_match"
        ],
    })


bucket_df = pd.DataFrame(
    bucket_rows
)

print(
    bucket_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# NUMBER OF LABEL TYPES
# ============================================================

def n_types(example):

    return get_example_stats(
        example
    )["n_types"]


type_buckets = {
    "1_type": lambda x: n_types(x) == 1,
    "2_types": lambda x: n_types(x) == 2,
    "3plus_types": lambda x: n_types(x) >= 3,
}


type_rows = []

print()
print("=" * 100)
print("LABEL COMPOSITION ANALYSIS")
print("=" * 100)

for bucket_name, func in type_buckets.items():

    subset = ds["test"].filter(
        func
    )

    if len(subset) == 0:
        continue

    tokenized = prepare(
        subset
    )

    result = trainer.evaluate(
        tokenized
    )

    type_rows.append({
        "bucket": bucket_name,
        "n": len(subset),
        "precision": result[
            "eval_precision"
        ],
        "recall": result[
            "eval_recall"
        ],
        "f1": result[
            "eval_f1"
        ],
        "exact_match": result[
            "eval_sentence_exact_match"
        ],
    })


type_df = pd.DataFrame(
    type_rows
)

print(
    type_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "generalization_summary.csv"
    ),
    index=False
)

bucket_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "entity_count.csv"
    ),
    index=False
)

type_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "label_count.csv"
    ),
    index=False
)


trainer.save_model(
    os.path.join(
        OUTPUT_DIR,
        "best_model"
    )
)

tokenizer.save_pretrained(
    os.path.join(
        OUTPUT_DIR,
        "best_model"
    )
)


print()
print(
    "Saved under:",
    OUTPUT_DIR
)
