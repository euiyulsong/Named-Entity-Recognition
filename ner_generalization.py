# ner_generalization.py
#
# Goal
# ============================================================
# Train ONLY on single-entity / single-turn NER examples,
# then test whether the model generalizes to:
#
#   1. single entity
#   2. exactly 2 entities
#   3. 3+ entities
#   4. multi-entity
#   5. multi-type (different entity labels in one sentence)
#   6. increasing entity-count buckets
#   7. increasing label-type-count buckets
#
# This tests:
#
#   atomic NER learning
#        ->
#   compositional multi-entity / multi-label generalization
#
#
# Install:
#   pip install -U torch transformers datasets accelerate seqeval pandas numpy
#
# Run:
#   python3 ner_generalization.py
#
# Quick run:
#   QUICK=1 python3 ner_generalization.py
#
# Different model:
#   MODEL_NAME=distilbert-base-cased QUICK=1 python3 ner_generalization.py
#
# GPU:
#   CUDA_VISIBLE_DEVICES=0 python3 ner_generalization.py


import os
import random
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "bert-base-cased",
)

QUICK = os.environ.get(
    "QUICK",
    "0",
) == "1"

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "./ner_generalization_results",
)

MAX_LENGTH = 256

TRAIN_EPOCHS = 2 if QUICK else 4

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64

LEARNING_RATE = 2e-5

# QUICK mode limits
MAX_TRAIN = 2000 if QUICK else None
MAX_EVAL = 1000 if QUICK else None


# ============================================================
# REPRODUCIBILITY
# ============================================================

set_seed(SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 100)
print("CONFIG")
print("=" * 100)

print("MODEL_NAME       :", MODEL_NAME)
print("QUICK            :", QUICK)
print("OUTPUT_DIR       :", OUTPUT_DIR)
print("TRAIN_EPOCHS     :", TRAIN_EPOCHS)
print("CUDA AVAILABLE   :", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "GPU              :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 100)
print("LOADING CoNLL-2003")
print("=" * 100)


def load_conll():
    """
    Try normal HF dataset loading first.

    lhoestq/conll2003 is parquet-based, so unlike
    eriktks/conll2003 it does not require executing
    the old conll2003.py dataset script.

    If repo loading fails for some reason, fall back
    to explicit remote parquet URLs.
    """

    try:

        print(
            "[1] Trying:",
            'load_dataset("lhoestq/conll2003")'
        )

        data = load_dataset(
            "lhoestq/conll2003"
        )

        print("[OK] Hub dataset loaded.")

        return data

    except Exception as e:

        print()
        print("[WARNING]")
        print(
            "Direct repository loading failed:"
        )
        print(repr(e))

        print()
        print(
            "Falling back to explicit parquet URLs..."
        )

        base = (
            "https://huggingface.co/datasets/"
            "lhoestq/conll2003/resolve/main/data"
        )

        data_files = {
            "train":
                f"{base}/train-00000-of-00001.parquet",

            "validation":
                f"{base}/validation-00000-of-00001.parquet",

            "test":
                f"{base}/test-00000-of-00001.parquet",
        }

        data = load_dataset(
            "parquet",
            data_files=data_files,
        )

        print(
            "[OK] Explicit parquet loading succeeded."
        )

        return data


ds = load_conll()


print()
print(ds)

print()
print("train features:")
print(ds["train"].features)

print()
print("sample:")
print(ds["train"][0])


# ============================================================
# LABELS
# ============================================================

#
# IMPORTANT:
#
# CoNLL-2003 integer mapping:
#
# 0 O
# 1 B-PER
# 2 I-PER
# 3 B-ORG
# 4 I-ORG
# 5 B-LOC
# 6 I-LOC
# 7 B-MISC
# 8 I-MISC
#

DEFAULT_LABELS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]


def infer_label_names(dataset):

    try:

        feature = (
            dataset["train"]
            .features["ner_tags"]
        )

        if hasattr(feature, "feature"):

            inner = feature.feature

            if hasattr(inner, "names"):

                names = inner.names

                if names is not None:
                    return list(names)

    except Exception as e:

        print(
            "[WARNING] Could not infer labels:",
            repr(e)
        )

    return DEFAULT_LABELS


label_names = infer_label_names(ds)


print()
print("=" * 100)
print("LABELS")
print("=" * 100)

for i, label in enumerate(label_names):
    print(i, label)


id2label = {
    i: label
    for i, label in enumerate(label_names)
}

label2id = {
    label: i
    for i, label in enumerate(label_names)
}


# ============================================================
# BIO -> ENTITY SPANS
# ============================================================

def bio_to_spans(tokens, tag_ids):
    """
    Convert BIO tags to entity spans.

    Example:

    tokens:
      ["John", "Smith", "joined", "Google"]

    labels:
      B-PER I-PER O B-ORG

    output:
      [
        {
          type: PER,
          start: 0,
          end: 2,
          text: "John Smith"
        },
        {
          type: ORG,
          start: 3,
          end: 4,
          text: "Google"
        }
      ]
    """

    tags = [
        id2label[int(x)]
        for x in tag_ids
    ]

    spans = []

    current_type = None
    current_start = None

    # append artificial O to close last entity
    for i, tag in enumerate(
        tags + ["O"]
    ):

        if tag == "O":

            if current_type is not None:

                spans.append({
                    "start": current_start,
                    "end": i,
                    "type": current_type,
                    "text": " ".join(
                        tokens[current_start:i]
                    ),
                })

                current_type = None
                current_start = None

            continue

        if "-" not in tag:
            continue

        prefix, entity_type = tag.split(
            "-",
            1,
        )

        if prefix == "B":

            # Close previous entity if necessary
            if current_type is not None:

                spans.append({
                    "start": current_start,
                    "end": i,
                    "type": current_type,
                    "text": " ".join(
                        tokens[current_start:i]
                    ),
                })

            current_type = entity_type
            current_start = i

        elif prefix == "I":

            #
            # Invalid I-tag sequence:
            #
            # O I-PER
            #
            # treat I-PER as a new span.
            #

            if current_type != entity_type:

                if current_type is not None:

                    spans.append({
                        "start": current_start,
                        "end": i,
                        "type": current_type,
                        "text": " ".join(
                            tokens[current_start:i]
                        ),
                    })

                current_type = entity_type
                current_start = i

    return spans


# ============================================================
# EXAMPLE STATISTICS
# ============================================================

def get_stats(example):

    spans = bio_to_spans(
        example["tokens"],
        example["ner_tags"],
    )

    entity_types = [
        span["type"]
        for span in spans
    ]

    unique_types = set(
        entity_types
    )

    return {
        "num_entities":
            len(spans),

        "num_types":
            len(unique_types),

        "types":
            sorted(unique_types),

        "spans":
            spans,
    }


# ============================================================
# CHECK ORIGINAL DISTRIBUTION
# ============================================================

def dataset_distribution(dataset):

    entity_counts = Counter()
    type_counts = Counter()
    labels = Counter()

    for ex in dataset:

        stats = get_stats(ex)

        entity_counts[
            stats["num_entities"]
        ] += 1

        type_counts[
            stats["num_types"]
        ] += 1

        for t in stats["types"]:
            labels[t] += 1

    return (
        entity_counts,
        type_counts,
        labels,
    )


print()
print("=" * 100)
print("ORIGINAL TRAIN DISTRIBUTION")
print("=" * 100)

ec, tc, lc = dataset_distribution(
    ds["train"]
)

print("entity count:")
print(dict(sorted(ec.items())))

print()
print("number of unique entity types:")
print(dict(sorted(tc.items())))

print()
print("entity types:")
print(dict(lc))


# ============================================================
# SPLIT FUNCTIONS
# ============================================================

#
# Training:
#
# Exactly ONE entity span.
#
# This is our "atomic / single-hop" training set.
#

def single_entity_filter(example):

    stats = get_stats(example)

    return (
        stats["num_entities"] == 1
    )


#
# Exactly one entity AND one type.
#
# With one entity this is naturally one type,
# but keeping the condition explicit makes
# the experiment definition clear.
#

def atomic_filter(example):

    stats = get_stats(example)

    return (
        stats["num_entities"] == 1
        and
        stats["num_types"] == 1
    )


def two_entity_filter(example):

    stats = get_stats(example)

    return (
        stats["num_entities"] == 2
    )


def three_plus_filter(example):

    stats = get_stats(example)

    return (
        stats["num_entities"] >= 3
    )


def multi_entity_filter(example):

    stats = get_stats(example)

    return (
        stats["num_entities"] >= 2
    )


#
# Most important compositional test:
#
# At least 2 entities
# AND
# at least 2 distinct entity types.
#
# e.g.
#
#   John [PER]
#   Google [ORG]
#   London [LOC]
#

def multi_type_filter(example):

    stats = get_stats(example)

    return (
        stats["num_entities"] >= 2
        and
        stats["num_types"] >= 2
    )


#
# Multi entities but same label type.
#
# Example:
#
# John [PER] met Mary [PER].
#
# Useful to separate:
#
# entity-count generalization
#
# from
#
# label-composition generalization
#

def multi_entity_same_type_filter(
    example
):

    stats = get_stats(example)

    return (
        stats["num_entities"] >= 2
        and
        stats["num_types"] == 1
    )


# ============================================================
# BUILD DATASETS
# ============================================================

print()
print("=" * 100)
print("FILTER DATASETS")
print("=" * 100)


train_atomic = ds["train"].filter(
    atomic_filter,
    desc="train: atomic examples",
)

validation_atomic = ds[
    "validation"
].filter(
    atomic_filter,
    desc="validation: atomic examples",
)

test_atomic = ds["test"].filter(
    atomic_filter,
    desc="test: atomic examples",
)

test_two = ds["test"].filter(
    two_entity_filter,
    desc="test: 2 entities",
)

test_three_plus = ds["test"].filter(
    three_plus_filter,
    desc="test: 3+ entities",
)

test_multi = ds["test"].filter(
    multi_entity_filter,
    desc="test: multi entity",
)

test_multi_type = ds["test"].filter(
    multi_type_filter,
    desc="test: multi type",
)

test_same_type_multi = ds[
    "test"
].filter(
    multi_entity_same_type_filter,
    desc="test: multi entity same type",
)


# ============================================================
# QUICK LIMIT
# ============================================================

def limit_dataset(
    dataset,
    max_size,
    seed=SEED,
):

    if max_size is None:
        return dataset

    n = min(
        len(dataset),
        max_size,
    )

    return (
        dataset
        .shuffle(seed=seed)
        .select(range(n))
    )


train_atomic = limit_dataset(
    train_atomic,
    MAX_TRAIN,
)

validation_atomic = limit_dataset(
    validation_atomic,
    MAX_EVAL,
)

test_atomic = limit_dataset(
    test_atomic,
    MAX_EVAL,
)

test_two = limit_dataset(
    test_two,
    MAX_EVAL,
)

test_three_plus = limit_dataset(
    test_three_plus,
    MAX_EVAL,
)

test_multi = limit_dataset(
    test_multi,
    MAX_EVAL,
)

test_multi_type = limit_dataset(
    test_multi_type,
    MAX_EVAL,
)

test_same_type_multi = limit_dataset(
    test_same_type_multi,
    MAX_EVAL,
)


print()
print(
    f"train atomic            : "
    f"{len(train_atomic):,}"
)

print(
    f"validation atomic       : "
    f"{len(validation_atomic):,}"
)

print(
    f"test atomic             : "
    f"{len(test_atomic):,}"
)

print(
    f"test exactly 2 entities : "
    f"{len(test_two):,}"
)

print(
    f"test 3+ entities        : "
    f"{len(test_three_plus):,}"
)

print(
    f"test multi entity       : "
    f"{len(test_multi):,}"
)

print(
    f"test multi type         : "
    f"{len(test_multi_type):,}"
)

print(
    f"test multi same type    : "
    f"{len(test_same_type_multi):,}"
)


if len(train_atomic) == 0:

    raise RuntimeError(
        "No single-entity training examples found."
    )


# ============================================================
# DISPLAY EXAMPLES
# ============================================================

print()
print("=" * 100)
print("ATOMIC TRAIN EXAMPLES")
print("=" * 100)

for i in range(
    min(
        5,
        len(train_atomic),
    )
):

    ex = train_atomic[i]

    stats = get_stats(ex)

    print()
    print(
        "TEXT:",
        " ".join(ex["tokens"])
    )

    print(
        "ENTITIES:",
        stats["spans"]
    )


print()
print("=" * 100)
print("MULTI-TYPE TEST EXAMPLES")
print("=" * 100)

for i in range(
    min(
        10,
        len(test_multi_type),
    )
):

    ex = test_multi_type[i]

    stats = get_stats(ex)

    print()
    print(
        "TEXT:",
        " ".join(ex["tokens"])
    )

    print(
        "ENTITIES:",
        stats["spans"]
    )


# ============================================================
# TOKENIZER
# ============================================================

print()
print("=" * 100)
print("TOKENIZER")
print("=" * 100)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)


def tokenize_and_align_labels(
    examples
):

    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    aligned_labels = []

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

            elif (
                word_idx
                != previous_word_idx
            ):

                label_ids.append(
                    int(
                        ner_tags[
                            word_idx
                        ]
                    )
                )

            else:

                #
                # Ignore remaining subtokens.
                #
                # Example:
                #
                # Washington
                #
                # -> Wash ##ington
                #
                # Only first subtoken receives
                # the gold label.
                #

                label_ids.append(-100)

            previous_word_idx = word_idx

        aligned_labels.append(
            label_ids
        )

    tokenized["labels"] = (
        aligned_labels
    )

    return tokenized


def prepare_dataset(
    dataset
):

    return dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=
            dataset.column_names,
        desc="tokenize",
    )


print("Preparing train...")

tokenized_train = (
    prepare_dataset(
        train_atomic
    )
)

print("Preparing validation...")

tokenized_validation = (
    prepare_dataset(
        validation_atomic
    )
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 100)
print("MODEL")
print("=" * 100)

model = (
    AutoModelForTokenClassification
    .from_pretrained(
        MODEL_NAME,

        num_labels=
            len(label_names),

        id2label=id2label,

        label2id=label2id,
    )
)


data_collator = (
    DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )
)


# ============================================================
# METRICS
# ============================================================

def decode_predictions(
    predictions,
    labels,
):

    pred_ids = np.argmax(
        predictions,
        axis=-1,
    )

    pred_sequences = []
    gold_sequences = []

    for pred_seq, gold_seq in zip(
        pred_ids,
        labels,
    ):

        preds = []
        golds = []

        for pred_id, gold_id in zip(
            pred_seq,
            gold_seq,
        ):

            if gold_id == -100:
                continue

            preds.append(
                id2label[
                    int(pred_id)
                ]
            )

            golds.append(
                id2label[
                    int(gold_id)
                ]
            )

        pred_sequences.append(
            preds
        )

        gold_sequences.append(
            golds
        )

    return (
        pred_sequences,
        gold_sequences,
    )


def compute_metrics(
    eval_prediction
):

    predictions, labels = (
        eval_prediction
    )

    preds, golds = (
        decode_predictions(
            predictions,
            labels,
        )
    )

    precision = precision_score(
        golds,
        preds,
        zero_division=0,
    )

    recall = recall_score(
        golds,
        preds,
        zero_division=0,
    )

    f1 = f1_score(
        golds,
        preds,
        zero_division=0,
    )

    #
    # Exact sentence-level sequence match:
    #
    # Every token label must be correct.
    #

    sentence_exact = np.mean([
        pred == gold
        for pred, gold
        in zip(
            preds,
            golds,
        )
    ])

    #
    # Token accuracy
    #

    correct = 0
    total = 0

    for pred, gold in zip(
        preds,
        golds,
    ):

        for p, g in zip(
            pred,
            gold,
        ):

            correct += int(
                p == g
            )

            total += 1

    token_accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return {
        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "sentence_exact_match":
            float(sentence_exact),

        "token_accuracy":
            float(token_accuracy),
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

print()
print("=" * 100)
print("TRAINING")
print("=" * 100)


training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    learning_rate=
        LEARNING_RATE,

    per_device_train_batch_size=
        TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=
        EVAL_BATCH_SIZE,

    num_train_epochs=
        TRAIN_EPOCHS,

    weight_decay=0.01,

    #
    # Recent transformers:
    #
    # eval_strategy
    #
    # Older versions:
    #
    # evaluation_strategy
    #

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=20,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=1,

    fp16=torch.cuda.is_available(),

    report_to="none",

    seed=SEED,

    data_seed=SEED,
)


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=
        tokenized_train,

    eval_dataset=
        tokenized_validation,

    data_collator=
        data_collator,

    compute_metrics=
        compute_metrics,
)


trainer.train()


# ============================================================
# GENERAL EVALUATION FUNCTION
# ============================================================

def evaluate_split(
    split_name,
    raw_dataset,
):

    print()
    print("=" * 100)
    print(
        "EVALUATING:",
        split_name
    )
    print("=" * 100)

    if len(raw_dataset) == 0:

        print(
            "No examples.",
            "Skipping."
        )

        return None

    tokenized = (
        prepare_dataset(
            raw_dataset
        )
    )

    result = trainer.evaluate(
        eval_dataset=tokenized,
    )

    row = {

        "split":
            split_name,

        "n":
            len(raw_dataset),

        "precision":
            result[
                "eval_precision"
            ],

        "recall":
            result[
                "eval_recall"
            ],

        "f1":
            result[
                "eval_f1"
            ],

        "sentence_exact_match":
            result[
                "eval_sentence_exact_match"
            ],

        "token_accuracy":
            result[
                "eval_token_accuracy"
            ],
    }

    print()
    print(row)

    return row


# ============================================================
# MAIN GENERALIZATION TEST
# ============================================================

main_eval_sets = {

    "single_entity":
        test_atomic,

    "exactly_2_entities":
        test_two,

    "3plus_entities":
        test_three_plus,

    "multi_entity":
        test_multi,

    "multi_entity_same_type":
        test_same_type_multi,

    "multi_type":
        test_multi_type,
}


main_rows = []

for name, dataset in (
    main_eval_sets.items()
):

    row = evaluate_split(
        name,
        dataset,
    )

    if row is not None:
        main_rows.append(row)


main_df = pd.DataFrame(
    main_rows
)


print()
print("=" * 100)
print("MAIN GENERALIZATION RESULTS")
print("=" * 100)

print(
    main_df.to_string(
        index=False,

        float_format=
            lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# GENERALIZATION GAP
# ============================================================

print()
print("=" * 100)
print("GENERALIZATION GAP")
print("=" * 100)


baseline_rows = main_df[
    main_df["split"]
    == "single_entity"
]


gap_rows = []


if len(baseline_rows) > 0:

    baseline_f1 = float(
        baseline_rows.iloc[0]["f1"]
    )

    baseline_exact = float(
        baseline_rows.iloc[0][
            "sentence_exact_match"
        ]
    )

    for _, row in (
        main_df.iterrows()
    ):

        f1 = float(
            row["f1"]
        )

        exact = float(
            row[
                "sentence_exact_match"
            ]
        )

        f1_delta = (
            f1 - baseline_f1
        )

        exact_delta = (
            exact
            - baseline_exact
        )

        relative_f1_drop = (
            (
                baseline_f1 - f1
            )
            / baseline_f1
            if baseline_f1 > 0
            else 0.0
        )

        gap_rows.append({

            "split":
                row["split"],

            "f1":
                f1,

            "f1_delta":
                f1_delta,

            "relative_f1_drop":
                relative_f1_drop,

            "exact":
                exact,

            "exact_delta":
                exact_delta,
        })


gap_df = pd.DataFrame(
    gap_rows
)


if len(gap_df) > 0:

    print(
        gap_df.to_string(
            index=False,

            float_format=
                lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# ENTITY COUNT BUCKET TEST
# ============================================================

print()
print("=" * 100)
print("ENTITY COUNT GENERALIZATION")
print("=" * 100)


def entity_count_filter(
    wanted
):

    def _filter(example):

        n = get_stats(
            example
        )["num_entities"]

        if wanted == "4+":
            return n >= 4

        return n == wanted

    return _filter


entity_buckets = [
    ("1_entity", 1),
    ("2_entities", 2),
    ("3_entities", 3),
    ("4plus_entities", "4+"),
]


entity_rows = []

for bucket_name, condition in (
    entity_buckets
):

    subset = ds[
        "test"
    ].filter(
        entity_count_filter(
            condition
        ),
        desc=bucket_name,
    )

    subset = limit_dataset(
        subset,
        MAX_EVAL,
    )

    row = evaluate_split(
        bucket_name,
        subset,
    )

    if row is not None:
        entity_rows.append(row)


entity_df = pd.DataFrame(
    entity_rows
)


print()
print("=" * 100)
print("ENTITY COUNT RESULTS")
print("=" * 100)

print(
    entity_df.to_string(
        index=False,

        float_format=
            lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# LABEL TYPE COUNT TEST
# ============================================================

print()
print("=" * 100)
print("LABEL TYPE COMPOSITION TEST")
print("=" * 100)


def type_count_filter(
    wanted
):

    def _filter(example):

        n = get_stats(
            example
        )["num_types"]

        if wanted == "3+":
            return n >= 3

        return n == wanted

    return _filter


type_buckets = [
    ("1_type", 1),
    ("2_types", 2),
    ("3plus_types", "3+"),
]


type_rows = []

for bucket_name, condition in (
    type_buckets
):

    subset = ds[
        "test"
    ].filter(
        type_count_filter(
            condition
        ),
        desc=bucket_name,
    )

    subset = limit_dataset(
        subset,
        MAX_EVAL,
    )

    row = evaluate_split(
        bucket_name,
        subset,
    )

    if row is not None:
        type_rows.append(row)


type_df = pd.DataFrame(
    type_rows
)


print()
print("=" * 100)
print("LABEL TYPE RESULTS")
print("=" * 100)

print(
    type_df.to_string(
        index=False,

        float_format=
            lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# ENTITY TYPE-SPECIFIC TEST
# ============================================================

print()
print("=" * 100)
print("ENTITY TYPE SPECIFIC TEST")
print("=" * 100)


def contains_type(
    entity_type
):

    def _filter(example):

        stats = get_stats(
            example
        )

        return (
            entity_type
            in stats["types"]
        )

    return _filter


entity_type_rows = []

for entity_type in [
    "PER",
    "ORG",
    "LOC",
    "MISC",
]:

    subset = ds[
        "test"
    ].filter(
        contains_type(
            entity_type
        ),
        desc=f"type={entity_type}",
    )

    subset = limit_dataset(
        subset,
        MAX_EVAL,
    )

    row = evaluate_split(
        f"contains_{entity_type}",
        subset,
    )

    if row is not None:

        entity_type_rows.append(
            row
        )


entity_type_df = pd.DataFrame(
    entity_type_rows
)


print()
print(
    entity_type_df.to_string(
        index=False,

        float_format=
            lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# DETAILED ERROR ANALYSIS
# ============================================================

print()
print("=" * 100)
print("ERROR ANALYSIS ON MULTI-TYPE")
print("=" * 100)


def predict_raw_dataset(
    raw_dataset,
):

    tokenized = prepare_dataset(
        raw_dataset
    )

    output = trainer.predict(
        tokenized
    )

    preds, golds = (
        decode_predictions(
            output.predictions,
            output.label_ids,
        )
    )

    return preds, golds


error_examples = []


if len(test_multi_type) > 0:

    preds, golds = (
        predict_raw_dataset(
            test_multi_type
        )
    )

    for i, (
        pred,
        gold,
    ) in enumerate(
        zip(
            preds,
            golds,
        )
    ):

        if pred == gold:
            continue

        raw = (
            test_multi_type[i]
        )

        stats = get_stats(
            raw
        )

        error_examples.append({

            "text":
                " ".join(
                    raw["tokens"]
                ),

            "gold_entities":
                stats["spans"],

            "gold_tags":
                gold,

            "pred_tags":
                pred,

            "num_entities":
                stats[
                    "num_entities"
                ],

            "num_types":
                stats[
                    "num_types"
                ],
        })


print(
    "number of erroneous "
    "multi-type sentences:",
    len(error_examples)
)


print()
print("-" * 100)
print("ERROR EXAMPLES")
print("-" * 100)


for ex in error_examples[:20]:

    print()
    print(
        "TEXT:",
        ex["text"]
    )

    print(
        "GOLD ENTITIES:",
        ex["gold_entities"]
    )

    print(
        "GOLD TAGS:",
        ex["gold_tags"]
    )

    print(
        "PRED TAGS:",
        ex["pred_tags"]
    )


# ============================================================
# SAVE EVERYTHING
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


main_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "generalization_summary.csv",
    ),
    index=False,
)


if len(gap_df) > 0:

    gap_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "generalization_gap.csv",
        ),
        index=False,
    )


entity_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "entity_count_results.csv",
    ),
    index=False,
)


type_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "label_type_results.csv",
    ),
    index=False,
)


entity_type_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "entity_type_results.csv",
    ),
    index=False,
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "multi_type_errors.json",
    ),
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        error_examples,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_dir = os.path.join(
    OUTPUT_DIR,
    "best_model",
)


trainer.save_model(
    model_dir
)

tokenizer.save_pretrained(
    model_dir
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("FINAL RESULT")
print("=" * 100)

print()
print(
    main_df.to_string(
        index=False,

        float_format=
            lambda x:
            f"{x:.4f}",
    )
)


if len(gap_df) > 0:

    print()
    print(
        "GENERALIZATION GAP"
    )

    print(
        gap_df.to_string(
            index=False,

            float_format=
                lambda x:
                f"{x:.4f}",
        )
    )


print()
print("=" * 100)
print("SAVED FILES")
print("=" * 100)

print(
    os.path.join(
        OUTPUT_DIR,
        "generalization_summary.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "generalization_gap.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "entity_count_results.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "label_type_results.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "entity_type_results.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "multi_type_errors.json"
    )
)

print(
    model_dir
)
