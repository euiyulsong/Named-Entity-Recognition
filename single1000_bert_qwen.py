# single1000_bert_qwen.py
#
# ============================================================
# EXPERIMENT
# ============================================================
#
# Train:
#   EXACT SAME SIZE: 1,000 single-entity CoNLL-2003 examples
#
# Models:
#   1. bert-base-cased
#   2. Qwen/Qwen3.5-0.8B
#
# single-entity:
#   num_entities == 1
#
# Evaluation:
#   single_entity
#   exactly_2_entities
#   3plus_entities
#   multi_entity
#   multi_entity_same_type
#   multi_type
#
# Compare against existing MULTI-ENTITY-1000 results.
#
#
# Install:
#
# pip install -U \
#   torch transformers datasets accelerate \
#   peft seqeval pandas numpy tqdm
#
#
# Run:
#
# CUDA_VISIBLE_DEVICES=0 python3 single1000_bert_qwen.py
#
# Quick:
#
# QUICK=1 python3 single1000_bert_qwen.py
#
# ============================================================

import os
import re
import gc
import json
import math
import random
from collections import Counter

import numpy as np
import pandas as pd

import torch

from datasets import load_dataset

from tqdm import tqdm

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

BERT_MODEL = os.environ.get(
    "BERT_MODEL",
    "bert-base-cased",
)

QWEN_MODEL = os.environ.get(
    "QWEN_MODEL",
    "Qwen/Qwen3.5-0.8B",
)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "./single1000_results",
)

QUICK = (
    os.environ.get("QUICK", "0")
    == "1"
)


# ============================================================
# TRAIN SIZE
# ============================================================

SINGLE_TRAIN_N = (
    200 if QUICK
    else 1000
)


# ============================================================
# IMPORTANT
#
# Keep these EXACTLY the same as your multi1000 experiment.
# ============================================================

BERT_EPOCHS = (
    1 if QUICK
    else 4
)

QWEN_EPOCHS = (
    1 if QUICK
    else 3
)

MAX_EVAL = (
    200 if QUICK
    else None
)


# ============================================================
# BERT CONFIG
# ============================================================

BERT_MAX_LEN = 256

BERT_TRAIN_BATCH = 32

BERT_EVAL_BATCH = 64

BERT_LR = 2e-5


# ============================================================
# QWEN CONFIG
# ============================================================

QWEN_MAX_LEN = 256

QWEN_TRAIN_BATCH = 16

QWEN_GRAD_ACCUM = 1

QWEN_EVAL_BATCH = int(
    os.environ.get(
        "QWEN_EVAL_BATCH",
        "32",
    )
)

QWEN_LR = 2e-4

QWEN_MAX_NEW_TOKENS = 180


# ============================================================
# YOUR EXISTING MULTI-1000 RESULTS
# ============================================================

MULTI1000_BERT = {

    "single_entity":
        0.6767,

    "exactly_2_entities":
        0.8338,

    "3plus_entities":
        0.8064,

    "multi_entity":
        0.8174,

    "multi_entity_same_type":
        0.8622,

    "multi_type":
        0.8050,
}


MULTI1000_QWEN = {

    "single_entity":
        0.7046,

    "exactly_2_entities":
        0.8576,

    "3plus_entities":
        0.8324,

    "multi_entity":
        0.8433,

    "multi_entity_same_type":
        0.8365,

    "multi_type":
        0.8427,
}


# ============================================================
# SEED
# ============================================================

set_seed(SEED)

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


print("=" * 100)
print("CONFIG")
print("=" * 100)

print(
    "BERT MODEL       :",
    BERT_MODEL
)

print(
    "QWEN MODEL       :",
    QWEN_MODEL
)

print(
    "SINGLE TRAIN N   :",
    SINGLE_TRAIN_N
)

print(
    "BERT EPOCHS      :",
    BERT_EPOCHS
)

print(
    "QWEN EPOCHS      :",
    QWEN_EPOCHS
)

print(
    "CUDA             :",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU              :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD CoNLL
# ============================================================

print()
print("=" * 100)
print("LOAD CoNLL-2003")
print("=" * 100)


def load_conll():

    try:

        return load_dataset(
            "lhoestq/conll2003"
        )

    except Exception:

        base = (
            "https://huggingface.co/"
            "datasets/lhoestq/conll2003/"
            "resolve/main/data"
        )

        return load_dataset(
            "parquet",
            data_files={

                "train":
                    f"{base}/train-00000-of-00001.parquet",

                "validation":
                    f"{base}/validation-00000-of-00001.parquet",

                "test":
                    f"{base}/test-00000-of-00001.parquet",
            },
        )


ds = load_conll()


# ============================================================
# LABELS
# ============================================================

LABELS = [
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


try:

    names = (
        ds["train"]
        .features["ner_tags"]
        .feature.names
    )

    if names:
        LABELS = list(names)

except Exception:
    pass


ID2LABEL = {
    i: label
    for i, label
    in enumerate(LABELS)
}


LABEL2ID = {
    label: i
    for i, label
    in enumerate(LABELS)
}


print(ds)

print(
    "labels:",
    LABELS
)


# ============================================================
# BIO -> ENTITY
# ============================================================

def bio_to_entities(
    tokens,
    tag_ids,
):

    tags = [
        ID2LABEL[int(x)]
        for x in tag_ids
    ]

    entities = []

    current_type = None
    current_start = None


    for i, tag in enumerate(
        tags + ["O"]
    ):

        if tag == "O":

            if current_type is not None:

                entities.append({

                    "text":
                        " ".join(
                            tokens[
                                current_start:i
                            ]
                        ),

                    "type":
                        current_type,
                })

                current_type = None
                current_start = None

            continue


        if "-" not in tag:
            continue


        prefix, entity_type = (
            tag.split("-", 1)
        )


        if prefix == "B":

            if current_type is not None:

                entities.append({

                    "text":
                        " ".join(
                            tokens[
                                current_start:i
                            ]
                        ),

                    "type":
                        current_type,
                })


            current_type = entity_type

            current_start = i


        elif prefix == "I":

            if current_type != entity_type:

                if current_type is not None:

                    entities.append({

                        "text":
                            " ".join(
                                tokens[
                                    current_start:i
                                ]
                            ),

                        "type":
                            current_type,
                    })


                current_type = entity_type

                current_start = i


    return entities


def get_stats(example):

    entities = bio_to_entities(

        example["tokens"],

        example["ner_tags"],
    )


    entity_types = {

        entity["type"]

        for entity in entities
    }


    return {

        "num_entities":
            len(entities),

        "num_types":
            len(entity_types),

        "entities":
            entities,
    }


# ============================================================
# FILTERS
# ============================================================

def single_filter(x):

    return (
        get_stats(x)["num_entities"]
        == 1
    )


def exactly_two_filter(x):

    return (
        get_stats(x)["num_entities"]
        == 2
    )


def three_plus_filter(x):

    return (
        get_stats(x)["num_entities"]
        >= 3
    )


def multi_filter(x):

    return (
        get_stats(x)["num_entities"]
        >= 2
    )


def same_type_multi_filter(x):

    s = get_stats(x)

    return (

        s["num_entities"] >= 2

        and

        s["num_types"] == 1
    )


def multi_type_filter(x):

    s = get_stats(x)

    return (

        s["num_entities"] >= 2

        and

        s["num_types"] >= 2
    )


# ============================================================
# SINGLE-ENTITY TRAIN POOL
# ============================================================

print()
print("=" * 100)
print("BUILD SINGLE-ENTITY TRAIN SET")
print("=" * 100)


train_single_all = (
    ds["train"]
    .filter(
        single_filter,
        desc="single train"
    )
)


print(
    "all single-entity train:",
    len(train_single_all)
)


if len(train_single_all) < SINGLE_TRAIN_N:

    raise RuntimeError(
        f"Need {SINGLE_TRAIN_N}, "
        f"but only {len(train_single_all)} examples."
    )


# ============================================================
# IMPORTANT:
#
# select exactly 1000 with fixed seed
# ============================================================

train_single = (
    train_single_all
    .shuffle(seed=SEED)
    .select(
        range(
            SINGLE_TRAIN_N
        )
    )
)


print(
    "selected single train:",
    len(train_single)
)


# ============================================================
# CHECK TRAIN DISTRIBUTION
# ============================================================

type_counter = Counter()


for ex in train_single:

    s = get_stats(ex)

    assert (
        s["num_entities"] == 1
    )

    entity_type = (
        s["entities"][0]["type"]
    )

    type_counter[
        entity_type
    ] += 1


print()
print(
    "single-train label distribution:"
)

print(
    dict(
        sorted(
            type_counter.items()
        )
    )
)


# ============================================================
# SAVE TRAIN IDS
# ============================================================

train_ids = [
    str(x["id"])
    for x in train_single
]


with open(
    os.path.join(
        OUTPUT_DIR,
        "single1000_train_ids.json"
    ),
    "w",
) as f:

    json.dump(
        train_ids,
        f,
        indent=2,
    )


# ============================================================
# TEST SETS
# ============================================================

def limit_data(data):

    if MAX_EVAL is None:
        return data

    n = min(
        MAX_EVAL,
        len(data)
    )

    return (
        data
        .shuffle(seed=SEED)
        .select(
            range(n)
        )
    )


test_sets = {

    "single_entity":
        limit_data(
            ds["test"].filter(
                single_filter
            )
        ),

    "exactly_2_entities":
        limit_data(
            ds["test"].filter(
                exactly_two_filter
            )
        ),

    "3plus_entities":
        limit_data(
            ds["test"].filter(
                three_plus_filter
            )
        ),

    "multi_entity":
        limit_data(
            ds["test"].filter(
                multi_filter
            )
        ),

    "multi_entity_same_type":
        limit_data(
            ds["test"].filter(
                same_type_multi_filter
            )
        ),

    "multi_type":
        limit_data(
            ds["test"].filter(
                multi_type_filter
            )
        ),
}


print()
print("=" * 100)
print("TEST SPLITS")
print("=" * 100)


for name, data in (
    test_sets.items()
):

    print(
        f"{name:28s}",
        len(data)
    )


# ########################################################################
#
# BERT
#
# ########################################################################

print()
print()
print("#" * 100)
print("# BERT SINGLE-1000")
print("#" * 100)


bert_tokenizer = (
    AutoTokenizer.from_pretrained(
        BERT_MODEL,
        use_fast=True,
    )
)


# ============================================================
# BERT TOKEN ALIGNMENT
# ============================================================

def bert_tokenize(
    examples,
):

    encoded = bert_tokenizer(

        examples["tokens"],

        truncation=True,

        is_split_into_words=True,

        max_length=
            BERT_MAX_LEN,
    )


    labels = []


    for batch_idx, ner_tags in enumerate(
        examples["ner_tags"]
    ):

        word_ids = (
            encoded.word_ids(
                batch_index=batch_idx
            )
        )


        aligned = []

        previous_word = None


        for word_id in word_ids:

            if word_id is None:

                aligned.append(
                    -100
                )

            elif (
                word_id
                != previous_word
            ):

                aligned.append(
                    int(
                        ner_tags[
                            word_id
                        ]
                    )
                )

            else:

                aligned.append(
                    -100
                )


            previous_word = word_id


        labels.append(
            aligned
        )


    encoded["labels"] = labels

    return encoded


def bert_prepare(
    data,
):

    return data.map(

        bert_tokenize,

        batched=True,

        remove_columns=
            data.column_names,

        desc="BERT tokenize",
    )


bert_train = (
    bert_prepare(
        train_single
    )
)


# ============================================================
# BERT MODEL
# ============================================================

bert_model = (
    AutoModelForTokenClassification
    .from_pretrained(

        BERT_MODEL,

        num_labels=
            len(LABELS),

        id2label=
            ID2LABEL,

        label2id=
            LABEL2ID,
    )
)


bert_collator = (
    DataCollatorForTokenClassification(
        bert_tokenizer
    )
)


# ============================================================
# BERT METRIC
# ============================================================

def bert_decode(
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
                ID2LABEL[
                    int(pred_id)
                ]
            )

            golds.append(
                ID2LABEL[
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


def bert_metrics(
    eval_pred,
):

    predictions, labels = (
        eval_pred
    )


    preds, golds = (
        bert_decode(
            predictions,
            labels,
        )
    )


    precision = (
        precision_score(
            golds,
            preds,
            zero_division=0,
        )
    )


    recall = (
        recall_score(
            golds,
            preds,
            zero_division=0,
        )
    )


    f1 = f1_score(
        golds,
        preds,
        zero_division=0,
    )


    sentence_exact = float(
        np.mean([
            pred == gold
            for pred, gold
            in zip(
                preds,
                golds
            )
        ])
    )


    return {

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "sentence_exact_match":
            sentence_exact,
    }


# ============================================================
# BERT TRAIN
# ============================================================

bert_args = TrainingArguments(

    output_dir=
        os.path.join(
            OUTPUT_DIR,
            "bert_trainer",
        ),

    num_train_epochs=
        BERT_EPOCHS,

    learning_rate=
        BERT_LR,

    per_device_train_batch_size=
        BERT_TRAIN_BATCH,

    per_device_eval_batch_size=
        BERT_EVAL_BATCH,

    weight_decay=0.01,

    logging_steps=10,

    save_strategy="no",

    report_to="none",

    bf16=torch.cuda.is_available(),

    fp16=False,

    seed=SEED,
)


bert_trainer = Trainer(

    model=bert_model,

    args=bert_args,

    train_dataset=
        bert_train,

    data_collator=
        bert_collator,

    compute_metrics=
        bert_metrics,
)


print()
print("=" * 100)
print("TRAIN BERT SINGLE-1000")
print("=" * 100)


bert_trainer.train()


# ============================================================
# BERT EVALUATION
# ============================================================

bert_rows = []


for split_name, data in (
    test_sets.items()
):

    prepared = bert_prepare(
        data
    )


    result = (
        bert_trainer.evaluate(
            prepared
        )
    )


    bert_rows.append({

        "model":
            "BERT_single1000",

        "split":
            split_name,

        "n":
            len(data),

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
    })


bert_df = pd.DataFrame(
    bert_rows
)


print()
print("=" * 100)
print("BERT SINGLE-1000 RESULT")
print("=" * 100)


print(
    bert_df.to_string(

        index=False,

        float_format=
            lambda x:
            f"{x:.4f}",
    )
)


bert_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "bert_single1000.csv"
    ),

    index=False,
)


# ============================================================
# CLEAR BERT
# ============================================================

del bert_model
del bert_trainer

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ########################################################################
#
# QWEN
#
# ########################################################################

print()
print()
print("#" * 100)
print("# QWEN SINGLE-1000")
print("#" * 100)


qwen_tokenizer = (
    AutoTokenizer.from_pretrained(

        QWEN_MODEL,

        trust_remote_code=True,
    )
)


if (
    qwen_tokenizer.pad_token_id
    is None
):

    qwen_tokenizer.pad_token = (
        qwen_tokenizer.eos_token
    )


qwen_tokenizer.padding_side = (
    "left"
)


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a named entity recognition system.

Extract all named entities from the input sentence.

Allowed entity types:
PER = person
ORG = organization
LOC = location
MISC = miscellaneous named entity

Return ONLY a JSON array.

Each entity must have exactly:
"text"
"type"

Example:
[{"text":"John Smith","type":"PER"},{"text":"Google","type":"ORG"}]

If there are no entities:
[]

Do not explain.
"""


def sentence(
    example,
):

    return " ".join(
        example["tokens"]
    )


def gold_entities(
    example,
):

    return bio_to_entities(

        example["tokens"],

        example["ner_tags"],
    )


def gold_json(
    example,
):

    return json.dumps(

        gold_entities(
            example
        ),

        ensure_ascii=False,

        separators=(",", ":"),
    )


def user_prompt(
    example,
):

    return (
        "Sentence:\n"
        +
        sentence(example)
        +
        "\n\nJSON:"
    )


def qwen_prompt(
    example,
):

    messages = [

        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },

        {
            "role":
                "user",

            "content":
                user_prompt(
                    example
                ),
        },
    ]


    return (
        qwen_tokenizer
        .apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True,
        )
    )


# ============================================================
# QWEN TRAIN DATASET
# ============================================================

class QwenNERDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        data,
    ):

        self.data = data


    def __len__(
        self,
    ):

        return len(
            self.data
        )


    def __getitem__(
        self,
        idx,
    ):

        example = (
            self.data[idx]
        )


        prompt = (
            qwen_prompt(
                example
            )
        )


        answer = (

            gold_json(
                example
            )

            +

            qwen_tokenizer.eos_token
        )


        prompt_ids = (
            qwen_tokenizer(

                prompt,

                add_special_tokens=False,

            )["input_ids"]
        )


        answer_ids = (
            qwen_tokenizer(

                answer,

                add_special_tokens=False,

            )["input_ids"]
        )


        # ====================================================
        # truncation
        # ====================================================

        max_prompt_length = (

            QWEN_MAX_LEN

            -

            len(answer_ids)
        )


        if (
            len(prompt_ids)
            >
            max_prompt_length
        ):

            prompt_ids = (
                prompt_ids[
                    -max_prompt_length:
                ]
            )


        answer_room = (

            QWEN_MAX_LEN

            -

            len(prompt_ids)
        )


        answer_ids = (
            answer_ids[
                :answer_room
            ]
        )


        input_ids = (

            prompt_ids

            +

            answer_ids
        )


        # ====================================================
        # loss ONLY on assistant response
        # ====================================================

        labels = (

            [-100]
            * len(prompt_ids)

            +

            answer_ids
        )


        attention_mask = (

            [1]
            * len(input_ids)
        )


        return {

            "input_ids":
                torch.tensor(
                    input_ids,
                    dtype=torch.long,
                ),

            "attention_mask":
                torch.tensor(
                    attention_mask,
                    dtype=torch.long,
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.long,
                ),
        }


# ============================================================
# QWEN COLLATOR
# ============================================================

class QwenCollator:

    def __call__(
        self,
        features,
    ):

        max_len = max(

            len(
                x["input_ids"]
            )

            for x
            in features
        )


        # pad to multiple of 8
        max_len = (

            math.ceil(
                max_len / 8
            )

            * 8
        )


        batch_input_ids = []
        batch_masks = []
        batch_labels = []


        for feature in features:

            pad_n = (

                max_len

                -

                len(
                    feature[
                        "input_ids"
                    ]
                )
            )


            batch_input_ids.append(

                torch.cat([

                    feature[
                        "input_ids"
                    ],

                    torch.full(

                        (pad_n,),

                        qwen_tokenizer.pad_token_id,

                        dtype=torch.long,
                    ),
                ])
            )


            batch_masks.append(

                torch.cat([

                    feature[
                        "attention_mask"
                    ],

                    torch.zeros(

                        pad_n,

                        dtype=torch.long,
                    ),
                ])
            )


            batch_labels.append(

                torch.cat([

                    feature[
                        "labels"
                    ],

                    torch.full(

                        (pad_n,),

                        -100,

                        dtype=torch.long,
                    ),
                ])
            )


        return {

            "input_ids":
                torch.stack(
                    batch_input_ids
                ),

            "attention_mask":
                torch.stack(
                    batch_masks
                ),

            "labels":
                torch.stack(
                    batch_labels
                ),
        }


# ============================================================
# QWEN MODEL
# ============================================================

print()
print(
    "Loading Qwen:",
    QWEN_MODEL
)


qwen_model = (
    AutoModelForCausalLM
    .from_pretrained(

        QWEN_MODEL,

        trust_remote_code=True,

        dtype=
            torch.bfloat16,

        device_map={
            "": 0
        },

        attn_implementation=
            "sdpa",
    )
)


qwen_model.config.use_cache = (
    False
)


# ============================================================
# LoRA
# ============================================================

lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    task_type="CAUSAL_LM",

    bias="none",

    target_modules=[
        "q_proj",
        "v_proj",
    ],
)


qwen_model = (
    get_peft_model(

        qwen_model,

        lora_config,
    )
)


qwen_model.print_trainable_parameters()


qwen_train_dataset = (
    QwenNERDataset(
        train_single
    )
)


# ============================================================
# QWEN TRAINING
# ============================================================

qwen_args = TrainingArguments(

    output_dir=
        os.path.join(
            OUTPUT_DIR,
            "qwen_trainer",
        ),

    num_train_epochs=
        QWEN_EPOCHS,

    learning_rate=
        QWEN_LR,

    per_device_train_batch_size=
        QWEN_TRAIN_BATCH,

    gradient_accumulation_steps=
        QWEN_GRAD_ACCUM,

    bf16=True,

    fp16=False,

    gradient_checkpointing=False,

    optim="adamw_torch",

    logging_steps=10,

    save_strategy="no",

    report_to="none",

    remove_unused_columns=False,

    dataloader_num_workers=4,

    dataloader_pin_memory=True,

    tf32=True,

    seed=SEED,
)


qwen_trainer = Trainer(

    model=
        qwen_model,

    args=
        qwen_args,

    train_dataset=
        qwen_train_dataset,

    data_collator=
        QwenCollator(),
)


print()
print("=" * 100)
print("TRAIN QWEN SINGLE-1000")
print("=" * 100)


qwen_trainer.train()


qwen_model.config.use_cache = True


# ============================================================
# JSON PARSER
# ============================================================

VALID_TYPES = {
    "PER",
    "ORG",
    "LOC",
    "MISC",
}


def extract_json(
    text,
):

    text = (

        text

        .replace(
            "```json",
            ""
        )

        .replace(
            "```",
            ""
        )

        .strip()
    )


    try:

        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            list
        ):

            return parsed

    except Exception:
        pass


    start = (
        text.find("[")
    )


    if start < 0:
        return []


    depth = 0
    in_string = False
    escaped = False


    for i in range(
        start,
        len(text)
    ):

        char = text[i]


        if escaped:

            escaped = False
            continue


        if (
            char == "\\"
            and
            in_string
        ):

            escaped = True
            continue


        if char == '"':

            in_string = (
                not in_string
            )

            continue


        if in_string:
            continue


        if char == "[":

            depth += 1


        elif char == "]":

            depth -= 1


            if depth == 0:

                candidate = (
                    text[
                        start:i + 1
                    ]
                )


                try:

                    parsed = (
                        json.loads(
                            candidate
                        )
                    )


                    if isinstance(
                        parsed,
                        list
                    ):

                        return parsed


                except Exception:

                    return []


    return []


def normalize_prediction(
    text,
):

    parsed = extract_json(
        text
    )


    result = []


    for item in parsed:

        if not isinstance(
            item,
            dict
        ):

            continue


        entity_text = str(
            item.get(
                "text",
                ""
            )
        ).strip()


        entity_type = str(
            item.get(
                "type",
                ""
            )
        ).strip().upper()


        if not entity_text:
            continue


        if (
            entity_type
            not in VALID_TYPES
        ):

            continue


        result.append({

            "text":
                entity_text,

            "type":
                entity_type,
        })


    return result


# ============================================================
# QWEN ENTITY METRIC
# ============================================================

def normalize_surface(
    text,
):

    return " ".join(
        text.split()
    )


def entity_counter(
    entities,
):

    return Counter(

        (

            normalize_surface(
                entity["text"]
            ),

            entity["type"],
        )

        for entity
        in entities
    )


def score_entities(
    gold_lists,
    pred_lists,
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    exact = 0


    for gold, pred in zip(
        gold_lists,
        pred_lists
    ):

        gold_counter = (
            entity_counter(
                gold
            )
        )

        pred_counter = (
            entity_counter(
                pred
            )
        )


        tp = sum(

            (
                gold_counter
                &
                pred_counter
            ).values()
        )


        fp = (

            sum(
                pred_counter.values()
            )

            -

            tp
        )


        fn = (

            sum(
                gold_counter.values()
            )

            -

            tp
        )


        total_tp += tp
        total_fp += fp
        total_fn += fn


        if (
            gold_counter
            ==
            pred_counter
        ):

            exact += 1


    precision = (

        total_tp

        /

        (
            total_tp
            +
            total_fp
        )

        if (
            total_tp
            +
            total_fp
        ) > 0

        else 0.0
    )


    recall = (

        total_tp

        /

        (
            total_tp
            +
            total_fn
        )

        if (
            total_tp
            +
            total_fn
        ) > 0

        else 0.0
    )


    f1 = (

        2
        *
        precision
        *
        recall

        /

        (
            precision
            +
            recall
        )

        if (
            precision
            +
            recall
        ) > 0

        else 0.0
    )


    sentence_exact = (

        exact
        /
        len(gold_lists)

        if len(gold_lists) > 0

        else 0.0
    )


    return {

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "sentence_exact_match":
            sentence_exact,
    }


# ============================================================
# QWEN GENERATION
# ============================================================

@torch.inference_mode()
def qwen_eval(
    split_name,
    data,
):

    qwen_model.eval()


    gold_lists = []
    pred_lists = []


    for start in tqdm(

        range(
            0,
            len(data),
            QWEN_EVAL_BATCH,
        ),

        desc=
            split_name,
    ):

        end = min(

            start
            +
            QWEN_EVAL_BATCH,

            len(data)
        )


        examples = [

            data[i]

            for i in range(
                start,
                end
            )
        ]


        prompts = [

            qwen_prompt(
                example
            )

            for example
            in examples
        ]


        encoded = (
            qwen_tokenizer(

                prompts,

                padding=True,

                truncation=True,

                max_length=
                    QWEN_MAX_LEN,

                return_tensors=
                    "pt",
            )
        )


        encoded = {

            key:
                value.to(
                    qwen_model.device
                )

            for key, value
            in encoded.items()
        }


        prompt_len = (
            encoded[
                "input_ids"
            ].shape[1]
        )


        generated = (
            qwen_model.generate(

                **encoded,

                do_sample=False,

                max_new_tokens=
                    QWEN_MAX_NEW_TOKENS,

                pad_token_id=
                    qwen_tokenizer.pad_token_id,

                eos_token_id=
                    qwen_tokenizer.eos_token_id,
            )
        )


        output_ids = (
            generated[
                :,
                prompt_len:
            ]
        )


        outputs = (
            qwen_tokenizer
            .batch_decode(

                output_ids,

                skip_special_tokens=
                    True,
            )
        )


        for example, output in zip(
            examples,
            outputs,
        ):

            gold_lists.append(
                gold_entities(
                    example
                )
            )


            pred_lists.append(
                normalize_prediction(
                    output
                )
            )


    return score_entities(
        gold_lists,
        pred_lists,
    )


# ============================================================
# QWEN EVAL
# ============================================================

qwen_rows = []


for split_name, data in (
    test_sets.items()
):

    result = qwen_eval(
        split_name,
        data,
    )


    qwen_rows.append({

        "model":
            "Qwen_single1000",

        "split":
            split_name,

        "n":
            len(data),

        **result,
    })


qwen_df = pd.DataFrame(
    qwen_rows
)


print()
print("=" * 100)
print("QWEN SINGLE-1000 RESULT")
print("=" * 100)


print(
    qwen_df.to_string(

        index=False,

        float_format=
            lambda x:
            f"{x:.4f}",
    )
)


qwen_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "qwen_single1000.csv"
    ),

    index=False,
)


# ============================================================
# SINGLE1000 vs MULTI1000
# ============================================================

comparison_rows = []


for (
    model_name,
    single_df,
    multi_results
) in [

    (
        "BERT",
        bert_df,
        MULTI1000_BERT,
    ),

    (
        "Qwen",
        qwen_df,
        MULTI1000_QWEN,
    ),

]:

    for _, row in (
        single_df.iterrows()
    ):

        split = row[
            "split"
        ]


        single_f1 = float(
            row["f1"]
        )


        multi_f1 = float(
            multi_results[
                split
            ]
        )


        comparison_rows.append({

            "model":
                model_name,

            "split":
                split,

            "single1000_f1":
                single_f1,

            "multi1000_f1":
                multi_f1,

            "multi_minus_single":
                (
                    multi_f1
                    -
                    single_f1
                ),

            "relative_change":
                (

                    (
                        multi_f1
                        -
                        single_f1
                    )

                    /
                    single_f1

                    if single_f1 > 0

                    else 0.0
                ),
        })


comparison_df = pd.DataFrame(
    comparison_rows
)


print()
print()
print("=" * 120)
print(
    "SINGLE-1000 vs MULTI-1000"
)
print("=" * 120)


print(
    comparison_df.to_string(

        index=False,

        float_format=
            lambda x:
            f"{x:.4f}",
    )
)


comparison_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "single1000_vs_multi1000.csv"
    ),

    index=False,
)


# ============================================================
# FINAL SIDE-BY-SIDE
# ============================================================

final_rows = []


for split_name in (
    test_sets.keys()
):

    bert_single = float(

        bert_df.loc[

            bert_df["split"]
            ==
            split_name,

            "f1",

        ].iloc[0]
    )


    bert_multi = (
        MULTI1000_BERT[
            split_name
        ]
    )


    qwen_single = float(

        qwen_df.loc[

            qwen_df["split"]
            ==
            split_name,

            "f1",

        ].iloc[0]
    )


    qwen_multi = (
        MULTI1000_QWEN[
            split_name
        ]
    )


    final_rows.append({

        "split":
            split_name,

        "bert_single1000":
            bert_single,

        "bert_multi1000":
            bert_multi,

        "bert_delta":
            (
                bert_multi
                -
                bert_single
            ),

        "qwen_single1000":
            qwen_single,

        "qwen_multi1000":
            qwen_multi,

        "qwen_delta":
            (
                qwen_multi
                -
                qwen_single
            ),
    })


final_df = pd.DataFrame(
    final_rows
)


print()
print()
print("=" * 140)
print(
    "FINAL FAIR COMPARISON: TRAIN N = 1000"
)
print("=" * 140)


print(
    final_df.to_string(

        index=False,

        float_format=
            lambda x:
            f"{x:.4f}",
    )
)


final_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "final_single1000_vs_multi1000.csv"
    ),

    index=False,
)


# ============================================================
# CARDINALITY GENERALIZATION GAP
# ============================================================

print()
print()
print("=" * 120)
print("CARDINALITY GENERALIZATION")
print("=" * 120)


for model_name, single_df in [

    (
        "BERT",
        bert_df,
    ),

    (
        "Qwen",
        qwen_df,
    ),

]:

    single_f1 = float(

        single_df.loc[

            single_df["split"]
            ==
            "single_entity",

            "f1",

        ].iloc[0]
    )


    multi_f1 = float(

        single_df.loc[

            single_df["split"]
            ==
            "multi_entity",

            "f1",

        ].iloc[0]
    )


    multi_type_f1 = float(

        single_df.loc[

            single_df["split"]
            ==
            "multi_type",

            "f1",

        ].iloc[0]
    )


    print()
    print(
        model_name
    )


    print(
        f"single F1          : "
        f"{single_f1:.4f}"
    )


    print(
        f"multi F1           : "
        f"{multi_f1:.4f}"
    )


    print(
        f"multi-type F1      : "
        f"{multi_type_f1:.4f}"
    )


    print(
        f"single -> multi gap: "
        f"{multi_f1-single_f1:+.4f}"
    )


    print(
        f"single -> type gap : "
        f"{multi_type_f1-single_f1:+.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

bert_save_dir = (
    os.path.join(
        OUTPUT_DIR,
        "bert_single1000_model"
    )
)


# bert_model object was deleted to free GPU memory,
# but Trainer output can be omitted since primary goal is eval.
#
# If you want the BERT model saved, move trainer.save_model()
# before del bert_model above.


qwen_adapter_dir = (
    os.path.join(
        OUTPUT_DIR,
        "qwen_single1000_lora"
    )
)


qwen_model.save_pretrained(
    qwen_adapter_dir
)


qwen_tokenizer.save_pretrained(
    qwen_adapter_dir
)


# ============================================================
# SAVED FILES
# ============================================================

print()
print("=" * 100)
print("SAVED")
print("=" * 100)


for filename in [

    "single1000_train_ids.json",

    "bert_single1000.csv",

    "qwen_single1000.csv",

    "single1000_vs_multi1000.csv",

    "final_single1000_vs_multi1000.csv",

]:

    print(
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )


print(
    qwen_adapter_dir
)
