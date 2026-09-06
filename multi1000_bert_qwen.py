# multi1000_bert_qwen.py
#
# ============================================================
# EXPERIMENT
# ============================================================
#
# Train:
#   EXACT SAME 1,000 multi-entity CoNLL-2003 train examples
#
# Models:
#   1. bert-base-cased
#   2. Qwen/Qwen3.5-0.8B
#
# multi-entity:
#   num_entities >= 2
#
# Evaluation:
#   single_entity
#   exactly_2_entities
#   3plus_entities
#   multi_entity
#   multi_entity_same_type
#   multi_type
#
# Compare against previous SINGLE-ENTITY training results.
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
# CUDA_VISIBLE_DEVICES=0 python3 multi1000_bert_qwen.py
#
# Quick:
#
# QUICK=1 python3 multi1000_bert_qwen.py
#
# Qwen model override:
#
# QWEN_MODEL=Qwen/Qwen3.5-0.8B python3 multi1000_bert_qwen.py
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
from torch.nn.utils.rnn import pad_sequence

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
    PeftModel,
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
    "./multi1000_results",
)

QUICK = (
    os.environ.get("QUICK", "0")
    == "1"
)


# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------

MULTI_TRAIN_N = (
    200 if QUICK
    else 1000
)

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


# ------------------------------------------------------------
# BERT
# ------------------------------------------------------------

BERT_MAX_LEN = 256

BERT_TRAIN_BATCH = 32
BERT_EVAL_BATCH = 64

BERT_LR = 2e-5


# ------------------------------------------------------------
# QWEN
# ------------------------------------------------------------

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
# PREVIOUS SINGLE-ENTITY RESULTS
# ============================================================
#
# These are YOUR previous experimental results.
#
# Used only to print delta.
#
# ============================================================

PREVIOUS_BERT_SINGLE_TRAIN = {

    "single_entity":
        0.8775,

    "exactly_2_entities":
        0.7413,

    "3plus_entities":
        0.7825,

    "multi_entity":
        0.7659,

    "multi_entity_same_type":
        0.8732,

    "multi_type":
        0.7356,
}


PREVIOUS_QWEN_SINGLE_TRAIN = {

    "single_entity":
        0.8688,

    "exactly_2_entities":
        0.5606,

    "3plus_entities":
        0.3213,

    "multi_entity":
        0.4262,

    "multi_entity_same_type":
        0.5246,

    "multi_type":
        0.3964,
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


# ============================================================
# LOAD DATA
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
            }
        )


ds = load_conll()


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
    i: x
    for i, x
    in enumerate(LABELS)
}


LABEL2ID = {
    x: i
    for i, x
    in enumerate(LABELS)
}


print(ds)
print("labels:", LABELS)


# ============================================================
# BIO -> ENTITIES
# ============================================================

def bio_to_entities(
    tokens,
    tag_ids,
):

    tags = [
        ID2LABEL[int(x)]
        for x in tag_ids
    ]

    result = []

    current_type = None
    current_start = None

    for i, tag in enumerate(
        tags + ["O"]
    ):

        if tag == "O":

            if current_type is not None:

                result.append({

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

                result.append({

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

                    result.append({

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


    return result


def stats(example):

    entities = bio_to_entities(
        example["tokens"],
        example["ner_tags"],
    )

    types = {
        x["type"]
        for x in entities
    }

    return {

        "num_entities":
            len(entities),

        "num_types":
            len(types),

        "entities":
            entities,
    }


# ============================================================
# SPLITS
# ============================================================

def single_filter(x):

    return (
        stats(x)["num_entities"]
        == 1
    )


def exactly_two_filter(x):

    return (
        stats(x)["num_entities"]
        == 2
    )


def three_plus_filter(x):

    return (
        stats(x)["num_entities"]
        >= 3
    )


def multi_filter(x):

    return (
        stats(x)["num_entities"]
        >= 2
    )


def same_type_multi_filter(x):

    s = stats(x)

    return (
        s["num_entities"] >= 2
        and
        s["num_types"] == 1
    )


def multi_type_filter(x):

    s = stats(x)

    return (
        s["num_entities"] >= 2
        and
        s["num_types"] >= 2
    )


# ============================================================
# TRAIN = MULTI ENTITY
# ============================================================

print()
print("=" * 100)
print("BUILD MULTI-ENTITY TRAIN DATA")
print("=" * 100)


train_multi_all = (
    ds["train"]
    .filter(
        multi_filter,
        desc="multi train"
    )
)


print(
    "all multi-entity train:",
    len(train_multi_all)
)


if len(train_multi_all) < MULTI_TRAIN_N:

    raise RuntimeError(
        f"Need {MULTI_TRAIN_N} examples, "
        f"but only {len(train_multi_all)} exist."
    )


# ------------------------------------------------------------
# ONE SHARED SAMPLE
#
# IMPORTANT:
# BERT and Qwen use EXACT SAME examples.
# ------------------------------------------------------------

train_multi = (
    train_multi_all
    .shuffle(seed=SEED)
    .select(
        range(MULTI_TRAIN_N)
    )
)


print(
    "selected train:",
    len(train_multi)
)


# ============================================================
# TRAIN DISTRIBUTION
# ============================================================

entity_count_counter = Counter()
type_count_counter = Counter()


for ex in train_multi:

    s = stats(ex)

    entity_count_counter[
        s["num_entities"]
    ] += 1

    type_count_counter[
        s["num_types"]
    ] += 1


print()
print("entity count distribution:")
print(
    dict(
        sorted(
            entity_count_counter.items()
        )
    )
)


print()
print("type count distribution:")
print(
    dict(
        sorted(
            type_count_counter.items()
        )
    )
)


# ============================================================
# SAVE TRAIN IDS
# ============================================================

train_ids = [
    str(x["id"])
    for x in train_multi
]


with open(
    os.path.join(
        OUTPUT_DIR,
        "shared_train_ids.json"
    ),
    "w"
) as f:

    json.dump(
        train_ids,
        f,
        indent=2,
    )


# ============================================================
# TEST SPLITS
# ============================================================

def limit_data(data):

    if MAX_EVAL is None:
        return data

    return (
        data
        .shuffle(seed=SEED)
        .select(
            range(
                min(
                    MAX_EVAL,
                    len(data)
                )
            )
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


for name, data in test_sets.items():

    print(
        f"{name:28s}",
        len(data)
    )


# ################################################################
#
# BERT
#
# ################################################################

print()
print()
print("#" * 100)
print("# BERT MULTI-1000")
print("#" * 100)


bert_tokenizer = (
    AutoTokenizer.from_pretrained(
        BERT_MODEL,
        use_fast=True,
    )
)


def bert_tokenize(examples):

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

        previous = None


        for word_id in word_ids:

            if word_id is None:

                aligned.append(-100)

            elif word_id != previous:

                aligned.append(
                    int(
                        ner_tags[
                            word_id
                        ]
                    )
                )

            else:

                aligned.append(-100)


            previous = word_id


        labels.append(
            aligned
        )


    encoded["labels"] = labels

    return encoded


def bert_prepare(data):

    return data.map(

        bert_tokenize,

        batched=True,

        remove_columns=
            data.column_names,
    )


bert_train = (
    bert_prepare(
        train_multi
    )
)


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


def bert_decode(
    predictions,
    labels,
):

    pred_ids = np.argmax(
        predictions,
        axis=-1
    )

    preds = []
    golds = []


    for pseq, gseq in zip(
        pred_ids,
        labels
    ):

        pp = []
        gg = []


        for p, g in zip(
            pseq,
            gseq
        ):

            if g == -100:
                continue


            pp.append(
                ID2LABEL[int(p)]
            )

            gg.append(
                ID2LABEL[int(g)]
            )


        preds.append(pp)
        golds.append(gg)


    return preds, golds


def bert_metrics(eval_pred):

    predictions, labels = (
        eval_pred
    )


    preds, golds = bert_decode(
        predictions,
        labels,
    )


    return {

        "precision":
            precision_score(
                golds,
                preds,
                zero_division=0,
            ),

        "recall":
            recall_score(
                golds,
                preds,
                zero_division=0,
            ),

        "f1":
            f1_score(
                golds,
                preds,
                zero_division=0,
            ),

        "sentence_exact_match":
            float(
                np.mean([
                    p == g
                    for p, g
                    in zip(
                        preds,
                        golds
                    )
                ])
            ),
    }


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

    fp16=False,

    bf16=torch.cuda.is_available(),

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
print("TRAIN BERT")
print("=" * 100)


bert_trainer.train()


# ============================================================
# BERT EVAL
# ============================================================

bert_rows = []


for name, data in test_sets.items():

    prepared = bert_prepare(
        data
    )

    r = bert_trainer.evaluate(
        prepared
    )


    bert_rows.append({

        "model":
            "BERT_multi1000",

        "split":
            name,

        "n":
            len(data),

        "precision":
            r["eval_precision"],

        "recall":
            r["eval_recall"],

        "f1":
            r["eval_f1"],

        "sentence_exact_match":
            r[
                "eval_sentence_exact_match"
            ],
    })


bert_df = pd.DataFrame(
    bert_rows
)


print()
print("=" * 100)
print("BERT MULTI-1000 RESULT")
print("=" * 100)

print(
    bert_df.to_string(
        index=False,
        float_format=
            lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# CLEAR BERT
# ============================================================

del bert_model
del bert_trainer

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ################################################################
#
# QWEN
#
# ################################################################

print()
print()
print("#" * 100)
print("# QWEN MULTI-1000")
print("#" * 100)


qwen_tokenizer = (
    AutoTokenizer.from_pretrained(
        QWEN_MODEL,
        trust_remote_code=True,
    )
)


if qwen_tokenizer.pad_token_id is None:

    qwen_tokenizer.pad_token = (
        qwen_tokenizer.eos_token
    )


qwen_tokenizer.padding_side = "left"


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


def sentence(ex):

    return " ".join(
        ex["tokens"]
    )


def gold_entities(ex):

    return bio_to_entities(
        ex["tokens"],
        ex["ner_tags"],
    )


def gold_json(ex):

    return json.dumps(
        gold_entities(ex),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def user_prompt(ex):

    return (
        "Sentence:\n"
        + sentence(ex)
        + "\n\nJSON:"
    )


def qwen_prompt(ex):

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
                user_prompt(ex),
        },
    ]


    return qwen_tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True,
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


    def __len__(self):

        return len(
            self.data
        )


    def __getitem__(
        self,
        idx,
    ):

        ex = self.data[idx]


        prompt = qwen_prompt(
            ex
        )


        answer = (
            gold_json(ex)
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


        # ------------------------------------------------
        # Truncate prompt first if necessary
        # ------------------------------------------------

        max_prompt = (
            QWEN_MAX_LEN
            - len(answer_ids)
        )


        if len(prompt_ids) > max_prompt:

            prompt_ids = (
                prompt_ids[
                    -max_prompt:
                ]
            )


        answer_room = (
            QWEN_MAX_LEN
            - len(prompt_ids)
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


        labels = (

            [-100]
            * len(prompt_ids)

            +

            answer_ids
        )


        attention = (
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
                    attention,
                    dtype=torch.long,
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.long,
                ),
        }


# ============================================================
# RIGHT-PADDING TRAIN COLLATOR
# ============================================================

class QwenCollator:

    def __call__(
        self,
        features,
    ):

        max_len = max(
            len(x["input_ids"])
            for x in features
        )


        max_len = (
            math.ceil(
                max_len / 8
            )
            * 8
        )


        inputs = []
        masks = []
        labels = []


        for x in features:

            pad_n = (
                max_len
                - len(
                    x["input_ids"]
                )
            )


            inputs.append(

                torch.cat([

                    x["input_ids"],

                    torch.full(
                        (pad_n,),
                        qwen_tokenizer.pad_token_id,
                        dtype=torch.long,
                    )
                ])
            )


            masks.append(

                torch.cat([

                    x["attention_mask"],

                    torch.zeros(
                        pad_n,
                        dtype=torch.long,
                    )
                ])
            )


            labels.append(

                torch.cat([

                    x["labels"],

                    torch.full(
                        (pad_n,),
                        -100,
                        dtype=torch.long,
                    )
                ])
            )


        return {

            "input_ids":
                torch.stack(inputs),

            "attention_mask":
                torch.stack(masks),

            "labels":
                torch.stack(labels),
        }


# ============================================================
# LOAD QWEN
# ============================================================

print()
print("Loading:", QWEN_MODEL)


qwen_model = (
    AutoModelForCausalLM
    .from_pretrained(

        QWEN_MODEL,

        trust_remote_code=True,

        dtype=
            torch.bfloat16,

        device_map={"": 0},

        attn_implementation=
            "sdpa",
    )
)


qwen_model.config.use_cache = False


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


qwen_train_data = (
    QwenNERDataset(
        train_multi
    )
)


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

    model=qwen_model,

    args=qwen_args,

    train_dataset=
        qwen_train_data,

    data_collator=
        QwenCollator(),
)


print()
print("=" * 100)
print("TRAIN QWEN")
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


def extract_json(text):

    text = (
        text
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )


    try:

        x = json.loads(text)

        if isinstance(x, list):
            return x

    except Exception:
        pass


    start = text.find("[")

    if start < 0:
        return []


    depth = 0
    in_string = False
    escaped = False


    for i in range(
        start,
        len(text)
    ):

        c = text[i]


        if escaped:

            escaped = False
            continue


        if c == "\\" and in_string:

            escaped = True
            continue


        if c == '"':

            in_string = (
                not in_string
            )

            continue


        if in_string:
            continue


        if c == "[":

            depth += 1


        elif c == "]":

            depth -= 1


            if depth == 0:

                candidate = (
                    text[
                        start:i + 1
                    ]
                )


                try:

                    x = json.loads(
                        candidate
                    )

                    if isinstance(
                        x,
                        list
                    ):
                        return x

                except Exception:

                    return []


    return []


def normalize_prediction(
    text,
):

    raw = extract_json(
        text
    )

    result = []


    for item in raw:

        if not isinstance(
            item,
            dict
        ):
            continue


        surface = str(
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


        if not surface:
            continue


        if entity_type not in VALID_TYPES:
            continue


        result.append({

            "text":
                surface,

            "type":
                entity_type,
        })


    return result


# ============================================================
# SCORE QWEN
# ============================================================

def normalize_surface(x):

    return " ".join(
        x.split()
    )


def entity_counter(entities):

    return Counter(

        (
            normalize_surface(
                x["text"]
            ),

            x["type"],
        )

        for x in entities
    )


def entity_score(
    gold_lists,
    pred_lists,
):

    tp = 0
    fp = 0
    fn = 0

    exact = 0


    for gold, pred in zip(
        gold_lists,
        pred_lists
    ):

        g = entity_counter(
            gold
        )

        p = entity_counter(
            pred
        )


        x = sum(
            (g & p).values()
        )


        tp += x

        fp += (
            sum(p.values())
            - x
        )

        fn += (
            sum(g.values())
            - x
        )


        if g == p:
            exact += 1


    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )


    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )


    f1 = (
        2
        * precision
        * recall
        /
        (precision + recall)

        if precision + recall > 0

        else 0
    )


    return {

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "sentence_exact_match":
            (
                exact
                /
                len(gold_lists)
            ),
    }


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def qwen_eval(
    name,
    data,
):

    qwen_model.eval()


    gold_lists = []
    pred_lists = []


    for start in tqdm(

        range(
            0,
            len(data),
            QWEN_EVAL_BATCH
        ),

        desc=name,
    ):

        end = min(
            start
            + QWEN_EVAL_BATCH,

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
            qwen_prompt(ex)
            for ex in examples
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
            k: v.to(
                qwen_model.device
            )
            for k, v
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


        for ex, output in zip(
            examples,
            outputs
        ):

            gold_lists.append(
                gold_entities(ex)
            )

            pred_lists.append(
                normalize_prediction(
                    output
                )
            )


    return entity_score(
        gold_lists,
        pred_lists,
    )


# ============================================================
# QWEN EVAL ALL
# ============================================================

qwen_rows = []


for name, data in test_sets.items():

    score = qwen_eval(
        name,
        data,
    )


    qwen_rows.append({

        "model":
            "Qwen_multi1000",

        "split":
            name,

        "n":
            len(data),

        **score,
    })


qwen_df = pd.DataFrame(
    qwen_rows
)


print()
print("=" * 100)
print("QWEN MULTI-1000 RESULT")
print("=" * 100)


print(
    qwen_df.to_string(

        index=False,

        float_format=
            lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

bert_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "bert_multi1000.csv"
    ),

    index=False,
)


qwen_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "qwen_multi1000.csv"
    ),

    index=False,
)


# ============================================================
# COMPARE TO PREVIOUS SINGLE-ENTITY TRAIN
# ============================================================

comparison_rows = []


for model_name, df, baseline in [

    (
        "BERT",
        bert_df,
        PREVIOUS_BERT_SINGLE_TRAIN,
    ),

    (
        "Qwen",
        qwen_df,
        PREVIOUS_QWEN_SINGLE_TRAIN,
    ),

]:

    for _, row in df.iterrows():

        split = row["split"]

        old = baseline[
            split
        ]

        new = float(
            row["f1"]
        )


        comparison_rows.append({

            "model":
                model_name,

            "split":
                split,

            "single_entity_train_f1":
                old,

            "multi1000_train_f1":
                new,

            "delta":
                new - old,

            "relative_change":
                (
                    (new - old)
                    / old
                    if old > 0
                    else 0
                ),
        })


comparison_df = pd.DataFrame(
    comparison_rows
)


print()
print()
print("=" * 120)
print(
    "SINGLE-ENTITY TRAIN vs MULTI-ENTITY-1000 TRAIN"
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
        "single_vs_multi1000.csv"
    ),

    index=False,
)


# ============================================================
# SIDE-BY-SIDE
# ============================================================

side_rows = []


for split in test_sets:

    bert_new = float(
        bert_df.loc[
            bert_df["split"]
            == split,
            "f1",
        ].iloc[0]
    )


    qwen_new = float(
        qwen_df.loc[
            qwen_df["split"]
            == split,
            "f1",
        ].iloc[0]
    )


    side_rows.append({

        "split":
            split,

        "bert_single_train":
            PREVIOUS_BERT_SINGLE_TRAIN[
                split
            ],

        "bert_multi1000":
            bert_new,

        "bert_delta":
            (
                bert_new
                -
                PREVIOUS_BERT_SINGLE_TRAIN[
                    split
                ]
            ),

        "qwen_single_train":
            PREVIOUS_QWEN_SINGLE_TRAIN[
                split
            ],

        "qwen_multi1000":
            qwen_new,

        "qwen_delta":
            (
                qwen_new
                -
                PREVIOUS_QWEN_SINGLE_TRAIN[
                    split
                ]
            ),
    })


side_df = pd.DataFrame(
    side_rows
)


print()
print()
print("=" * 140)
print("FINAL SIDE-BY-SIDE")
print("=" * 140)


print(
    side_df.to_string(

        index=False,

        float_format=
            lambda x:
            f"{x:.4f}",
    )
)


side_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "final_side_by_side.csv"
    ),

    index=False,
)


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(
    os.path.join(
        OUTPUT_DIR,
        "shared_train_ids.json"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "bert_multi1000.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "qwen_multi1000.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "single_vs_multi1000.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "final_side_by_side.csv"
    )
)
