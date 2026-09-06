# qwen35_ner_generalization.py
#
# ================================================================
# Experiment
# ================================================================
#
# Model:
#   Qwen/Qwen3.5-9B
#
# Compare:
#
#   A. ZERO-SHOT
#      pretrained/post-trained Qwen3.5-9B
#
#   B. SINGLE-ENTITY SFT
#      Fine-tune ONLY on CoNLL examples containing exactly
#      ONE entity span.
#
# Test:
#
#   single_entity
#   exactly_2_entities
#   3plus_entities
#   multi_entity
#   multi_entity_same_type
#   multi_type
#
#
# Metric:
#
# Exact entity tuple:
#
#   (surface text, entity type)
#
# Example:
#
#   John works at Google in London.
#
# Gold:
#   [
#     ("John", "PER"),
#     ("Google", "ORG"),
#     ("London", "LOC")
#   ]
#
#
# Run quick:
#
#   QUICK=1 python3 qwen35_ner_generalization.py
#
# Full:
#
#   python3 qwen35_ner_generalization.py
#
# Only evaluation:
#
#   DO_TRAIN=0 python3 qwen35_ner_generalization.py
#
# ================================================================

import os
import re
import gc
import json
import random
from collections import Counter

import numpy as np
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

from datasets import load_dataset, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)


# ================================================================
# CONFIG
# ================================================================

SEED = 42

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "Qwen/Qwen3.5-0.8B",
)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "./qwen35_ner_results",
)

ADAPTER_DIR = os.path.join(
    OUTPUT_DIR,
    "single_entity_lora",
)

QUICK = os.environ.get(
    "QUICK",
    "0",
) == "1"

DO_TRAIN = os.environ.get(
    "DO_TRAIN",
    "1",
) == "1"

MAX_TRAIN = (
    1000
    if QUICK
    else None
)

MAX_EVAL = (
    100
    if QUICK
    else None
)

EPOCHS = (
    1
    if QUICK
    else 1
)

TRAIN_BATCH_SIZE = 12

GRAD_ACCUM = 8

LEARNING_RATE = 2e-4

MAX_SEQ_LENGTH = 256

MAX_NEW_TOKENS = 180

TEMPERATURE = 0.0

# You can increase this if memory allows.
EVAL_BATCH_SIZE = int(
    os.environ.get(
        "EVAL_BATCH_SIZE",
        "50",
    )
)


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

print("MODEL               :", MODEL_NAME)
print("QUICK               :", QUICK)
print("DO_TRAIN            :", DO_TRAIN)
print("EPOCHS              :", EPOCHS)
print("MAX_TRAIN           :", MAX_TRAIN)
print("MAX_EVAL            :", MAX_EVAL)
print("EVAL_BATCH_SIZE     :", EVAL_BATCH_SIZE)

print(
    "CUDA                :",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU                 :",
        torch.cuda.get_device_name(0)
    )


# ================================================================
# DATASET
# ================================================================

print()
print("=" * 100)
print("LOADING CoNLL-2003")
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
    for i, x in enumerate(LABELS)
}


print(ds)
print("labels:", LABELS)


# ================================================================
# BIO -> ENTITY
# ================================================================

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

        prefix, ent_type = tag.split(
            "-",
            1,
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

            current_type = ent_type
            current_start = i

        elif prefix == "I":

            if current_type != ent_type:

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

                current_type = ent_type
                current_start = i

    return entities


def get_stats(example):

    ents = bio_to_entities(
        example["tokens"],
        example["ner_tags"],
    )

    types = {
        x["type"]
        for x in ents
    }

    return {
        "num_entities":
            len(ents),

        "num_types":
            len(types),

        "entities":
            ents,
    }


# ================================================================
# SPLITS
# ================================================================

def atomic_filter(x):

    s = get_stats(x)

    return (
        s["num_entities"] == 1
    )


def exactly_two_filter(x):

    return (
        get_stats(x)[
            "num_entities"
        ]
        == 2
    )


def three_plus_filter(x):

    return (
        get_stats(x)[
            "num_entities"
        ]
        >= 3
    )


def multi_entity_filter(x):

    return (
        get_stats(x)[
            "num_entities"
        ]
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


print()
print("=" * 100)
print("BUILDING SPLITS")
print("=" * 100)


train_atomic = ds["train"].filter(
    atomic_filter
)

test_atomic = ds["test"].filter(
    atomic_filter
)

test_two = ds["test"].filter(
    exactly_two_filter
)

test_three = ds["test"].filter(
    three_plus_filter
)

test_multi = ds["test"].filter(
    multi_entity_filter
)

test_same_type = ds["test"].filter(
    same_type_multi_filter
)

test_multi_type = ds["test"].filter(
    multi_type_filter
)


def limit_dataset(
    data,
    n,
):

    if n is None:
        return data

    n = min(
        n,
        len(data),
    )

    return (
        data
        .shuffle(seed=SEED)
        .select(range(n))
    )


train_atomic = limit_dataset(
    train_atomic,
    MAX_TRAIN,
)


test_sets = {

    "single_entity":
        limit_dataset(
            test_atomic,
            MAX_EVAL,
        ),

    "exactly_2_entities":
        limit_dataset(
            test_two,
            MAX_EVAL,
        ),

    "3plus_entities":
        limit_dataset(
            test_three,
            MAX_EVAL,
        ),

    "multi_entity":
        limit_dataset(
            test_multi,
            MAX_EVAL,
        ),

    "multi_entity_same_type":
        limit_dataset(
            test_same_type,
            MAX_EVAL,
        ),

    "multi_type":
        limit_dataset(
            test_multi_type,
            MAX_EVAL,
        ),
}


print(
    "train single entity:",
    len(train_atomic)
)

for name, data in test_sets.items():

    print(
        f"{name:28s}",
        len(data)
    )


# ================================================================
# PROMPT
# ================================================================

SYSTEM_PROMPT = """You are a named entity recognition system.

Extract all named entities from the input sentence.

Allowed entity types:
PER = person
ORG = organization
LOC = location
MISC = miscellaneous named entity

Return ONLY a JSON array.

Each entity must have exactly these keys:
"text"
"type"

Example output:
[{"text":"John Smith","type":"PER"},{"text":"Google","type":"ORG"}]

If there are no named entities, return:
[]

Do not explain your answer.
"""


def sentence_from_example(example):

    return " ".join(
        example["tokens"]
    )


def gold_json(example):

    entities = bio_to_entities(
        example["tokens"],
        example["ner_tags"],
    )

    return json.dumps(
        entities,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def make_user_prompt(example):

    sentence = sentence_from_example(
        example
    )

    return (
        "Sentence:\n"
        + sentence
        + "\n\nJSON:"
    )


# ================================================================
# TOKENIZER
# ================================================================

print()
print("=" * 100)
print("TOKENIZER")
print("=" * 100)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


tokenizer.padding_side = "left"


# ================================================================
# CHAT PROMPT
# ================================================================

def build_inference_prompt(example):

    messages = [
        {
            "role": "system",
            "content":
                SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content":
                make_user_prompt(
                    example
                ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_training_pair(example):

    messages = [
        {
            "role": "system",
            "content":
                SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content":
                make_user_prompt(
                    example
                ),
        },
    ]

    prompt = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    answer = (
        gold_json(example)
        +
        tokenizer.eos_token
    )

    return prompt, answer


# ================================================================
# JSON PARSER
# ================================================================

VALID_TYPES = {
    "PER",
    "ORG",
    "LOC",
    "MISC",
}


def extract_json_array(text):

    text = text.strip()

    # remove markdown fences if model ignored instruction
    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.I,
    )

    text = text.replace(
        "```",
        ""
    ).strip()

    # first try entire output
    try:

        obj = json.loads(text)

        if isinstance(obj, list):
            return obj

    except Exception:
        pass

    # find first JSON-looking array
    start = text.find("[")

    if start == -1:
        return []

    depth = 0

    in_string = False
    escape = False

    for i in range(
        start,
        len(text),
    ):

        ch = text[i]

        if escape:

            escape = False
            continue

        if ch == "\\" and in_string:

            escape = True
            continue

        if ch == '"':

            in_string = (
                not in_string
            )

            continue

        if in_string:
            continue

        if ch == "[":

            depth += 1

        elif ch == "]":

            depth -= 1

            if depth == 0:

                candidate = (
                    text[start:i + 1]
                )

                try:

                    obj = json.loads(
                        candidate
                    )

                    if isinstance(
                        obj,
                        list
                    ):
                        return obj

                except Exception:
                    return []

    return []


def normalize_prediction(text):

    parsed = extract_json_array(
        text
    )

    result = []

    for item in parsed:

        if not isinstance(
            item,
            dict
        ):
            continue

        ent_text = str(
            item.get(
                "text",
                ""
            )
        ).strip()

        ent_type = str(
            item.get(
                "type",
                ""
            )
        ).upper().strip()

        if not ent_text:
            continue

        if ent_type not in VALID_TYPES:
            continue

        result.append({
            "text":
                ent_text,

            "type":
                ent_type,
        })

    return result


# ================================================================
# ENTITY SCORING
# ================================================================

def normalize_surface(text):

    return " ".join(
        text.split()
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


def score_examples(
    gold_lists,
    pred_lists,
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    exact = 0

    for gold, pred in zip(
        gold_lists,
        pred_lists,
    ):

        gcounter = (
            entity_counter(gold)
        )

        pcounter = (
            entity_counter(pred)
        )

        tp = sum(
            (
                gcounter
                &
                pcounter
            ).values()
        )

        fp = (
            sum(
                pcounter.values()
            )
            - tp
        )

        fn = (
            sum(
                gcounter.values()
            )
            - tp
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if gcounter == pcounter:

            exact += 1

    precision = (
        total_tp
        /
        (total_tp + total_fp)
        if (
            total_tp + total_fp
        ) > 0
        else 0.0
    )

    recall = (
        total_tp
        /
        (total_tp + total_fn)
        if (
            total_tp + total_fn
        ) > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        /
        (precision + recall)
        if (
            precision + recall
        ) > 0
        else 0.0
    )

    sentence_exact = (
        exact
        /
        len(gold_lists)
        if gold_lists
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

        "tp":
            total_tp,

        "fp":
            total_fp,

        "fn":
            total_fn,
    }


# ================================================================
# LOAD MODEL
# ================================================================

def make_quant_config():

    return BitsAndBytesConfig(
        bnb_4bit_compute_dtype=
            torch.bfloat16,
    )


def load_base_model():

    print()
    print(
        "Loading:",
        MODEL_NAME
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(

            MODEL_NAME,

            trust_remote_code=True,

            torch_dtype=
                torch.bfloat16,

            device_map="auto",
        )
    )

    model.config.use_cache = True

    return model


# ================================================================
# GENERATION
# ================================================================

@torch.inference_mode()
def generate_predictions(
    model,
    dataset,
    desc,
):

    model.eval()

    gold_lists = []
    pred_lists = []
    detail_rows = []

    for start in tqdm(
        range(
            0,
            len(dataset),
            EVAL_BATCH_SIZE,
        ),
        desc=desc,
    ):

        end = min(
            start
            + EVAL_BATCH_SIZE,

            len(dataset),
        )

        examples = [
            dataset[i]
            for i in range(
                start,
                end,
            )
        ]

        prompts = [
            build_inference_prompt(
                ex
            )
            for ex in examples
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=
                MAX_SEQ_LENGTH,
        )

        device = next(
            model.parameters()
        ).device

        inputs = {
            k: v.to(device)
            for k, v
            in inputs.items()
        }

        prompt_len = (
            inputs[
                "input_ids"
            ].shape[1]
        )

        outputs = model.generate(

            **inputs,

            max_new_tokens=
                MAX_NEW_TOKENS,

            do_sample=False,

            pad_token_id=
                tokenizer.pad_token_id,

            eos_token_id=
                tokenizer.eos_token_id,
        )

        generated = outputs[
            :,
            prompt_len:
        ]

        texts = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        for ex, raw_output in zip(
            examples,
            texts,
        ):

            gold = bio_to_entities(
                ex["tokens"],
                ex["ner_tags"],
            )

            pred = (
                normalize_prediction(
                    raw_output
                )
            )

            gold_lists.append(
                gold
            )

            pred_lists.append(
                pred
            )

            detail_rows.append({

                "text":
                    sentence_from_example(
                        ex
                    ),

                "gold":
                    json.dumps(
                        gold,
                        ensure_ascii=False,
                    ),

                "prediction":
                    json.dumps(
                        pred,
                        ensure_ascii=False,
                    ),

                "raw_output":
                    raw_output,

                "num_entities":
                    get_stats(ex)[
                        "num_entities"
                    ],

                "num_types":
                    get_stats(ex)[
                        "num_types"
                    ],
            })

    scores = score_examples(
        gold_lists,
        pred_lists,
    )

    return (
        scores,
        detail_rows,
    )


def evaluate_all(
    model,
    method_name,
):

    rows = []

    all_details = []

    print()
    print("=" * 100)
    print(
        "EVALUATION:",
        method_name
    )
    print("=" * 100)

    for split_name, data in (
        test_sets.items()
    ):

        print()
        print(
            "split:",
            split_name
        )

        scores, details = (
            generate_predictions(
                model,
                data,
                desc=(
                    f"{method_name} "
                    f"{split_name}"
                ),
            )
        )

        row = {

            "method":
                method_name,

            "split":
                split_name,

            "n":
                len(data),

            **scores,
        }

        rows.append(
            row
        )

        for d in details:

            d["method"] = method_name
            d["split"] = split_name

            all_details.append(
                d
            )

        print(row)

    return (
        pd.DataFrame(rows),
        all_details,
    )


# ================================================================
# ZERO-SHOT
# ================================================================

print()
print("=" * 100)
print("ZERO-SHOT")
print("=" * 100)


base_model = load_base_model()


zero_df, zero_details = (
    evaluate_all(
        base_model,
        "zero_shot",
    )
)


zero_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "zero_shot_results.csv",
    ),
    index=False,
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "zero_shot_details.json",
    ),
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        zero_details,
        f,
        ensure_ascii=False,
        indent=2,
    )


print()
print("ZERO-SHOT RESULT")
print(
    zero_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ================================================================
# RELEASE ZERO-SHOT MODEL
# ================================================================

del base_model

gc.collect()

torch.cuda.empty_cache()


# ================================================================
# TRAINING DATASET
# ================================================================

class CausalNERDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        hf_dataset,
    ):

        self.data = (
            hf_dataset
        )

    def __len__(self):

        return len(
            self.data
        )

    def __getitem__(
        self,
        idx,
    ):

        ex = self.data[idx]

        prompt, answer = (
            build_training_pair(
                ex
            )
        )

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=
                MAX_SEQ_LENGTH,
        )["input_ids"]

        answer_ids = tokenizer(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=192,
        )["input_ids"]

        max_answer_room = (
            MAX_SEQ_LENGTH
            - len(prompt_ids)
        )

        if max_answer_room <= 0:

            prompt_ids = prompt_ids[
                :
                MAX_SEQ_LENGTH - 32
            ]

            max_answer_room = 32

        answer_ids = answer_ids[
            :
            max_answer_room
        ]

        input_ids = (
            prompt_ids
            + answer_ids
        )

        #
        # Important:
        #
        # loss is ONLY on assistant output.
        #
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


class DataCollator:

    def __call__(
        self,
        features,
    ):

        input_ids = [
            x["input_ids"]
            for x in features
        ]

        masks = [
            x["attention_mask"]
            for x in features
        ]

        labels = [
            x["labels"]
            for x in features
        ]

        input_ids = pad_sequence(

            input_ids,

            batch_first=True,

            padding_value=
                tokenizer.pad_token_id,
        )

        masks = pad_sequence(

            masks,

            batch_first=True,

            padding_value=0,
        )

        labels = pad_sequence(

            labels,

            batch_first=True,

            padding_value=-100,
        )

        return {

            "input_ids":
                input_ids,

            "attention_mask":
                masks,

            "labels":
                labels,
        }


# ================================================================
# QLORA TRAIN
# ================================================================

if DO_TRAIN:

    print()
    print("=" * 100)
    print("QLORA FINE-TUNING")
    print("=" * 100)

    train_model = (
        AutoModelForCausalLM
        .from_pretrained(

            MODEL_NAME,

            trust_remote_code=True,


            torch_dtype=
                torch.bfloat16,

            device_map="auto",
        )
    )

    train_model.config.use_cache = False

    train_model = (
        train_model
    )


    # ------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------

    #
    # Targeting common projection modules.
    #
    # If Qwen3.5 architecture/version complains about
    # a missing module, remove that name.
    #
    lora_config = LoraConfig(

        r=16,

        lora_alpha=32,

        lora_dropout=0.05,

        bias="none",

        task_type="CAUSAL_LM",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


    train_model = get_peft_model(
        train_model,
        lora_config,
    )


    train_model.print_trainable_parameters()


    train_dataset = (
        CausalNERDataset(
            train_atomic
        )
    )


    args = TrainingArguments(

        output_dir=
            os.path.join(
                OUTPUT_DIR,
                "trainer",
            ),

        num_train_epochs=
            EPOCHS,

        learning_rate=
            LEARNING_RATE,

        per_device_train_batch_size=
            TRAIN_BATCH_SIZE,

        gradient_accumulation_steps=
            GRAD_ACCUM,

        logging_steps=10,

        save_strategy="epoch",

        save_total_limit=1,

        bf16=True,

        fp16=False,

        gradient_checkpointing=True,

        optim="paged_adamw_8bit",

        lr_scheduler_type=
            "cosine",

        weight_decay=0.01,

        report_to="none",

        remove_unused_columns=False,

        seed=SEED,
    )


    trainer = Trainer(

        model=train_model,

        args=args,

        train_dataset=
            train_dataset,

        data_collator=
            DataCollator(),
    )


    trainer.train()


    train_model.save_pretrained(
        ADAPTER_DIR
    )

    tokenizer.save_pretrained(
        ADAPTER_DIR
    )


    del trainer
    del train_model

    gc.collect()

    torch.cuda.empty_cache()


# ================================================================
# LOAD FINE-TUNED MODEL
# ================================================================

print()
print("=" * 100)
print("LOAD FINE-TUNED MODEL")
print("=" * 100)


ft_base_model = (
    AutoModelForCausalLM
    .from_pretrained(

        MODEL_NAME,

        trust_remote_code=True,

        torch_dtype=
            torch.bfloat16,

        device_map="auto",
    )
)


ft_model = PeftModel.from_pretrained(

    ft_base_model,

    ADAPTER_DIR,
)


ft_model.eval()


# ================================================================
# FINE-TUNED EVAL
# ================================================================

ft_df, ft_details = (
    evaluate_all(
        ft_model,
        "single_entity_sft",
    )
)


ft_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "finetuned_results.csv",
    ),
    index=False,
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "finetuned_details.json",
    ),
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        ft_details,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ================================================================
# COMPARISON
# ================================================================

combined = pd.concat(
    [
        zero_df,
        ft_df,
    ],
    ignore_index=True,
)


combined.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_results.csv",
    ),
    index=False,
)


comparison = combined.pivot(
    index="split",
    columns="method",
    values=[
        "precision",
        "recall",
        "f1",
        "sentence_exact_match",
    ],
)


comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "comparison.csv",
    )
)


# ================================================================
# SFT DELTA
# ================================================================

merged = zero_df.merge(

    ft_df,

    on="split",

    suffixes=(
        "_zero",
        "_sft",
    ),
)


merged["f1_delta_sft"] = (
    merged["f1_sft"]
    -
    merged["f1_zero"]
)


merged["exact_delta_sft"] = (
    merged[
        "sentence_exact_match_sft"
    ]
    -
    merged[
        "sentence_exact_match_zero"
    ]
)


# ------------------------------------------------------------
# Generalization gap WITHIN each method
# ------------------------------------------------------------

for method in [
    "zero",
    "sft",
]:

    baseline = float(

        merged.loc[
            merged["split"]
            == "single_entity",

            f"f1_{method}",
        ].iloc[0]
    )

    merged[
        f"generalization_gap_{method}"
    ] = (
        merged[
            f"f1_{method}"
        ]
        -
        baseline
    )

    merged[
        f"relative_drop_{method}"
    ] = (
        (
            baseline
            -
            merged[
                f"f1_{method}"
            ]
        )
        /
        baseline
        if baseline > 0
        else 0
    )


merged.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "zero_vs_sft.csv",
    ),
    index=False,
)


# ================================================================
# FINAL OUTPUT
# ================================================================

display_cols = [

    "split",

    "n_zero",

    "f1_zero",

    "f1_sft",

    "f1_delta_sft",

    "sentence_exact_match_zero",

    "sentence_exact_match_sft",

    "generalization_gap_zero",

    "generalization_gap_sft",

    "relative_drop_zero",

    "relative_drop_sft",
]


print()
print("=" * 120)
print("ZERO-SHOT vs SINGLE-ENTITY SFT")
print("=" * 120)

print(
    merged[
        display_cols
    ].to_string(

        index=False,

        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ================================================================
# KEY CONTRAST
# ================================================================

print()
print("=" * 120)
print("KEY COMPOSITIONAL CONTRAST")
print("=" * 120)


for method in [
    "zero",
    "sft",
]:

    single = float(

        merged.loc[
            merged["split"]
            == "single_entity",

            f"f1_{method}",
        ].iloc[0]
    )

    same = float(

        merged.loc[
            merged["split"]
            == "multi_entity_same_type",

            f"f1_{method}",
        ].iloc[0]
    )

    multi_type = float(

        merged.loc[
            merged["split"]
            == "multi_type",

            f"f1_{method}",
        ].iloc[0]
    )

    print()
    print(method.upper())

    print(
        f"single entity          : "
        f"{single:.4f}"
    )

    print(
        f"same-type multi        : "
        f"{same:.4f}"
    )

    print(
        f"multi-type             : "
        f"{multi_type:.4f}"
    )

    print(
        f"same-type gap          : "
        f"{same-single:+.4f}"
    )

    print(
        f"multi-type gap         : "
        f"{multi_type-single:+.4f}"
    )


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(
    os.path.join(
        OUTPUT_DIR,
        "zero_shot_results.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "finetuned_results.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "zero_vs_sft.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "comparison.csv"
    )
)

print(
    ADAPTER_DIR
)
