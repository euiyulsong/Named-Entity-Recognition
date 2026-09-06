import os
import re
import gc
import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_LLM = "Qwen/Qwen3.5-0.8B"
DEFAULT_ENCODER = "microsoft/deberta-v3-base"

DATA_DIR = Path("./multiturn_ner_data")
RESULT_DIR = Path("./multiturn_ner_results")

IGNORE_VALUES = {
    "",
    "none",
    "not mentioned",
    "notmentioned",
    "null",
}

# These are legitimate dialogue-state values.
# Do NOT remove dontcare.
LEGIT_SPECIAL_VALUES = {
    "dontcare",
    "yes",
    "no",
}

SEED = 42


# ============================================================
# UTILS
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(x):
    if x is None:
        return ""

    x = str(x).lower().strip()

    x = x.replace("’", "'")
    x = x.replace("‘", "'")
    x = x.replace("“", '"')
    x = x.replace("”", '"')

    x = re.sub(r"\s+", " ", x)

    return x


def normalize_slot(x):
    return normalize_text(x)


def dump_json(obj, path):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_json(path):
    with open(
        path,
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_jsonl(rows, path):
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_jsonl(path):
    rows = []

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


# ============================================================
# MULTIWOZ PARSING
# ============================================================

def is_user_turn(turn):
    speaker = turn.get("speaker")

    if isinstance(speaker, int):
        return speaker == 0

    return str(speaker).upper() == "USER"


def unpack_sequence(seq):
    if seq is None:
        return []

    if isinstance(seq, list):
        return seq

    if isinstance(seq, dict):
        if not seq:
            return []

        lengths = [
            len(v)
            for v in seq.values()
            if isinstance(v, list)
        ]

        if not lengths:
            return []

        n = min(lengths)

        return [
            {
                key: (
                    value[i]
                    if isinstance(value, list)
                    else value
                )
                for key, value in seq.items()
            }
            for i in range(n)
        ]

    raise TypeError(
        f"Unexpected sequence type: {type(seq)}"
    )


def extract_slot_values(frame):
    state = frame.get("state") or {}

    slot_values = unpack_sequence(
        state.get("slots_values", [])
    )

    result = {}

    for item in slot_values:

        if not isinstance(item, dict):
            continue

        slot = item.get(
            "slots_values_name"
        )

        values = item.get(
            "slots_values_list",
            []
        )

        if isinstance(values, str):
            values = [values]

        if not slot or not values:
            continue

        value = normalize_text(
            values[0]
        )

        slot = normalize_slot(slot)

        if value in IGNORE_VALUES:
            continue

        result[slot] = value

    return result


def extract_belief_state(turn):
    state = {}

    frames = unpack_sequence(
        turn.get("frames", [])
    )

    for frame in frames:

        local = extract_slot_values(
            frame
        )

        state.update(local)

    return state

def make_context(
    history,
    current,
    max_history_turns=12,
):
    history = history[
        -max_history_turns:
    ]

    pieces = list(history)

    pieces.append(
        f"[USER] {current}"
    )

    return "\n".join(pieces)


def find_value(
    text,
    value,
    use_last=True,
):
    """
    Case-insensitive exact substring match.
    Returns char offsets.
    """

    t = normalize_text(text)
    v = normalize_text(value)

    if not v:
        return None

    if use_last:
        idx = t.rfind(v)
    else:
        idx = t.find(v)

    if idx < 0:
        return None

    return idx, idx + len(v)


def find_source_hop(
    history_turns,
    current_utterance,
    value,
):
    """
    0  = current user turn
    1  = immediately previous utterance
    2+ = farther history

    Search most recent first.
    """

    v = normalize_text(value)

    if v in normalize_text(
        current_utterance
    ):
        return 0

    for hop, text in enumerate(
        reversed(history_turns),
        start=1,
    ):

        if v in normalize_text(text):
            return hop

    return None


def state_domain_count(state):
    domains = set()

    for slot in state:

        if "-" in slot:
            domains.add(
                slot.split(
                    "-",
                    1,
                )[0]
            )

    return len(domains)


def build_examples(
    split,
    max_history_turns=12,
):
    print(
        f"\nLoading MultiWOZ: {split}"
    )

    ds = load_dataset(
        "pfb30/multi_woz_v22",
        "v2.2_active_only",
        split=split,
    )

    examples = []

    for dialogue in tqdm(
        ds,
        desc=f"prepare:{split}",
    ):

        history = []

        dialogue_id = dialogue[
            "dialogue_id"
        ]

        services = list(
            dialogue.get(
                "services",
                []
            )
        )

        dialogue_multi_domain = (
            len(set(services)) >= 2
        )

        turns = unpack_sequence(
            dialogue["turns"]
        )

        for turn_index, turn in enumerate(
            turns
        ):

            utterance = (
                turn["utterance"]
                .strip()
            )

            if not is_user_turn(turn):

                history.append(
                    f"[SYSTEM] {utterance}"
                )

                continue

            state = extract_belief_state(
                turn
            )

            # 이하 기존 코드 그대로
            if not state:

                history.append(
                    f"[USER] {utterance}"
                )

                continue

            context = make_context(
                history,
                utterance,
                max_history_turns=
                    max_history_turns,
            )

            slot_meta = {}

            all_extractable = True
            has_implicit = False
            max_hop = 0

            for slot, value in state.items():

                in_current = (
                    normalize_text(value)
                    in normalize_text(
                        utterance
                    )
                )

                hop = find_source_hop(
                    history,
                    utterance,
                    value,
                )

                span = find_value(
                    context,
                    value,
                )

                extractable = (
                    span is not None
                )

                if not extractable:
                    all_extractable = False

                if not in_current:
                    has_implicit = True

                if hop is not None:
                    max_hop = max(
                        max_hop,
                        hop,
                    )

                slot_meta[slot] = {
                    "value": value,
                    "explicit":
                        in_current,
                    "hop": hop,
                    "extractable":
                        extractable,
                }

            examples.append({
                "dialogue_id":
                    dialogue_id,

                "turn_index":
                    turn_index,

                "turn_id":
                    turn.get(
                        "turn_id",
                        str(turn_index),
                    ),

                "services":
                    services,

                "dialogue_multi_domain":
                    dialogue_multi_domain,

                "state_domain_count":
                    state_domain_count(
                        state
                    ),

                "history":
                    history[
                        -max_history_turns:
                    ],

                "utterance":
                    utterance,

                "context":
                    context,

                "state":
                    state,

                "slot_meta":
                    slot_meta,

                "has_implicit":
                    has_implicit,

                "max_hop":
                    max_hop,

                "all_extractable":
                    all_extractable,
            })

            history.append(
                f"[USER] {utterance}"
            )

    return examples


# ============================================================
# HARD SUBSET
# ============================================================

def is_hard_example(
    x,
    min_hop=2,
):
    """
    Hard test:
    - dialogue is genuinely multi-domain
    - at least one inherited/context value
    - requires >= min_hop utterance distance
    - all target values physically appear somewhere in
      available context, so encoder span model has a fair shot.
    """

    return (
        x["dialogue_multi_domain"]
        and x["has_implicit"]
        and x["max_hop"] >= min_hop
        and x["all_extractable"]
    )


def prepare_data(args):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_slots = set()

    stats = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        examples = build_examples(
            split,
            max_history_turns=
                args.max_history_turns,
        )

        for x in examples:
            all_slots.update(
                x["state"].keys()
            )

        hard = [
            x
            for x in examples
            if is_hard_example(
                x,
                args.hard_min_hop,
            )
        ]

        write_jsonl(
            examples,
            DATA_DIR / f"{split}.jsonl",
        )

        write_jsonl(
            hard,
            DATA_DIR
            / f"{split}_hard.jsonl",
        )

        stats[split] = {
            "all": len(examples),
            "hard": len(hard),
        }

        print(
            f"{split:12s}",
            f"all={len(examples):6d}",
            f"hard={len(hard):6d}",
        )

    slots = sorted(all_slots)

    dump_json(
        slots,
        DATA_DIR / "slots.json",
    )

    dump_json(
        stats,
        DATA_DIR / "stats.json",
    )

    print(
        "\nSlot ontology:"
    )

    for s in slots:
        print(" ", s)

    print(
        "\n#slots:",
        len(slots),
    )


# ============================================================
# COMMON EVALUATION
# ============================================================

def pair_set(state):
    return {
        (
            normalize_slot(k),
            normalize_text(v),
        )
        for k, v in state.items()
    }


def score_states(
    predictions,
    examples,
):
    tp = 0
    fp = 0
    fn = 0

    em = []

    for pred, ex in zip(
        predictions,
        examples,
    ):

        gold = ex["state"]

        p = pair_set(pred)
        g = pair_set(gold)

        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)

        em.append(
            int(p == g)
        )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "n": len(examples),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "joint_em":
            float(np.mean(em))
            if em
            else 0.0,
    }


def evaluate_groups(
    predictions,
    examples,
):
    groups = {
        "overall":
            list(range(
                len(examples)
            )),

        "has_explicit":
            [],

        "has_implicit":
            [],

        "hop0":
            [],

        "hop1":
            [],

        "hop2":
            [],

        "hop3plus":
            [],

        "single_domain_dialogue":
            [],

        "multi_domain_dialogue":
            [],
    }

    for i, x in enumerate(examples):

        metas = x["slot_meta"].values()

        if any(
            m["explicit"]
            for m in metas
        ):
            groups[
                "has_explicit"
            ].append(i)

        if x["has_implicit"]:
            groups[
                "has_implicit"
            ].append(i)

        h = x["max_hop"]

        if h == 0:
            groups["hop0"].append(i)

        elif h == 1:
            groups["hop1"].append(i)

        elif h == 2:
            groups["hop2"].append(i)

        elif h >= 3:
            groups[
                "hop3plus"
            ].append(i)

        if x[
            "dialogue_multi_domain"
        ]:
            groups[
                "multi_domain_dialogue"
            ].append(i)
        else:
            groups[
                "single_domain_dialogue"
            ].append(i)

    results = {}

    for name, indices in groups.items():

        if not indices:
            continue

        p = [
            predictions[i]
            for i in indices
        ]

        e = [
            examples[i]
            for i in indices
        ]

        results[name] = score_states(
            p,
            e,
        )

    return results


def print_results(
    model_name,
    predictions,
    examples,
):
    results = evaluate_groups(
        predictions,
        examples,
    )

    rows = []

    for group, r in results.items():

        rows.append({
            "model": model_name,
            "group": group,
            **r,
        })

    df = pd.DataFrame(rows)

    print("\n")
    print("=" * 100)
    print(model_name)
    print("=" * 100)

    cols = [
        "group",
        "n",
        "precision",
        "recall",
        "f1",
        "joint_em",
    ]

    print(
        df[cols]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    return df


# ============================================================
# LLM PROMPT
# ============================================================

def build_llm_prompt(
    example,
    slots,
):
    slot_text = ", ".join(slots)

    return f"""Extract the complete dialogue belief state from the conversation.

The answer is a JSON object mapping slot names to slot values.

Important rules:
- Use the current user message AND previous dialogue turns.
- Recover values inherited from previous turns.
- Keep previously established constraints if they are still part of the user's current goal.
- Do not invent values.
- Use ONLY these slot names:
{slot_text}
- Return ONLY a valid JSON object.
- Do not include Markdown.
- Use lowercase slot names.

Conversation:
{example["context"]}

JSON:"""


def parse_json_object(text):
    text = text.strip()

    # remove fences
    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"```$",
        "",
        text,
    )

    start = text.find("{")
    end = text.rfind("}")

    if (
        start < 0
        or end < start
    ):
        return {}

    candidate = text[
        start:end + 1
    ]

    try:
        obj = json.loads(
            candidate
        )

    except Exception:
        return {}

    if not isinstance(
        obj,
        dict,
    ):
        return {}

    clean = {}

    for k, v in obj.items():

        if isinstance(
            v,
            (str, int, float),
        ):
            k = normalize_slot(k)
            v = normalize_text(v)

            if (
                k
                and v
                and v not in IGNORE_VALUES
            ):
                clean[k] = v

    return clean


# ============================================================
# LLM ZERO SHOT
# ============================================================

def load_llm(
    model_name,
    use_4bit=False,
):
    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_name,
            use_fast=True,
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    kwargs = {
        "device_map": "auto",
    }

    if use_4bit:

        kwargs[
            "quantization_config"
        ] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=
                torch.bfloat16,
            bnb_4bit_quant_type=
                "nf4",
            bnb_4bit_use_double_quant=
                True,
        )

    else:
        kwargs[
            "torch_dtype"
        ] = (
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_name,
            **kwargs,
        )
    )

    model.eval()

    return tokenizer, model


@torch.inference_mode()
def llm_generate_batch(
    tokenizer,
    model,
    prompts,
    max_new_tokens=256,
):
    """Generate a batch of JSON answers for LLM evaluation."""
    if not prompts:
        return []

    rendered = []
    for prompt in prompts:
        messages = [{
            "role": "user",
            "content": prompt,
        }]
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    # Decoder-only batched generation should left-pad.
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    tokenizer.padding_side = old_padding_side

    device = next(model.parameters()).device
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    outputs = model.generate(
        **inputs,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # All padded input rows have the same tensor length.
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[:, input_len:]

    return tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )


@torch.inference_mode()
def llm_generate_one(
    tokenizer,
    model,
    prompt,
    max_new_tokens=256,
):
    return llm_generate_batch(
        tokenizer,
        model,
        [prompt],
        max_new_tokens=max_new_tokens,
    )[0]


def run_llm_zero(args):
    slots = load_json(
        DATA_DIR / "slots.json"
    )

    path = (
        DATA_DIR
        / (
            "test_hard.jsonl"
            if args.hard
            else "test.jsonl"
        )
    )

    examples = read_jsonl(path)

    if args.max_eval is not None:
        examples = examples[:args.max_eval]

    tokenizer, model = load_llm(
        args.llm_model,
        args.llm_4bit,
    )

    predictions = []
    details = []

    batch_size = max(1, args.eval_batch_size)

    for start in tqdm(
        range(0, len(examples), batch_size),
        desc="LLM zero-shot batches",
    ):
        batch_examples = examples[
            start:start + batch_size
        ]

        prompts = [
            build_llm_prompt(x, slots)
            for x in batch_examples
        ]

        raws = llm_generate_batch(
            tokenizer,
            model,
            prompts,
            args.max_new_tokens,
        )

        for x, raw in zip(batch_examples, raws):
            pred = parse_json_object(raw)
            predictions.append(pred)
            details.append({
                "dialogue_id": x["dialogue_id"],
                "turn_id": x["turn_id"],
                "gold": x["state"],
                "pred": pred,
                "raw": raw,
            })

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        details,
        RESULT_DIR / "llm_zero_details.jsonl",
    )

    df = print_results(
        "LLM_ZERO",
        predictions,
        examples,
    )

    df.to_csv(
        RESULT_DIR / "llm_zero_metrics.csv",
        index=False,
    )


# ============================================================
# LLM SFT DATASET
# ============================================================

class CausalSFTDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        rows,
        tokenizer,
        slots,
        max_length=2048,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.slots = slots
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):

        x = self.rows[idx]

        prompt = build_llm_prompt(
            x,
            self.slots,
        )

        answer = json.dumps(
            x["state"],
            ensure_ascii=False,
            sort_keys=True,
        )

        prompt_messages = [{
            "role": "user",
            "content": prompt,
        }]

        full_messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        prompt_text = (
            self.tokenizer
            .apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=
                    True,
            )
        )

        full_text = (
            self.tokenizer
            .apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=
                    False,
            )
        )

        prompt_ids = (
            self.tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]
        )

        full = self.tokenizer(
            full_text,
            max_length=
                self.max_length,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = full[
            "input_ids"
        ]

        attention_mask = full[
            "attention_mask"
        ]

        labels = input_ids.copy()

        prompt_len = min(
            len(prompt_ids),
            len(labels),
        )

        labels[
            :prompt_len
        ] = [-100] * prompt_len

        return {
            "input_ids":
                input_ids,
            "attention_mask":
                attention_mask,
            "labels":
                labels,
        }


class CausalCollator:

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(
        self,
        features,
    ):
        max_len = max(
            len(x["input_ids"])
            for x in features
        )

        batch_ids = []
        batch_mask = []
        batch_labels = []

        for x in features:

            pad_len = (
                max_len
                - len(x["input_ids"])
            )

            ids = (
                x["input_ids"]
                + [
                    self.tokenizer
                    .pad_token_id
                ] * pad_len
            )

            mask = (
                x["attention_mask"]
                + [0] * pad_len
            )

            labels = (
                x["labels"]
                + [-100] * pad_len
            )

            batch_ids.append(ids)
            batch_mask.append(mask)
            batch_labels.append(
                labels
            )

        return {
            "input_ids":
                torch.tensor(
                    batch_ids,
                    dtype=torch.long,
                ),
            "attention_mask":
                torch.tensor(
                    batch_mask,
                    dtype=torch.long,
                ),
            "labels":
                torch.tensor(
                    batch_labels,
                    dtype=torch.long,
                ),
        }


# ============================================================
# LLM LoRA TRAIN
# ============================================================

def train_llm(args):

    slots = load_json(
        DATA_DIR / "slots.json"
    )

    train_rows = read_jsonl(
        DATA_DIR / "train.jsonl"
    )

    random.Random(
        SEED
    ).shuffle(train_rows)

    if args.max_train:
        train_rows = train_rows[
            :args.max_train
        ]

    tokenizer, model = load_llm(
        args.llm_model,
        args.llm_4bit,
    )

    if args.llm_4bit:

        from peft import (
            prepare_model_for_kbit_training
        )

        model = (
            prepare_model_for_kbit_training(
                model
            )
        )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=
            args.lora_alpha,
        lora_dropout=0.05,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    model.print_trainable_parameters()

    dataset = CausalSFTDataset(
        train_rows,
        tokenizer,
        slots,
        max_length=
            args.llm_max_length,
    )

    collator = CausalCollator(
        tokenizer
    )

    training_args = (
        TrainingArguments(
            output_dir=
                args.llm_output,

            num_train_epochs=
                args.llm_epochs,

            per_device_train_batch_size=
                args.llm_batch,

            gradient_accumulation_steps=
                args.llm_grad_accum,

            learning_rate=
                args.llm_lr,

            warmup_ratio=0.03,

            weight_decay=0.01,

            logging_steps=10,

            save_strategy="epoch",

            bf16=(
                torch.cuda.is_available()
                and torch.cuda
                    .is_bf16_supported()
            ),

            fp16=(
                torch.cuda.is_available()
                and not torch.cuda
                    .is_bf16_supported()
            ),

            gradient_checkpointing=
                True,

            report_to="none",

            remove_unused_columns=
                False,

            dataloader_num_workers=2,
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model(
        args.llm_output
    )

    tokenizer.save_pretrained(
        args.llm_output
    )


# ============================================================
# LLM SFT EVAL
# ============================================================

def run_llm_sft_eval(args):
    slots = load_json(
        DATA_DIR / "slots.json"
    )

    examples = read_jsonl(
        DATA_DIR
        / (
            "test_hard.jsonl"
            if args.hard
            else "test.jsonl"
        )
    )

    if args.max_eval is not None:
        examples = examples[:args.max_eval]

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_output
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "device_map": "auto",
    }

    if args.llm_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = (
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        )

    base = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        **kwargs,
    )

    model = PeftModel.from_pretrained(
        base,
        args.llm_output,
    )
    model.eval()

    predictions = []
    details = []
    batch_size = max(1, args.eval_batch_size)

    for start in tqdm(
        range(0, len(examples), batch_size),
        desc="LLM SFT eval batches",
    ):
        batch_examples = examples[
            start:start + batch_size
        ]

        prompts = [
            build_llm_prompt(x, slots)
            for x in batch_examples
        ]

        raws = llm_generate_batch(
            tokenizer,
            model,
            prompts,
            args.max_new_tokens,
        )

        for x, raw in zip(batch_examples, raws):
            pred = parse_json_object(raw)
            predictions.append(pred)
            details.append({
                "dialogue_id": x["dialogue_id"],
                "turn_id": x["turn_id"],
                "gold": x["state"],
                "pred": pred,
                "raw": raw,
            })

    write_jsonl(
        details,
        RESULT_DIR / "llm_sft_details.jsonl",
    )

    df = print_results(
        "LLM_SFT",
        predictions,
        examples,
    )

    df.to_csv(
        RESULT_DIR / "llm_sft_metrics.csv",
        index=False,
    )


# ============================================================
# BERT/DEBERTA QA DATA
# ============================================================

def find_last_char_span(
    context,
    value,
):
    context_lower = (
        context.lower()
    )

    value_lower = (
        str(value)
        .lower()
        .strip()
    )

    start = context_lower.rfind(
        value_lower
    )

    if start < 0:
        return None

    return (
        start,
        start + len(value_lower),
    )


def align_char_to_tokens(
    encoded,
    start_char,
    end_char,
):
    """
    encoded must contain offset_mapping.
    sequence_id 1 = context.
    """

    offsets = encoded[
        "offset_mapping"
    ]

    seq_ids = encoded.sequence_ids()

    start_token = None
    end_token = None

    for i, (
        off,
        seq_id,
    ) in enumerate(
        zip(offsets, seq_ids)
    ):

        if seq_id != 1:
            continue

        a, b = off

        if (
            start_token is None
            and a <= start_char < b
        ):
            start_token = i

        if (
            a < end_char <= b
        ):
            end_token = i

    if (
        start_token is None
        or end_token is None
    ):
        return None

    return (
        start_token,
        end_token,
    )


class SlotQATrainDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        rows,
        tokenizer,
        slots,
        max_length=512,
        negative_ratio=3,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.slots = slots
        self.max_length = max_length

        rng = random.Random(SEED)

        for row in tqdm(
            rows,
            desc="Build QA train",
        ):

            gold = row["state"]

            # positives
            for slot, value in (
                gold.items()
            ):

                span = (
                    find_last_char_span(
                        row["context"],
                        value,
                    )
                )

                if span is None:
                    continue

                self.samples.append({
                    "context":
                        row["context"],
                    "slot":
                        slot,
                    "value":
                        value,
                    "answer":
                        span,
                })

            # negative slots
            negatives = [
                s
                for s in slots
                if s not in gold
            ]

            rng.shuffle(negatives)

            n_neg = min(
                len(negatives),
                max(
                    1,
                    negative_ratio
                    * max(
                        1,
                        len(gold),
                    ),
                ),
            )

            for slot in (
                negatives[:n_neg]
            ):

                self.samples.append({
                    "context":
                        row["context"],
                    "slot":
                        slot,
                    "value":
                        None,
                    "answer":
                        None,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x = self.samples[idx]

        question = (
            "Find the value for "
            f"slot {x['slot']}."
        )

        enc = self.tokenizer(
            question,
            x["context"],
            truncation="only_second",
            max_length=
                self.max_length,
            padding=False,
            return_offsets_mapping=
                True,
        )

        if x["answer"] is None:

            start_pos = 0
            end_pos = 0

        else:

            aligned = (
                align_char_to_tokens(
                    enc,
                    x["answer"][0],
                    x["answer"][1],
                )
            )

            if aligned is None:
                start_pos = 0
                end_pos = 0
            else:
                start_pos, end_pos = (
                    aligned
                )

        enc.pop(
            "offset_mapping"
        )

        enc[
            "start_positions"
        ] = start_pos

        enc[
            "end_positions"
        ] = end_pos

        return enc


class QACollator:

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(self, items):

        labels_start = [
            x.pop(
                "start_positions"
            )
            for x in items
        ]

        labels_end = [
            x.pop(
                "end_positions"
            )
            for x in items
        ]

        batch = (
            self.tokenizer.pad(
                items,
                padding=True,
                return_tensors="pt",
            )
        )

        batch[
            "start_positions"
        ] = torch.tensor(
            labels_start,
            dtype=torch.long,
        )

        batch[
            "end_positions"
        ] = torch.tensor(
            labels_end,
            dtype=torch.long,
        )

        return batch


# ============================================================
# BERT/DEBERTA TRAIN
# ============================================================

def train_encoder(args):

    slots = load_json(
        DATA_DIR / "slots.json"
    )

    rows = read_jsonl(
        DATA_DIR / "train.jsonl"
    )

    random.Random(
        SEED
    ).shuffle(rows)

    if args.max_train:
        rows = rows[
            :args.max_train
        ]

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.encoder_model,
            use_fast=True,
        )
    )

    model = (
        AutoModelForQuestionAnswering
        .from_pretrained(
            args.encoder_model
        )
    )

    train_ds = SlotQATrainDataset(
        rows,
        tokenizer,
        slots,
        max_length=
            args.encoder_max_length,
        negative_ratio=
            args.encoder_negative_ratio,
    )

    print(
        "QA training samples:",
        len(train_ds),
    )

    training_args = (
        TrainingArguments(
            output_dir=
                args.encoder_output,

            num_train_epochs=
                args.encoder_epochs,

            per_device_train_batch_size=
                args.encoder_batch,

            gradient_accumulation_steps=
                args.encoder_grad_accum,

            learning_rate=
                args.encoder_lr,

            warmup_ratio=0.05,

            weight_decay=0.01,

            logging_steps=50,

            save_strategy="epoch",

            bf16=(
                torch.cuda.is_available()
                and torch.cuda
                    .is_bf16_supported()
            ),

            fp16=(
                torch.cuda.is_available()
                and not torch.cuda
                    .is_bf16_supported()
            ),

            report_to="none",

            remove_unused_columns=
                False,

            dataloader_num_workers=2,
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=
            QACollator(tokenizer),
    )

    trainer.train()

    trainer.save_model(
        args.encoder_output
    )

    tokenizer.save_pretrained(
        args.encoder_output
    )


# ============================================================
# QA INFERENCE
# ============================================================

@torch.inference_mode()
def qa_score_batch(
    model,
    tokenizer,
    slots,
    contexts,
    max_length=512,
    max_answer_tokens=12,
):
    """Score a batch of (slot, context) QA pairs."""
    if not slots:
        return []

    questions = [
        f"Find the value for slot {slot}."
        for slot in slots
    ]

    enc = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        padding=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offsets_batch = enc["offset_mapping"].tolist()
    seq_ids_batch = [
        enc.sequence_ids(i)
        for i in range(len(slots))
    ]

    model_inputs = {
        k: v.to(model.device)
        for k, v in enc.items()
        if k != "offset_mapping"
    }

    out = model(**model_inputs)
    start_batch = out.start_logits.float().cpu()
    end_batch = out.end_logits.float().cpu()

    results = []

    for b in range(len(slots)):
        offsets = offsets_batch[b]
        seq_ids = seq_ids_batch[b]
        start_logits = start_batch[b]
        end_logits = end_batch[b]

        # CLS is token 0 for these encoders.
        null_score = (
            start_logits[0].item()
            + end_logits[0].item()
        )

        context_tokens = [
            i
            for i, seq in enumerate(seq_ids)
            if seq == 1
            and offsets[i][1] > offsets[i][0]
        ]

        if not context_tokens:
            results.append({
                "value": "",
                "margin": -1e9,
                "best_score": -1e9,
                "null_score": null_score,
            })
            continue

        k = min(25, len(context_tokens))

        masked_start = torch.full_like(
            start_logits,
            -1e9,
        )
        masked_end = torch.full_like(
            end_logits,
            -1e9,
        )

        for i in context_tokens:
            masked_start[i] = start_logits[i]
            masked_end[i] = end_logits[i]

        top_starts = torch.topk(
            masked_start,
            k=k,
        ).indices.tolist()

        top_ends = torch.topk(
            masked_end,
            k=k,
        ).indices.tolist()

        best_score = -1e30
        best_span = None

        for s in top_starts:
            for e in top_ends:
                if e < s:
                    continue
                if e - s + 1 > max_answer_tokens:
                    continue
                if seq_ids[s] != 1 or seq_ids[e] != 1:
                    continue

                score = (
                    start_logits[s].item()
                    + end_logits[e].item()
                )

                if score > best_score:
                    best_score = score
                    best_span = (s, e)

        if best_span is None:
            results.append({
                "value": "",
                "margin": -1e9,
                "best_score": best_score,
                "null_score": null_score,
            })
            continue

        s, e = best_span
        char_start = offsets[s][0]
        char_end = offsets[e][1]

        value = contexts[b][
            char_start:char_end
        ].strip()

        margin = best_score - null_score

        results.append({
            "value": normalize_text(value),
            "margin": margin,
            "best_score": best_score,
            "null_score": null_score,
        })

    return results


@torch.inference_mode()
def qa_score_slot(
    model,
    tokenizer,
    slot,
    context,
    max_length=512,
    max_answer_tokens=12,
):
    return qa_score_batch(
        model,
        tokenizer,
        [slot],
        [context],
        max_length=max_length,
        max_answer_tokens=max_answer_tokens,
    )[0]


def encoder_raw_predictions(
    model,
    tokenizer,
    rows,
    slots,
    args,
):
    """
    Batched inference over all (dialogue example, slot) pairs.
    The returned shape stays identical to the original code:
        list[dict[slot] -> score_result]
    """
    raw = [dict() for _ in rows]

    flat = []
    for row_idx, row in enumerate(rows):
        for slot in slots:
            flat.append((
                row_idx,
                slot,
                row["context"],
            ))

    batch_size = max(1, args.eval_batch_size)

    for start in tqdm(
        range(0, len(flat), batch_size),
        desc="Encoder QA batches",
    ):
        batch = flat[start:start + batch_size]

        batch_slots = [x[1] for x in batch]
        batch_contexts = [x[2] for x in batch]

        results = qa_score_batch(
            model,
            tokenizer,
            batch_slots,
            batch_contexts,
            max_length=args.encoder_max_length,
            max_answer_tokens=args.max_answer_tokens,
        )

        for (row_idx, slot, _), result in zip(
            batch,
            results,
        ):
            raw[row_idx][slot] = result

    return raw


def apply_encoder_threshold(
    raw_predictions,
    threshold,
):
    predictions = []

    for row in raw_predictions:

        state = {}

        for slot, result in (
            row.items()
        ):

            if (
                result["margin"]
                > threshold
                and result["value"]
            ):
                state[slot] = (
                    result["value"]
                )

        predictions.append(
            state
        )

    return predictions


# ============================================================
# THRESHOLD TUNING
# ============================================================

def tune_threshold(
    raw_predictions,
    examples,
):
    margins = []

    for row in raw_predictions:
        for x in row.values():
            if np.isfinite(
                x["margin"]
            ):
                margins.append(
                    x["margin"]
                )

    if not margins:
        return 0.0

    margins = np.array(
        margins
    )

    # candidates across actual score distribution
    quantiles = np.linspace(
        0.01,
        0.99,
        100,
    )

    candidates = np.unique(
        np.quantile(
            margins,
            quantiles,
        )
    )

    candidates = np.concatenate([
        candidates,
        np.array([
            -10,
            -5,
            -2,
            -1,
            0,
            1,
            2,
            5,
            10,
        ]),
    ])

    best_t = 0
    best_f1 = -1

    for threshold in candidates:

        preds = (
            apply_encoder_threshold(
                raw_predictions,
                threshold,
            )
        )

        score = score_states(
            preds,
            examples,
        )

        if score["f1"] > best_f1:

            best_f1 = score["f1"]
            best_t = float(
                threshold
            )

    print(
        "\nBest validation threshold:",
        best_t,
    )

    print(
        "Validation pair F1:",
        best_f1,
    )

    return best_t


# ============================================================
# BERT/DEBERTA EVALUATION
# ============================================================

def run_encoder_eval(args):

    slots = load_json(
        DATA_DIR / "slots.json"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.encoder_output,
            use_fast=True,
        )
    )

    model = (
        AutoModelForQuestionAnswering
        .from_pretrained(
            args.encoder_output
        )
    )

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Tune threshold on validation.
    # --------------------------------------------------------

    val_rows = read_jsonl(
        DATA_DIR
        / (
            "validation_hard.jsonl"
            if args.hard
            else "validation.jsonl"
        )
    )

    if args.max_val:
        val_rows = val_rows[
            :args.max_val
        ]

    print(
        "\nComputing validation "
        "slot margins..."
    )

    raw_val = (
        encoder_raw_predictions(
            model,
            tokenizer,
            val_rows,
            slots,
            args,
        )
    )

    threshold = tune_threshold(
        raw_val,
        val_rows,
    )

    dump_json(
        {
            "threshold":
                threshold
        },
        RESULT_DIR
        / "encoder_threshold.json",
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_rows = read_jsonl(
        DATA_DIR
        / (
            "test_hard.jsonl"
            if args.hard
            else "test.jsonl"
        )
    )

    if args.max_eval:
        test_rows = test_rows[
            :args.max_eval
        ]

    raw_test = (
        encoder_raw_predictions(
            model,
            tokenizer,
            test_rows,
            slots,
            args,
        )
    )

    predictions = (
        apply_encoder_threshold(
            raw_test,
            threshold,
        )
    )

    details = []

    for x, pred, raw in zip(
        test_rows,
        predictions,
        raw_test,
    ):

        details.append({
            "dialogue_id":
                x["dialogue_id"],
            "turn_id":
                x["turn_id"],
            "gold":
                x["state"],
            "pred":
                pred,
            "slot_scores":
                raw,
        })

    write_jsonl(
        details,
        RESULT_DIR
        / "encoder_details.jsonl",
    )

    df = print_results(
        "DEBERTA_FT",
        predictions,
        test_rows,
    )

    df.to_csv(
        RESULT_DIR
        / "encoder_metrics.csv",
        index=False,
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

def compare_results(args):

    files = [
        RESULT_DIR
        / "llm_zero_metrics.csv",

        RESULT_DIR
        / "llm_sft_metrics.csv",

        RESULT_DIR
        / "encoder_metrics.csv",
    ]

    frames = []

    for path in files:

        if path.exists():
            frames.append(
                pd.read_csv(path)
            )

    if not frames:

        print(
            "No result CSV files found."
        )

        return

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    order = [
        "overall",
        "has_explicit",
        "has_implicit",
        "hop0",
        "hop1",
        "hop2",
        "hop3plus",
        "single_domain_dialogue",
        "multi_domain_dialogue",
    ]

    df[
        "group_order"
    ] = (
        df["group"]
        .apply(
            lambda x:
                order.index(x)
                if x in order
                else 999
        )
    )

    df = df.sort_values([
        "group_order",
        "model",
    ])

    df = df.drop(
        columns=[
            "group_order"
        ]
    )

    print("\n")
    print("=" * 110)
    print("FINAL COMPARISON")
    print("=" * 110)

    print(
        df[
            [
                "model",
                "group",
                "n",
                "precision",
                "recall",
                "f1",
                "joint_em",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    df.to_csv(
        RESULT_DIR
        / "comparison.csv",
        index=False,
    )

    # overall only
    overall = (
        df[
            df["group"]
            == "overall"
        ]
        .sort_values(
            "f1",
            ascending=False,
        )
    )

    print("\n")
    print("=" * 90)
    print("OVERALL RANKING")
    print("=" * 90)

    print(
        overall[
            [
                "model",
                "f1",
                "joint_em",
                "precision",
                "recall",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_data(args):

    path = (
        DATA_DIR
        / (
            "test_hard.jsonl"
            if args.hard
            else "test.jsonl"
        )
    )

    rows = read_jsonl(path)

    print(
        "examples:",
        len(rows),
    )

    hop_counter = defaultdict(int)
    domain_counter = defaultdict(int)

    for x in rows:

        h = x["max_hop"]

        if h >= 3:
            hop_counter[
                "3+"
            ] += 1
        else:
            hop_counter[
                str(h)
            ] += 1

        domain_counter[
            str(
                x[
                    "state_domain_count"
                ]
            )
        ] += 1

    print(
        "hop:",
        dict(hop_counter),
    )

    print(
        "active domains:",
        dict(domain_counter),
    )

    print(
        "\nEXAMPLES"
    )

    for x in rows[:5]:

        print(
            "\n" + "=" * 100
        )

        print(
            x["context"]
        )

        print(
            "\nGOLD:"
        )

        print(
            json.dumps(
                x["state"],
                indent=2,
                ensure_ascii=False,
            )
        )

        print(
            "max_hop:",
            x["max_hop"],
        )


# ============================================================
# MAIN
# ============================================================

def get_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "prepare",
            "inspect",
            "llm_zero",
            "llm_train",
            "llm_eval",
            "encoder_train",
            "encoder_eval",
            "compare",
        ],
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    p.add_argument(
        "--max-history-turns",
        type=int,
        default=12,
    )

    p.add_argument(
        "--hard-min-hop",
        type=int,
        default=2,
    )

    p.add_argument(
        "--hard",
        action="store_true",
        help="Use hard multi-turn subset.",
    )

    p.add_argument(
        "--max-train",
        type=int,
        default=None,
    )

    p.add_argument(
        "--max-val",
        type=int,
        default=100,
        help="Validation examples used for encoder threshold tuning (default: 100).",
    )

    p.add_argument(
        "--max-eval",
        type=int,
        default=100,
        help="Maximum number of test examples to evaluate (default: 100). Use 0 for all.",
    )

    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=100,
        help="Inference batch size for LLM and encoder evaluation (default: 100).",
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    p.add_argument(
        "--llm-model",
        default=DEFAULT_LLM,
    )

    p.add_argument(
        "--llm-output",
        default=
            "./qwen_multiturn_ner_lora",
    )

    p.add_argument(
        "--llm-4bit",
        action="store_true",
    )

    p.add_argument(
        "--llm-max-length",
        type=int,
        default=2048,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    p.add_argument(
        "--llm-epochs",
        type=float,
        default=1,
    )

    p.add_argument(
        "--llm-batch",
        type=int,
        default=100,
    )

    p.add_argument(
        "--llm-grad-accum",
        type=int,
        default=16,
    )

    p.add_argument(
        "--llm-lr",
        type=float,
        default=2e-4,
    )

    p.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )

    p.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    p.add_argument(
        "--encoder-model",
        default=
            DEFAULT_ENCODER,
    )

    p.add_argument(
        "--encoder-output",
        default=
            "./deberta_multiturn_ner",
    )

    p.add_argument(
        "--encoder-max-length",
        type=int,
        default=512,
    )

    p.add_argument(
        "--encoder-negative-ratio",
        type=int,
        default=3,
    )

    p.add_argument(
        "--encoder-epochs",
        type=float,
        default=1,
    )

    p.add_argument(
        "--encoder-batch",
        type=int,
        default=100,
    )

    p.add_argument(
        "--encoder-grad-accum",
        type=int,
        default=1,
    )

    p.add_argument(
        "--encoder-lr",
        type=float,
        default=2e-5,
    )

    p.add_argument(
        "--max-answer-tokens",
        type=int,
        default=100,
    )

    return p.parse_args()


def main():

    args = get_args()

    # 0 means no limit / evaluate the full split.
    if args.max_eval == 0:
        args.max_eval = None
    if args.max_val == 0:
        args.max_val = None

    seed_everything(SEED)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.mode == "prepare":
        prepare_data(args)

    elif args.mode == "inspect":
        inspect_data(args)

    elif args.mode == "llm_zero":
        run_llm_zero(args)

    elif args.mode == "llm_train":
        train_llm(args)

    elif args.mode == "llm_eval":
        run_llm_sft_eval(args)

    elif args.mode == "encoder_train":
        train_encoder(args)

    elif args.mode == "encoder_eval":
        run_encoder_eval(args)

    elif args.mode == "compare":
        compare_results(args)


if __name__ == "__main__":
    main()
