# train_ner_compare.py

import argparse
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from transformers import AutoTokenizer, AutoModel


# ============================================================
# Utils
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_dataset_stats(ds, label_names):
    print("\n" + "=" * 72)
    print("DATA EXPLORATION")
    print("=" * 72)

    for split in ds:
        print(f"{split:12s}: {len(ds[split]):,} sentences")

    counter = Counter()

    lengths = []

    for row in ds["train"]:
        counter.update(row["ner_tags"])
        lengths.append(len(row["tokens"]))

    print("\nNER tags:")
    for i, name in enumerate(label_names):
        print(f"{i:2d}: {name}")

    print("\nTrain tag distribution:")
    for idx, count in counter.most_common():
        print(f"{label_names[idx]:10s}: {count:,}")

    print("\nSentence lengths:")
    print(f"mean: {np.mean(lengths):.2f}")
    print(f"p95 : {np.percentile(lengths, 95):.1f}")
    print(f"max : {max(lengths)}")

    print("\nExamples:")

    for row in ds["train"].select(range(3)):
        print("\nSentence:")
        print(" ".join(row["tokens"]))

        ents = []
        for tok, tag_id in zip(row["tokens"], row["ner_tags"]):
            tag = label_names[tag_id]
            if tag != "O":
                ents.append(f"{tok}/{tag}")

        print("Entities:")
        print(" ".join(ents) if ents else "(none)")


# ============================================================
# LSTM dataset
# ============================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(train_split, max_vocab=30000):
    counter = Counter()

    for row in train_split:
        counter.update(tok.lower() for tok in row["tokens"])

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, _ in counter.most_common(max_vocab - 2):
        vocab[token] = len(vocab)

    return vocab


class LSTMDataset(Dataset):
    def __init__(self, split, vocab):
        self.data = split
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        ids = [
            self.vocab.get(tok.lower(), self.vocab[UNK_TOKEN])
            for tok in row["tokens"]
        ]

        labels = row["ner_tags"]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "tokens": row["tokens"],
        }


def collate_lstm(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []
    masks = []

    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n

        input_ids.append(
            torch.cat([
                x["input_ids"],
                torch.zeros(pad, dtype=torch.long),
            ])
        )

        labels.append(
            torch.cat([
                x["labels"],
                torch.full((pad,), -100, dtype=torch.long),
            ])
        )

        masks.append(
            torch.cat([
                torch.ones(n, dtype=torch.bool),
                torch.zeros(pad, dtype=torch.bool),
            ])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "mask": torch.stack(masks),
    }


# ============================================================
# BiLSTM
# ============================================================

class BiLSTMNER(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        emb_dim=128,
        hidden_dim=256,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_labels,
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        logits = self.classifier(x)

        return logits


# ============================================================
# CRF
# ============================================================

try:
    from torchcrf import CRF
except ImportError:
    CRF = None


class BiLSTMCRFNER(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        emb_dim=128,
        hidden_dim=256,
    ):
        super().__init__()

        if CRF is None:
            raise RuntimeError("pip install torchcrf")

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_labels,
        )

        self.crf = CRF(
            num_labels,
            batch_first=True,
        )

    def emissions(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)

        return self.classifier(x)

    def loss(self, input_ids, labels, mask):
        emissions = self.emissions(input_ids)

        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0

        return -self.crf(
            emissions,
            safe_labels,
            mask=mask,
            reduction="mean",
        )

    def decode(self, input_ids, mask):
        emissions = self.emissions(input_ids)

        return self.crf.decode(
            emissions,
            mask=mask,
        )


# ============================================================
# BERT dataset
# ============================================================

class BERTDataset(Dataset):
    def __init__(
        self,
        split,
        tokenizer,
        max_length=128,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        row = self.split[idx]

        encoding = self.tokenizer(
            row["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )

        word_ids = encoding.word_ids()

        labels = []
        previous_word = None

        for word_id in word_ids:

            if word_id is None:
                labels.append(-100)

            elif word_id != previous_word:
                labels.append(row["ner_tags"][word_id])

            else:
                # only evaluate first subword
                labels.append(-100)

            previous_word = word_id

        return {
            "input_ids": torch.tensor(
                encoding["input_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                encoding["attention_mask"],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


def collate_bert(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    ids = []
    masks = []
    labels = []

    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n

        ids.append(
            torch.cat([
                x["input_ids"],
                torch.full(
                    (pad,),
                    pad_token_id,
                    dtype=torch.long,
                ),
            ])
        )

        masks.append(
            torch.cat([
                x["attention_mask"],
                torch.zeros(pad, dtype=torch.long),
            ])
        )

        labels.append(
            torch.cat([
                x["labels"],
                torch.full(
                    (pad,),
                    -100,
                    dtype=torch.long,
                ),
            ])
        )

    return {
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(masks),
        "labels": torch.stack(labels),
    }


# ============================================================
# BERT
# ============================================================

class BERTNER(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            num_labels,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = self.dropout(
            outputs.last_hidden_state
        )

        return self.classifier(hidden)


# ============================================================
# BERT + CRF
# ============================================================

class BERTCRFNER(nn.Module):
    def __init__(
        self,
        model_name,
        num_labels,
    ):
        super().__init__()

        if CRF is None:
            raise RuntimeError("pip install torchcrf")

        self.encoder = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,
            num_labels,
        )

        self.crf = CRF(
            num_labels,
            batch_first=True,
        )

    def emissions(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = self.dropout(
            outputs.last_hidden_state
        )

        return self.classifier(hidden)

    def loss(
        self,
        input_ids,
        attention_mask,
        labels,
    ):
        emissions = self.emissions(
            input_ids,
            attention_mask,
        )

        # CRF cannot use -100 labels.
        valid_mask = labels != -100

        # create compact sequences containing only
        # first subword positions
        all_emissions = []
        all_labels = []

        for i in range(labels.size(0)):
            idx = valid_mask[i]

            all_emissions.append(
                emissions[i][idx]
            )
            all_labels.append(
                labels[i][idx]
            )

        lengths = [
            len(x)
            for x in all_labels
        ]

        max_len = max(lengths)

        padded_e = emissions.new_zeros(
            len(all_emissions),
            max_len,
            emissions.size(-1),
        )

        padded_l = labels.new_zeros(
            len(all_labels),
            max_len,
        )

        mask = torch.zeros(
            len(all_labels),
            max_len,
            dtype=torch.bool,
            device=labels.device,
        )

        for i, (e, l) in enumerate(
            zip(all_emissions, all_labels)
        ):
            n = len(l)

            padded_e[i, :n] = e
            padded_l[i, :n] = l
            mask[i, :n] = True

        loss = -self.crf(
            padded_e,
            padded_l,
            mask=mask,
            reduction="mean",
        )

        return loss

    def decode(
        self,
        input_ids,
        attention_mask,
        labels,
    ):
        emissions = self.emissions(
            input_ids,
            attention_mask,
        )

        valid_mask = labels != -100

        sequences = []

        for i in range(labels.size(0)):
            sequences.append(
                emissions[i][valid_mask[i]]
            )

        max_len = max(
            len(x)
            for x in sequences
        )

        padded = emissions.new_zeros(
            len(sequences),
            max_len,
            emissions.size(-1),
        )

        mask = torch.zeros(
            len(sequences),
            max_len,
            dtype=torch.bool,
            device=emissions.device,
        )

        for i, seq in enumerate(sequences):
            n = len(seq)
            padded[i, :n] = seq
            mask[i, :n] = True

        return self.crf.decode(
            padded,
            mask=mask,
        )


# ============================================================
# Evaluation
# ============================================================

def compute_metrics(
    all_true,
    all_pred,
):
    return {
        "precision": precision_score(
            all_true,
            all_pred,
        ),
        "recall": recall_score(
            all_true,
            all_pred,
        ),
        "f1": f1_score(
            all_true,
            all_pred,
        ),
    }


@torch.no_grad()
def evaluate_lstm(
    model,
    loader,
    label_names,
    device,
    use_crf=False,
):
    model.eval()

    true_sequences = []
    pred_sequences = []

    for batch in loader:

        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        if use_crf:

            preds = model.decode(
                ids,
                mask,
            )

            for i, pred in enumerate(preds):

                n = int(mask[i].sum())

                gold = labels[i, :n].tolist()

                true_sequences.append(
                    [label_names[x] for x in gold]
                )

                pred_sequences.append(
                    [label_names[x] for x in pred]
                )

        else:

            logits = model(ids)
            preds = logits.argmax(-1)

            for i in range(ids.size(0)):

                valid = labels[i] != -100

                gold = labels[i][valid].tolist()
                pred = preds[i][valid].tolist()

                true_sequences.append(
                    [label_names[x] for x in gold]
                )

                pred_sequences.append(
                    [label_names[x] for x in pred]
                )

    return (
        compute_metrics(
            true_sequences,
            pred_sequences,
        ),
        true_sequences,
        pred_sequences,
    )


@torch.no_grad()
def evaluate_bert(
    model,
    loader,
    label_names,
    device,
    use_crf=False,
):
    model.eval()

    true_sequences = []
    pred_sequences = []

    for batch in loader:

        ids = batch["input_ids"].to(device)

        mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch["labels"].to(device)

        if use_crf:

            predictions = model.decode(
                ids,
                mask,
                labels,
            )

            for i, pred in enumerate(predictions):

                gold = labels[i][
                    labels[i] != -100
                ].tolist()

                true_sequences.append(
                    [label_names[x] for x in gold]
                )

                pred_sequences.append(
                    [label_names[x] for x in pred]
                )

        else:

            logits = model(
                ids,
                mask,
            )

            predictions = logits.argmax(-1)

            for i in range(ids.size(0)):

                valid = labels[i] != -100

                gold = labels[i][valid].tolist()

                pred = predictions[i][
                    valid
                ].tolist()

                true_sequences.append(
                    [label_names[x] for x in gold]
                )

                pred_sequences.append(
                    [label_names[x] for x in pred]
                )

    return (
        compute_metrics(
            true_sequences,
            pred_sequences,
        ),
        true_sequences,
        pred_sequences,
    )


# ============================================================
# Training
# ============================================================

def train_lstm(
    model,
    train_loader,
    val_loader,
    label_names,
    device,
    epochs,
    lr,
    use_crf=False,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    model.to(device)

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0

        start = time.time()

        for batch in train_loader:

            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            if use_crf:

                loss = model.loss(
                    ids,
                    labels,
                    mask,
                )

            else:

                logits = model(ids)

                loss = criterion(
                    logits.reshape(
                        -1,
                        logits.size(-1),
                    ),
                    labels.reshape(-1),
                )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        metrics, _, _ = evaluate_lstm(
            model,
            val_loader,
            label_names,
            device,
            use_crf,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={total_loss / len(train_loader):.4f} | "
            f"val_P={metrics['precision']:.4f} | "
            f"val_R={metrics['recall']:.4f} | "
            f"val_F1={metrics['f1']:.4f} | "
            f"time={time.time()-start:.1f}s"
        )


def train_bert(
    model,
    train_loader,
    val_loader,
    label_names,
    device,
    epochs,
    lr,
    use_crf=False,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    model.to(device)

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0

        start = time.time()

        for batch in train_loader:

            ids = batch["input_ids"].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_crf:

                loss = model.loss(
                    ids,
                    attention_mask,
                    labels,
                )

            else:

                logits = model(
                    ids,
                    attention_mask,
                )

                loss = criterion(
                    logits.reshape(
                        -1,
                        logits.size(-1),
                    ),
                    labels.reshape(-1),
                )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        metrics, _, _ = evaluate_bert(
            model,
            val_loader,
            label_names,
            device,
            use_crf,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={total_loss / len(train_loader):.4f} | "
            f"val_P={metrics['precision']:.4f} | "
            f"val_R={metrics['recall']:.4f} | "
            f"val_F1={metrics['f1']:.4f} | "
            f"time={time.time()-start:.1f}s"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--val-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--test-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--bert-model",
        default="bert-base-cased",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    seed_everything()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("Loading CoNLL-2003...")

    ds = load_dataset(
        "eriktks/conll2003"
    )

    label_names = (
        ds["train"]
        .features["ner_tags"]
        .feature.names
    )

    print_dataset_stats(
        ds,
        label_names,
    )

    train_split = ds["train"].select(
        range(
            min(
                args.train_samples,
                len(ds["train"]),
            )
        )
    )

    val_split = ds["validation"].select(
        range(
            min(
                args.val_samples,
                len(ds["validation"]),
            )
        )
    )

    test_split = ds["test"].select(
        range(
            min(
                args.test_samples,
                len(ds["test"]),
            )
        )
    )

    num_labels = len(label_names)

    results = {}

    # --------------------------------------------------------
    # LSTM data
    # --------------------------------------------------------

    vocab = build_vocab(
        train_split
    )

    print("\nVocabulary:", len(vocab))

    train_lstm_ds = LSTMDataset(
        train_split,
        vocab,
    )

    val_lstm_ds = LSTMDataset(
        val_split,
        vocab,
    )

    test_lstm_ds = LSTMDataset(
        test_split,
        vocab,
    )

    train_lstm_loader = DataLoader(
        train_lstm_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_lstm,
    )

    val_lstm_loader = DataLoader(
        val_lstm_ds,
        batch_size=args.batch_size,
        collate_fn=collate_lstm,
    )

    test_lstm_loader = DataLoader(
        test_lstm_ds,
        batch_size=args.batch_size,
        collate_fn=collate_lstm,
    )

    # --------------------------------------------------------
    # 1. BiLSTM
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("BiLSTM")
    print("=" * 72)

    model = BiLSTMNER(
        len(vocab),
        num_labels,
    )

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )

    train_lstm(
        model,
        train_lstm_loader,
        val_lstm_loader,
        label_names,
        device,
        args.epochs,
        lr=1e-3,
    )

    metrics, y_true, y_pred = evaluate_lstm(
        model,
        test_lstm_loader,
        label_names,
        device,
    )

    results["BiLSTM"] = metrics

    print("\nTEST")
    print(classification_report(y_true, y_pred))

    del model
    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 2. BiLSTM + CRF
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("BiLSTM + CRF")
    print("=" * 72)

    model = BiLSTMCRFNER(
        len(vocab),
        num_labels,
    )

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )

    train_lstm(
        model,
        train_lstm_loader,
        val_lstm_loader,
        label_names,
        device,
        args.epochs,
        lr=1e-3,
        use_crf=True,
    )

    metrics, y_true, y_pred = evaluate_lstm(
        model,
        test_lstm_loader,
        label_names,
        device,
        use_crf=True,
    )

    results["BiLSTM+CRF"] = metrics

    print("\nTEST")
    print(classification_report(y_true, y_pred))

    del model
    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # BERT data
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_model
    )

    train_bert_ds = BERTDataset(
        train_split,
        tokenizer,
        args.max_length,
    )

    val_bert_ds = BERTDataset(
        val_split,
        tokenizer,
        args.max_length,
    )

    test_bert_ds = BERTDataset(
        test_split,
        tokenizer,
        args.max_length,
    )

    collate = lambda x: collate_bert(
        x,
        tokenizer.pad_token_id,
    )

    train_bert_loader = DataLoader(
        train_bert_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    val_bert_loader = DataLoader(
        val_bert_ds,
        batch_size=args.batch_size,
        collate_fn=collate,
    )

    test_bert_loader = DataLoader(
        test_bert_ds,
        batch_size=args.batch_size,
        collate_fn=collate,
    )

    # --------------------------------------------------------
    # 3. BERT
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("BERT")
    print("=" * 72)

    model = BERTNER(
        args.bert_model,
        num_labels,
    )

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )

    train_bert(
        model,
        train_bert_loader,
        val_bert_loader,
        label_names,
        device,
        args.epochs,
        lr=2e-5,
    )

    metrics, y_true, y_pred = evaluate_bert(
        model,
        test_bert_loader,
        label_names,
        device,
    )

    results["BERT"] = metrics

    print("\nTEST")
    print(classification_report(y_true, y_pred))

    del model
    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 4. BERT + CRF
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("BERT + CRF")
    print("=" * 72)

    model = BERTCRFNER(
        args.bert_model,
        num_labels,
    )

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )

    train_bert(
        model,
        train_bert_loader,
        val_bert_loader,
        label_names,
        device,
        args.epochs,
        lr=2e-5,
        use_crf=True,
    )

    metrics, y_true, y_pred = evaluate_bert(
        model,
        test_bert_loader,
        label_names,
        device,
        use_crf=True,
    )

    results["BERT+CRF"] = metrics

    print("\nTEST")
    print(classification_report(y_true, y_pred))

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)

    print(
        f"{'Model':20s} "
        f"{'Precision':>10s} "
        f"{'Recall':>10s} "
        f"{'F1':>10s}"
    )

    print("-" * 55)

    for name, m in results.items():
        print(
            f"{name:20s} "
            f"{m['precision']:10.4f} "
            f"{m['recall']:10.4f} "
            f"{m['f1']:10.4f}"
        )


if __name__ == "__main__":
    main()
