import os, re, json, random, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)

SEED = 42
DATA_DIR = Path('./multiturn_ner_data')
RESULT_DIR = Path('./multiturn_ner_results')
DEFAULT_MODEL = 'microsoft/deberta-v3-base'

IGNORE_VALUES = {'', 'none', 'not mentioned', 'notmentioned', 'null'}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(x):
    if x is None:
        return ''
    x = str(x).lower().strip()
    x = x.replace('’', "'").replace('‘', "'")
    x = x.replace('“', '"').replace('”', '"')
    x = re.sub(r'\s+', ' ', x)
    return x


def normalize_slot(x):
    return normalize_text(x)


def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_slots():
    with open(DATA_DIR / 'slots.json', encoding='utf-8') as f:
        return json.load(f)


def pair_set(state):
    return {(normalize_slot(k), normalize_text(v)) for k, v in state.items()}


def score_states(predictions, examples):
    tp = fp = fn = 0
    em = []
    for pred, ex in zip(predictions, examples):
        p = pair_set(pred)
        g = pair_set(ex['state'])
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
        em.append(int(p == g))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'n': len(examples),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'joint_em': float(np.mean(em)) if em else 0.0,
    }


def evaluate_groups(predictions, examples):
    groups = {
        'overall': list(range(len(examples))),
        'has_explicit': [],
        'has_implicit': [],
        'hop0': [], 'hop1': [], 'hop2': [], 'hop3plus': [],
        'single_domain_dialogue': [],
        'multi_domain_dialogue': [],
    }
    for i, x in enumerate(examples):
        metas = x.get('slot_meta', {}).values()
        if any(m.get('explicit', False) for m in metas):
            groups['has_explicit'].append(i)
        if x.get('has_implicit', False):
            groups['has_implicit'].append(i)
        h = x.get('max_hop', 0)
        if h == 0: groups['hop0'].append(i)
        elif h == 1: groups['hop1'].append(i)
        elif h == 2: groups['hop2'].append(i)
        else: groups['hop3plus'].append(i)
        if x.get('dialogue_multi_domain', False):
            groups['multi_domain_dialogue'].append(i)
        else:
            groups['single_domain_dialogue'].append(i)

    out = {}
    for name, idxs in groups.items():
        if idxs:
            out[name] = score_states([predictions[i] for i in idxs], [examples[i] for i in idxs])
    return out


def print_results(model_name, predictions, examples):
    rows = []
    for group, r in evaluate_groups(predictions, examples).items():
        rows.append({'model': model_name, 'group': group, **r})
    df = pd.DataFrame(rows)
    print('\n' + '=' * 100)
    print(model_name)
    print('=' * 100)
    print(df[['group','n','precision','recall','f1','joint_em']].to_string(
        index=False, float_format=lambda x: f'{x:.4f}'))
    return df


def find_last_char_span(context, value):
    c = context.lower()
    v = str(value).lower().strip()
    if not v:
        return None
    s = c.rfind(v)
    return None if s < 0 else (s, s + len(v))


def cls_index_from_ids(input_ids, tokenizer):
    cls_id = tokenizer.cls_token_id
    if cls_id is not None and cls_id in input_ids:
        return input_ids.index(cls_id)
    # DeBERTa normally has CLS at 0; fallback only.
    return 0


def align_span(enc, start_char, end_char):
    offsets = enc['offset_mapping']
    seq_ids = enc.sequence_ids()
    start_tok = end_tok = None
    for i, (off, sid) in enumerate(zip(offsets, seq_ids)):
        if sid != 1:
            continue
        a, b = off
        if start_tok is None and a <= start_char < b:
            start_tok = i
        if a < end_char <= b:
            end_tok = i
    if start_tok is None or end_tok is None:
        return None
    return start_tok, end_tok


class StableSlotQADataset(torch.utils.data.Dataset):
    """
    Important fixes vs previous version:
      1) positive span that is truncated is DROPPED, never relabeled as no-answer.
      2) negatives use the actual CLS token index.
      3) examples are pre-tokenized once so train-time labels are fixed and auditable.
    """
    def __init__(self, rows, tokenizer, slots, max_length=512, negative_ratio=1.0):
        self.items = []
        self.stats = defaultdict(int)
        rng = random.Random(SEED)

        for row in tqdm(rows, desc='Build stable QA train'):
            gold = row['state']

            # Positive slot-value examples.
            for slot, value in gold.items():
                self.stats['positive_total'] += 1
                span = find_last_char_span(row['context'], value)
                if span is None:
                    self.stats['positive_no_string_match'] += 1
                    continue

                q = f'Find the value for slot {slot}.'
                enc = tokenizer(
                    q,
                    row['context'],
                    truncation='only_second',
                    max_length=max_length,
                    padding=False,
                    return_offsets_mapping=True,
                )
                aligned = align_span(enc, span[0], span[1])
                if aligned is None:
                    # Critical: do not turn a truncated positive into a negative.
                    self.stats['positive_truncated'] += 1
                    continue

                start_pos, end_pos = aligned
                enc.pop('offset_mapping')
                enc['start_positions'] = start_pos
                enc['end_positions'] = end_pos
                self.items.append(enc)
                self.stats['positive_kept'] += 1

            # Sample no-answer slots.
            negatives = [s for s in slots if s not in gold]
            rng.shuffle(negatives)
            n_neg = min(len(negatives), int(round(negative_ratio * max(1, len(gold)))))
            for slot in negatives[:n_neg]:
                q = f'Find the value for slot {slot}.'
                enc = tokenizer(
                    q,
                    row['context'],
                    truncation='only_second',
                    max_length=max_length,
                    padding=False,
                    return_offsets_mapping=False,
                )
                cls_idx = cls_index_from_ids(enc['input_ids'], tokenizer)
                enc['start_positions'] = cls_idx
                enc['end_positions'] = cls_idx
                self.items.append(enc)
                self.stats['negative_kept'] += 1

        print('\nTRAIN DATA AUDIT')
        for k in sorted(self.stats):
            print(f'{k:28s}: {self.stats[k]}')
        total_pos = self.stats['positive_total']
        if total_pos:
            print(f"positive token-window coverage: {self.stats['positive_kept']/total_pos:.4f}")
        print(f'total QA train items         : {len(self.items)}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return dict(self.items[idx])


class QACollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items):
        items = [dict(x) for x in items]
        starts = [x.pop('start_positions') for x in items]
        ends = [x.pop('end_positions') for x in items]
        batch = self.tokenizer.pad(items, padding=True, return_tensors='pt')
        batch['start_positions'] = torch.tensor(starts, dtype=torch.long)
        batch['end_positions'] = torch.tensor(ends, dtype=torch.long)
        return batch


def train(args):
    seed_everything()
    slots = load_slots()
    rows = read_jsonl(DATA_DIR / 'train.jsonl')
    random.Random(SEED).shuffle(rows)
    if args.max_train > 0:
        rows = rows[:args.max_train]

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.encoder_model)

    ds = StableSlotQADataset(
        rows, tokenizer, slots,
        max_length=args.max_length,
        negative_ratio=args.negative_ratio,
    )

    use_bf16 = args.precision == 'bf16'
    use_fp16 = args.precision == 'fp16'

    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=20,
        save_strategy='epoch',
        bf16=use_bf16,
        fp16=use_fp16,
        report_to='none',
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=QACollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


@torch.inference_mode()
def qa_score_batch(model, tokenizer, slots, contexts, max_length=512, max_answer_tokens=12):
    questions = [f'Find the value for slot {s}.' for s in slots]
    enc = tokenizer(
        questions,
        contexts,
        truncation='only_second',
        max_length=max_length,
        padding=True,
        return_offsets_mapping=True,
        return_tensors='pt',
    )
    offsets_batch = enc['offset_mapping'].tolist()
    seq_ids_batch = [enc.sequence_ids(i) for i in range(len(slots))]
    input_ids_cpu = enc['input_ids'].tolist()
    model_inputs = {k: v.to(model.device) for k, v in enc.items() if k != 'offset_mapping'}
    out = model(**model_inputs)
    start_batch = out.start_logits.float().cpu()
    end_batch = out.end_logits.float().cpu()

    results = []
    for b in range(len(slots)):
        offsets = offsets_batch[b]
        seq_ids = seq_ids_batch[b]
        s_logits = start_batch[b]
        e_logits = end_batch[b]
        cls_idx = cls_index_from_ids(input_ids_cpu[b], tokenizer)
        null_score = s_logits[cls_idx].item() + e_logits[cls_idx].item()

        context_tokens = [i for i, sid in enumerate(seq_ids)
                          if sid == 1 and offsets[i][1] > offsets[i][0]]
        if not context_tokens:
            results.append({'value':'', 'margin':-1e9, 'best_score':-1e9, 'null_score':null_score})
            continue

        k = min(25, len(context_tokens))
        ms = torch.full_like(s_logits, -1e9)
        me = torch.full_like(e_logits, -1e9)
        for i in context_tokens:
            ms[i] = s_logits[i]
            me[i] = e_logits[i]
        top_s = torch.topk(ms, k=k).indices.tolist()
        top_e = torch.topk(me, k=k).indices.tolist()

        best_score = -1e30
        best_span = None
        for s in top_s:
            for e in top_e:
                if e < s or e - s + 1 > max_answer_tokens:
                    continue
                if seq_ids[s] != 1 or seq_ids[e] != 1:
                    continue
                score = s_logits[s].item() + e_logits[e].item()
                if score > best_score:
                    best_score = score
                    best_span = (s, e)

        if best_span is None:
            results.append({'value':'', 'margin':-1e9, 'best_score':best_score, 'null_score':null_score})
            continue

        s, e = best_span
        cs, ce = offsets[s][0], offsets[e][1]
        value = contexts[b][cs:ce].strip()
        results.append({
            'value': normalize_text(value),
            'margin': best_score - null_score,
            'best_score': best_score,
            'null_score': null_score,
        })
    return results


def raw_predictions(model, tokenizer, rows, slots, args):
    raw = [dict() for _ in rows]
    flat = []
    for ri, row in enumerate(rows):
        for slot in slots:
            flat.append((ri, slot, row['context']))

    bs = max(1, args.eval_batch_size)
    for st in tqdm(range(0, len(flat), bs), desc='Encoder QA batches'):
        batch = flat[st:st+bs]
        results = qa_score_batch(
            model, tokenizer,
            [x[1] for x in batch],
            [x[2] for x in batch],
            max_length=args.max_length,
            max_answer_tokens=args.max_answer_tokens,
        )
        for (ri, slot, _), result in zip(batch, results):
            raw[ri][slot] = result
    return raw


def apply_threshold(raw, threshold):
    preds = []
    for row in raw:
        state = {}
        for slot, r in row.items():
            if r['value'] and r['margin'] >= threshold:
                state[slot] = r['value']
        preds.append(state)
    return preds


def tune_threshold(raw, examples):
    margins = [r['margin'] for row in raw for r in row.values()
               if np.isfinite(r['margin']) and r['margin'] > -1e8]
    if not margins:
        raise RuntimeError('No valid encoder margins. Inference produced no valid context spans.')

    arr = np.array(margins, dtype=np.float32)
    qs = np.linspace(0.0, 1.0, 121)
    candidates = np.unique(np.concatenate([
        np.quantile(arr, qs),
        np.array([-20,-10,-5,-2,-1,0,1,2,5,10,20], dtype=np.float32),
    ]))

    best = (-1.0, None)
    for t in candidates:
        s = score_states(apply_threshold(raw, float(t)), examples)
        if s['f1'] > best[0]:
            best = (s['f1'], float(t))
    print(f'Best validation threshold: {best[1]:.6f}')
    print(f'Validation pair F1       : {best[0]:.4f}')
    return best[1]


def token_window_gold_coverage(rows, tokenizer, max_length):
    total = kept = 0
    by_group = defaultdict(lambda: [0,0])
    for row in rows:
        for slot, value in row['state'].items():
            total += 1
            h = row.get('slot_meta', {}).get(slot, {}).get('hop')
            g = 'hop_unknown' if h is None else ('hop3plus' if h >= 3 else f'hop{h}')
            by_group[g][0] += 1
            span = find_last_char_span(row['context'], value)
            if span is None:
                continue
            enc = tokenizer(
                f'Find the value for slot {slot}.', row['context'],
                truncation='only_second', max_length=max_length,
                return_offsets_mapping=True,
            )
            ok = align_span(enc, span[0], span[1]) is not None
            if ok:
                kept += 1
                by_group[g][1] += 1
    print('\nTOKEN-WINDOW ORACLE COVERAGE')
    print(f'overall: {kept}/{total} = {kept/total if total else 0:.4f}')
    for g in sorted(by_group):
        t, k = by_group[g]
        print(f'{g:12s}: {k}/{t} = {k/t if t else 0:.4f}')
    return kept / total if total else 0.0


def evaluate(args):
    seed_everything()
    slots = load_slots()
    tokenizer = AutoTokenizer.from_pretrained(args.output, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.output)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    val_file = 'validation_hard.jsonl' if args.hard else 'validation.jsonl'
    test_file = 'test_hard.jsonl' if args.hard else 'test.jsonl'
    val_rows = read_jsonl(DATA_DIR / val_file)
    test_rows = read_jsonl(DATA_DIR / test_file)
    if args.max_val > 0:
        val_rows = val_rows[:args.max_val]
    if args.max_eval > 0:
        test_rows = test_rows[:args.max_eval]

    token_window_gold_coverage(test_rows, tokenizer, args.max_length)

    print('\nComputing validation margins...')
    raw_val = raw_predictions(model, tokenizer, val_rows, slots, args)
    threshold = tune_threshold(raw_val, val_rows)

    print('\nComputing test predictions...')
    raw_test = raw_predictions(model, tokenizer, test_rows, slots, args)
    preds = apply_threshold(raw_test, threshold)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    details = []
    for x, p, r in zip(test_rows, preds, raw_test):
        details.append({
            'dialogue_id': x['dialogue_id'],
            'turn_id': x['turn_id'],
            'gold': x['state'],
            'pred': p,
            'slot_scores': r,
        })
    write_jsonl(details, RESULT_DIR / 'encoder_fixed_details.jsonl')
    dump_json({'threshold': threshold}, RESULT_DIR / 'encoder_fixed_threshold.json')
    df = print_results('DEBERTA_FT_FIXED', preds, test_rows)
    df.to_csv(RESULT_DIR / 'encoder_fixed_metrics.csv', index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['train','eval'])
    p.add_argument('--encoder-model', default=DEFAULT_MODEL)
    p.add_argument('--output', default='./deberta_multiturn_ner_fixed')
    p.add_argument('--hard', action='store_true')

    p.add_argument('--max-train', type=int, default=1000, help='0 = all')
    p.add_argument('--max-val', type=int, default=100, help='0 = all')
    p.add_argument('--max-eval', type=int, default=100, help='0 = all')

    p.add_argument('--train-batch-size', type=int, default=12)
    p.add_argument('--eval-batch-size', type=int, default=100)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--epochs', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--negative-ratio', type=float, default=1.0)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--max-answer-tokens', type=int, default=12)
    p.add_argument('--precision', choices=['fp32','bf16','fp16'], default='fp32')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)
