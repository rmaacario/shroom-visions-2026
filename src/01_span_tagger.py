"""Text-only span tagger: XLM-R token classification to per-character probability.

Writes dev scores and predictions_{lang}.jsonl. Kaggle T4, ~20 min end to end.
Expects the data at DISTRIB below.
"""

import json, os, random, pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from scipy.stats import spearmanr

DISTRIB    = "/kaggle/working/distrib"
OUT_DIR    = "/kaggle/working"
MODEL_ID   = "xlm-roberta-base"
LANGS      = ["en", "fr", "it", "zh"]
CATEGORIES = ["invention", "mischaracterization", "OCR", "miscounting", "other"]
MAX_LEN    = 256
BATCH      = 16
EPOCHS     = 3
LR         = 2e-5
SEED       = 13

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# Copied verbatim from the organizers' scorer.py so dev numbers match the
# leaderboard exactly. Do not edit.

def score_cor(ref_dict, pred_dict, label_filtered_=None):
    assert ref_dict['id'] == pred_dict['id']
    ref_vec = [0.] * ref_dict['text_len']
    pred_vec = [0.] * ref_dict['text_len']
    ref_labels = (ref_dict['labels'] if label_filtered_ is None
                  else [s for s in ref_dict['labels'] if s['label'] == label_filtered_])
    pred_labels = (pred_dict['labels'] if label_filtered_ is None
                   else [s for s in pred_dict['labels'] if s['label'] == label_filtered_])
    for span in ref_labels:
        for idx in range(span['start'], span['end']):
            ref_vec[idx] += span['prob']
    for span in pred_labels:
        for idx in range(span['start'], span['end']):
            pred_vec[idx] = span['prob']
    ref_cmps = {round(f, 8) for f in ref_vec}
    pred_cmps = {round(f, 8) for f in pred_vec}
    if len(pred_cmps) == 1 or len(ref_cmps) == 1:
        if len(pred_cmps) != len(ref_cmps):
            return 0.0
        if ref_cmps == {0.0}:
            return float(pred_cmps == {0.0})
        return float(pred_cmps != {0.0})
    return spearmanr(ref_vec, pred_vec).correlation

def score_cor_lbl(ref_dict, pred_dict):
    all_labels = {s['label'] for d in [ref_dict, pred_dict] for s in d['labels']}
    if all_labels:
        return sum(score_cor(ref_dict, pred_dict, label_filtered_=l)
                   for l in all_labels) / len(all_labels)
    return 1.0

def score_iou(ref_dict, pred_dict):
    assert ref_dict['id'] == pred_dict['id']
    ref_i = {i for s in ref_dict['labels'] for i in range(s['start'], s['end'])}
    pred_i = {i for s in pred_dict['labels'] for i in range(s['start'], s['end'])}
    if not pred_i and not ref_i:
        return 1.
    return len(ref_i & pred_i) / len(ref_i | pred_i)

def evaluate(refs, preds):
    refs = sorted(refs, key=lambda r: r['id'])
    preds = sorted(preds, key=lambda r: r['id'])
    assert [r['id'] for r in refs] == [p['id'] for p in preds]
    return {
        'Cor':     float(np.mean([score_cor(r, p)     for r, p in zip(refs, preds)])),
        'Cor_lbl': float(np.mean([score_cor_lbl(r, p) for r, p in zip(refs, preds)])),
        'IoU':     float(np.mean([score_iou(r, p)     for r, p in zip(refs, preds)])),
    }

def load(lang, split_name):
    kind = "labeled" if split_name == "train" else "unlabeled"
    path = f"{DISTRIB}/shroom-vision.{split_name}.{lang}.{kind}.jsonl"
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]

def split_dev(rows, fraction=0.2, seed=SEED):
    """Stratified split preserving the clean / hallucinated ratio."""
    clean = [r for r in rows if not r["labels"]]
    dirty = [r for r in rows if r["labels"]]
    rng = random.Random(seed)
    rng.shuffle(clean); rng.shuffle(dirty)
    nc, nd = int(len(clean) * fraction), int(len(dirty) * fraction)
    dev = clean[:nc] + dirty[:nd]
    train = clean[nc:] + dirty[nd:]
    rng.shuffle(dev); rng.shuffle(train)
    return train, dev

def char_targets(row):
    n = len(row["response"])
    prob = np.zeros(n, dtype=np.float32)
    cat = np.full(n, -1, dtype=np.int64)
    for span in row.get("labels", []):
        for i in range(span["start"], min(span["end"], n)):
            if span["prob"] >= prob[i]:
                prob[i] = span["prob"]
                cat[i] = CATEGORIES.index(span["label"])
    return prob, cat

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def encode(row):
    """Tokenize response (segment 0) with the prompt as context (segment 1)."""
    enc = tokenizer(row["response"], text_pair=row["prompt"],
                    return_offsets_mapping=True, truncation="only_first",
                    max_length=MAX_LEN)
    seq_ids = enc.sequence_ids()
    keep = [i for i, (a, b) in enumerate(enc["offset_mapping"])
            if b > a and seq_ids[i] == 0]
    return enc, keep

class SpanData(Dataset):
    def __init__(self, rows, labeled=True):
        self.rows, self.labeled = rows, labeled
        # Tokenize once up front. Doing it per access pins the CPU at 100% and
        # starves the GPU, since the loader runs in the main process.
        self.cache = [encode(r) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        enc, keep = self.cache[i]
        item = {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "keep": torch.tensor(keep, dtype=torch.long),
        }
        if self.labeled:
            prob, cat = char_targets(row)
            offs = [enc["offset_mapping"][i] for i in keep]
            # per token: max character probability, and the category at the peak
            tp = [float(prob[a:b].max()) if b > a else 0.0 for a, b in offs]
            tc = []
            for a, b in offs:
                seg = cat[a:b]
                seg = seg[seg >= 0]
                tc.append(int(np.bincount(seg).argmax()) if len(seg) else -100)
            item["tok_prob"] = torch.tensor(tp, dtype=torch.float)
            item["tok_cat"] = torch.tensor(tc, dtype=torch.long)
        return item

def collate(batch):
    pad = tokenizer.pad_token_id
    maxlen = max(len(b["input_ids"]) for b in batch)
    maxkeep = max(len(b["keep"]) for b in batch)
    out = {
        "input_ids": torch.full((len(batch), maxlen), pad, dtype=torch.long),
        "attention_mask": torch.zeros((len(batch), maxlen), dtype=torch.long),
        "keep": torch.zeros((len(batch), maxkeep), dtype=torch.long),
        "keep_mask": torch.zeros((len(batch), maxkeep), dtype=torch.bool),
    }
    has_labels = "tok_prob" in batch[0]
    if has_labels:
        out["tok_prob"] = torch.zeros((len(batch), maxkeep))
        out["tok_cat"] = torch.full((len(batch), maxkeep), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L, K = len(b["input_ids"]), len(b["keep"])
        out["input_ids"][i, :L] = b["input_ids"]
        out["attention_mask"][i, :L] = b["attention_mask"]
        out["keep"][i, :K] = b["keep"]
        out["keep_mask"][i, :K] = True
        if has_labels:
            out["tok_prob"][i, :K] = b["tok_prob"]
            out["tok_cat"][i, :K] = b["tok_cat"]
    return out

class SpanTagger(nn.Module):
    """One shared encoder, two token-level heads.

    span head  -> is this token hallucinated        (drives Cor)
    cat  head  -> which of the five categories      (drives Cor_lbl only)
    """

    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_ID)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.span_head = nn.Linear(h, 1)
        self.cat_head = nn.Linear(h, len(CATEGORIES))

    def forward(self, input_ids, attention_mask, keep, keep_mask):
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        idx = keep.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        tok = self.dropout(torch.gather(hidden, 1, idx))
        return self.span_head(tok).squeeze(-1), self.cat_head(tok)

def run_epoch(model, loader, optimizer=None, scheduler=None, log_every=50):
    train = optimizer is not None
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    total, nb = 0.0, 0
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            span_logit, cat_logit = model(batch["input_ids"], batch["attention_mask"],
                                          batch["keep"], batch["keep_mask"])
            m = batch["keep_mask"]
            # soft targets: BCE against the annotator probability itself
            l_span = (bce(span_logit, batch["tok_prob"]) * m).sum() / m.sum().clamp(min=1)
            l_cat = ce(cat_logit.reshape(-1, len(CATEGORIES)), batch["tok_cat"].reshape(-1))
            loss = l_span + 0.5 * l_cat
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        total += loss.item(); nb += 1
        if train and step % log_every == 0:
            print(f"  step {step}/{len(loader)}  loss {total / nb:.4f}", flush=True)
    return total / max(nb, 1)

@torch.no_grad()
def predict_char_probs(model, rows):
    """Return per-character probability and category arrays for each row."""
    model.eval()
    dataset = SpanData(rows, labeled=False)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, collate_fn=collate)
    results, cursor = [], 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        span_logit, cat_logit = model(batch["input_ids"], batch["attention_mask"],
                                      batch["keep"], batch["keep_mask"])
        probs = torch.sigmoid(span_logit).cpu().numpy()
        cats = cat_logit.argmax(-1).cpu().numpy()
        for i in range(len(probs)):
            row = rows[cursor]
            enc, keep = dataset.cache[cursor]
            cursor += 1
            offs = [enc["offset_mapping"][k] for k in keep]
            n = len(row["response"])
            cp = np.zeros(n, dtype=np.float32)
            cc = np.zeros(n, dtype=np.int64)
            for j, (a, b) in enumerate(offs):
                cp[a:min(b, n)] = probs[i][j]
                cc[a:min(b, n)] = cats[i][j]
            results.append((cp, cc))
    return results

def to_spans(char_prob, char_cat, threshold):
    """Contiguous above-threshold runs, split where the category changes."""
    spans, n, i = [], len(char_prob), 0
    while i < n:
        if char_prob[i] < threshold:
            i += 1; continue
        j, c = i, char_cat[i]
        while j < n and char_prob[j] >= threshold and char_cat[j] == c:
            j += 1
        spans.append({"start": int(i), "end": int(j),
                      "prob": float(round(float(np.mean(char_prob[i:j])), 6)),
                      "label": CATEGORIES[int(c)]})
        i = j
    return spans

def build_preds(rows, char_preds, threshold):
    return [{"id": r["id"], "labels": to_spans(cp, cc, threshold)}
            for r, (cp, cc) in zip(rows, char_preds)]

def tune_threshold(dev_rows, char_preds):
    """Pick the threshold maximizing Cor on dev. Cor is the reported metric."""
    refs = [{"id": r["id"], "labels": r["labels"], "text_len": len(r["response"])}
            for r in dev_rows]
    best = (None, -1, None)
    for t in np.arange(0.10, 0.91, 0.05):
        s = evaluate(refs, build_preds(dev_rows, char_preds, float(t)))
        if s["Cor"] > best[1]:
            best = (float(t), s["Cor"], s)
    return best

train_rows, dev_rows = [], []
for lang in LANGS:
    tr, dv = split_dev(load(lang, "train"))
    train_rows += tr; dev_rows += dv
print(f"train={len(train_rows)}  dev={len(dev_rows)}")

model = SpanTagger().to(DEVICE)
train_loader = DataLoader(SpanData(train_rows), batch_size=BATCH, shuffle=True,
                          collate_fn=collate)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(0.1 * len(train_loader) * EPOCHS), len(train_loader) * EPOCHS)

for epoch in range(EPOCHS):
    loss = run_epoch(model, train_loader, optimizer, scheduler)
    print(f"epoch {epoch + 1}: loss {loss:.4f}", flush=True)
    # checkpoint every epoch so a disconnect costs one epoch, not the whole run
    torch.save(model.state_dict(), f"{OUT_DIR}/span_tagger.pt")

print(f"\n{'lang':<6}{'thr':>6}{'Cor':>9}{'Cor_lbl':>9}{'IoU':>9}   (mark_none Cor)")
thresholds = {}
for lang in LANGS:
    rows = [r for r in dev_rows if r["language"] == lang]
    cps = predict_char_probs(model, rows)
    thr, cor, scores = tune_threshold(rows, cps)
    thresholds[lang] = thr
    refs = [{"id": r["id"], "labels": r["labels"], "text_len": len(r["response"])}
            for r in rows]
    floor = evaluate(refs, [{"id": r["id"], "labels": []} for r in rows])["Cor"]
    print(f"{lang:<6}{thr:>6.2f}{scores['Cor']:>9.3f}{scores['Cor_lbl']:>9.3f}"
          f"{scores['IoU']:>9.3f}   {floor:.3f}")

for lang in LANGS:
    rows = load(lang, "test")
    cps = predict_char_probs(model, rows)
    preds = build_preds(rows, cps, thresholds[lang])
    path = f"{OUT_DIR}/predictions_{lang}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_spans = sum(len(p["labels"]) for p in preds)
    n_empty = sum(1 for p in preds if not p["labels"])
    print(f"{lang}: {len(preds)} rows, {n_spans} spans, {n_empty} empty -> {path}")
