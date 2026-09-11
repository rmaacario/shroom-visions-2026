"""XLM-R tagger with the 38 visual features concatenated to its final-layer states.

The two models tokenize differently, so features are routed through characters:
scattered onto the characters each proxy token covers, then averaged over each
XLM-R token's span. Set USE_VIS = False for the text-only ablation row.
"""

VIS_DIM   = 38
VIS_PROJ  = 64          # width the visual features are projected to
USE_VIS   = True        # flip to False for the text-only ablation row

import json, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

print(f"USE_VIS={USE_VIS}")

def load_visual(split, lang):
    """id -> (n_chars, VIS_DIM) array of per-character visual features."""
    path = f"{OUT_DIR}/hidden_feats_{split}_{lang}.npz"
    if not os.path.exists(path):
        print(f"  missing {path}")
        return {}
    d = np.load(path)
    out = {}
    for key in d.files:
        if not key.endswith("|f"):
            continue
        rid = key[:-2]
        offs, feats = d[rid + "|off"], d[key]
        n = int(offs[-1][1]) if len(offs) else 0
        arr = np.zeros((n, VIS_DIM), dtype=np.float32)
        for (a, b), f in zip(offs, feats):
            arr[a:b] = f
        out[rid] = arr
    return out

VISUAL = {}
for split in ["train", "test"]:
    for lang in LANGS:
        VISUAL.update(load_visual(split, lang))
print(f"visual features for {len(VISUAL)} examples")

# Standardize using training statistics only — the six scalars are on wildly
# different scales (cosine ~1, distance ~10) and would otherwise swamp each other.
_sample = np.concatenate([v for k, v in list(VISUAL.items())[:2000]], axis=0)
VIS_MEAN = _sample.mean(axis=0)
VIS_STD = _sample.std(axis=0) + 1e-6
print("visual feature scale:", np.round(VIS_STD[:6], 3))

def visual_for_tokens(rid, offsets, n_chars):
    """Average the per-character visual features over each XLM-R token span."""
    arr = VISUAL.get(rid)
    out = np.zeros((len(offsets), VIS_DIM), dtype=np.float32)
    if arr is None:
        return out, 0.0
    for i, (a, b) in enumerate(offsets):
        b = min(b, len(arr))
        if b > a:
            out[i] = arr[a:b].mean(axis=0)
    return (out - VIS_MEAN) / VIS_STD, 1.0

class FusedData(Dataset):
    def __init__(self, rows, labeled=True):
        self.rows, self.labeled = rows, labeled
        self.cache = [encode(r) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        enc, keep = self.cache[i]
        offs = [enc["offset_mapping"][k] for k in keep]
        vis, has_vis = visual_for_tokens(row["id"], offs, len(row["response"]))
        item = {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "keep": torch.tensor(keep, dtype=torch.long),
            "vis": torch.tensor(vis),
            "has_vis": torch.tensor(has_vis),
        }
        if self.labeled:
            prob, cat = char_targets(row)
            tp = [float(prob[a:b].max()) if b > a else 0.0 for a, b in offs]
            tc = []
            for a, b in offs:
                seg = cat[a:b]; seg = seg[seg >= 0]
                tc.append(int(np.bincount(seg).argmax()) if len(seg) else -100)
            item["tok_prob"] = torch.tensor(tp, dtype=torch.float)
            item["tok_cat"] = torch.tensor(tc, dtype=torch.long)
        return item

def collate_fused(batch):
    pad = tokenizer.pad_token_id
    maxlen = max(len(b["input_ids"]) for b in batch)
    maxkeep = max(len(b["keep"]) for b in batch)
    n = len(batch)
    out = {
        "input_ids": torch.full((n, maxlen), pad, dtype=torch.long),
        "attention_mask": torch.zeros((n, maxlen), dtype=torch.long),
        "keep": torch.zeros((n, maxkeep), dtype=torch.long),
        "keep_mask": torch.zeros((n, maxkeep), dtype=torch.bool),
        "vis": torch.zeros((n, maxkeep, VIS_DIM)),
        "has_vis": torch.stack([b["has_vis"] for b in batch]),
    }
    has_labels = "tok_prob" in batch[0]
    if has_labels:
        out["tok_prob"] = torch.zeros((n, maxkeep))
        out["tok_cat"] = torch.full((n, maxkeep), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L, K = len(b["input_ids"]), len(b["keep"])
        out["input_ids"][i, :L] = b["input_ids"]
        out["attention_mask"][i, :L] = b["attention_mask"]
        out["keep"][i, :K] = b["keep"]
        out["keep_mask"][i, :K] = True
        out["vis"][i, :K] = b["vis"]
        if has_labels:
            out["tok_prob"][i, :K] = b["tok_prob"]
            out["tok_cat"][i, :K] = b["tok_cat"]
    return out

class FusedTagger(nn.Module):
    """XLM-R token states concatenated with projected visual-ablation features."""

    def __init__(self, use_vis=USE_VIS):
        super().__init__()
        self.use_vis = use_vis
        self.encoder = AutoModel.from_pretrained(MODEL_ID)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        if use_vis:
            self.vis_proj = nn.Sequential(
                nn.Linear(VIS_DIM, VIS_PROJ), nn.GELU(), nn.LayerNorm(VIS_PROJ))
            h = h + VIS_PROJ
        self.span_head = nn.Linear(h, 1)
        self.cat_head = nn.Linear(h, len(CATEGORIES))

    def forward(self, input_ids, attention_mask, keep, keep_mask, vis, has_vis):
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        idx = keep.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        tok = torch.gather(hidden, 1, idx)
        if self.use_vis:
            v = self.vis_proj(vis) * has_vis.view(-1, 1, 1)
            tok = torch.cat([tok, v], dim=-1)
        tok = self.dropout(tok)
        return self.span_head(tok).squeeze(-1), self.cat_head(tok)

def run_epoch_fused(model, loader, optimizer=None, scheduler=None, log_every=50):
    train = optimizer is not None
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    total, nb = 0.0, 0
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            span_logit, cat_logit = model(
                batch["input_ids"], batch["attention_mask"], batch["keep"],
                batch["keep_mask"], batch["vis"], batch["has_vis"])
            m = batch["keep_mask"]
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
def predict_fused(model, rows):
    model.eval()
    ds = FusedData(rows, labeled=False)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fused)
    results, cursor = [], 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        span_logit, cat_logit = model(
            batch["input_ids"], batch["attention_mask"], batch["keep"],
            batch["keep_mask"], batch["vis"], batch["has_vis"])
        probs = torch.sigmoid(span_logit).cpu().numpy()
        cats = cat_logit.argmax(-1).cpu().numpy()
        for i in range(len(probs)):
            row = rows[cursor]
            enc, keep = ds.cache[cursor]
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

train_rows, dev_rows = [], []
for lang in LANGS:
    tr, dv = split_dev(load(lang, "train"))
    train_rows += tr; dev_rows += dv
print(f"train={len(train_rows)}  dev={len(dev_rows)}")

model = FusedTagger(USE_VIS).to(DEVICE)
if MODEL_ID.endswith("large"):
    model.encoder.embeddings.requires_grad_(False)

loader = DataLoader(FusedData(train_rows), batch_size=BATCH, shuffle=True,
                    collate_fn=collate_fused)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=LR, weight_decay=0.01, foreach=False)
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(0.1 * len(loader) * EPOCHS), len(loader) * EPOCHS)

for epoch in range(EPOCHS):
    loss = run_epoch_fused(model, loader, optimizer, scheduler)
    print(f"epoch {epoch + 1}: loss {loss:.4f}", flush=True)
    torch.save(model.state_dict(), f"{OUT_DIR}/fused_tagger_vis{int(USE_VIS)}.pt")

print(f"\n{'lang':<6}{'thr':>6}{'Cor':>9}{'Cor_lbl':>9}{'IoU':>9}   (floor)")
thresholds = {}
for lang in LANGS:
    rows = [r for r in dev_rows if r["language"] == lang]
    cps = predict_fused(model, rows)
    thr, cor, scores = tune_threshold(rows, cps)
    thresholds[lang] = thr
    refs = [{"id": r["id"], "labels": r["labels"], "text_len": len(r["response"])}
            for r in rows]
    floor = evaluate(refs, [{"id": r["id"], "labels": []} for r in rows])["Cor"]
    print(f"{lang:<6}{thr:>6.2f}{scores['Cor']:>9.3f}{scores['Cor_lbl']:>9.3f}"
          f"{scores['IoU']:>9.3f}   {floor:.3f}")

for lang in LANGS:
    rows = load(lang, "test")
    cps = predict_fused(model, rows)
    preds = build_preds(rows, cps, thresholds[lang])
    path = f"{OUT_DIR}/predictions_vis_{lang}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{lang}: {len(preds)} rows, {sum(len(p['labels']) for p in preds)} spans "
          f"-> {path}")
