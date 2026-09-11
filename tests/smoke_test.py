"""Smoke test: run the span-tagger pipeline end to end on CPU with a tiny random model.

Verifies every tensor shape, the collate function, the token gather, both losses,
inference, span construction and scoring. It does not verify that the model
learns anything — only that nothing crashes and the plumbing is sound.
"""
import json
import pathlib
import sys
import types

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DISTRIB = ROOT.parent / "shroom-visions" / "data" / "distrib"

# The tagger script trains on import, so cut it off above the training block.
source = (SRC / "01_span_tagger.py").read_text()
head, _, _ = source.partition("train_rows, dev_rows = [], []")
source = head.replace('DISTRIB    = "/kaggle/working/distrib"', f'DISTRIB    = "{DISTRIB}"')

mod = types.ModuleType("day2")
mod.__dict__["__file__"] = str(SRC / "01_span_tagger.py")
exec(compile(source, "01_span_tagger.py", "exec"), mod.__dict__)

from transformers import XLMRobertaConfig, XLMRobertaModel  # noqa: E402

def tiny_encoder(self):
    cfg = XLMRobertaConfig(
        vocab_size=mod.tokenizer.vocab_size, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
        max_position_embeddings=512,
    )
    return XLMRobertaModel(cfg)

# swap the pretrained encoder for a small random one
_orig_init = mod.SpanTagger.__init__

def patched_init(self):
    torch.nn.Module.__init__(self)
    self.encoder = tiny_encoder(self)
    h = self.encoder.config.hidden_size
    self.dropout = torch.nn.Dropout(0.1)
    self.span_head = torch.nn.Linear(h, 1)
    self.cat_head = torch.nn.Linear(h, len(mod.CATEGORIES))

mod.SpanTagger.__init__ = patched_init

print("1. loading data")
train_rows, dev_rows = [], []
for lang in mod.LANGS:
    tr, dv = mod.split_dev(mod.load(lang, "train"))
    train_rows += tr[:40]
    dev_rows += dv[:40]
print(f"   train={len(train_rows)} dev={len(dev_rows)} (subsampled)")

print("2. dataset + collate")
from torch.utils.data import DataLoader  # noqa: E402
loader = DataLoader(mod.SpanData(train_rows), batch_size=8, shuffle=True,
                    collate_fn=mod.collate)
batch = next(iter(loader))
for k, v in batch.items():
    print(f"   {k:<15} {tuple(v.shape)} {v.dtype}")

print("3. forward + backward")
model = mod.SpanTagger()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
from transformers import get_linear_schedule_with_warmup  # noqa: E402
sched = get_linear_schedule_with_warmup(opt, 1, len(loader))
loss = mod.run_epoch(model, loader, opt, sched)
print(f"   loss = {loss:.4f}")
assert np.isfinite(loss), "loss is not finite"

print("4. inference -> char probabilities")
cps = mod.predict_char_probs(model, dev_rows)
assert len(cps) == len(dev_rows), "row count mismatch after inference"
for (cp, cc), row in zip(cps, dev_rows):
    assert len(cp) == len(row["response"]), "char array does not match response length"
    assert cc.max() < len(mod.CATEGORIES) and cc.min() >= 0, "category index out of range"
print(f"   {len(cps)} rows, prob range [{min(c.min() for c, _ in cps):.3f}, "
      f"{max(c.max() for c, _ in cps):.3f}]")

print("5. spans + threshold tuning + scoring")
thr, cor, scores = mod.tune_threshold(dev_rows, cps)
print(f"   best threshold {thr:.2f} -> {scores}")

print("6. submission format")
preds = mod.build_preds(dev_rows, cps, thr)
out = ROOT / "_smoke_predictions.jsonl"
with open(out, "w", encoding="utf-8") as fh:
    for p in preds:
        fh.write(json.dumps(p, ensure_ascii=False) + "\n")

sys.path.insert(0, str(ROOT.parent / "shroom-visions" / "kit" / "participant_kit"))
from format_checker import check_submission_files  # noqa: E402

# the checker wants one language per file; split the smoke output accordingly
ok = True
for lang in mod.LANGS:
    rows = [p for p, r in zip(preds, dev_rows) if r["language"] == lang]
    path = ROOT / f"_smoke_{lang}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for p in rows:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    errors, _, stats = check_submission_files([path])
    if errors:
        ok = False
        print(f"   {lang}: FAILED")
        for e in errors[:5]:
            print("     ", e)
    else:
        print(f"   {lang}: OK ({stats['rows']} rows, {stats['spans']} spans)")

print("\nSMOKE TEST PASSED" if ok else "\nSMOKE TEST FAILED")
sys.exit(0 if ok else 1)
