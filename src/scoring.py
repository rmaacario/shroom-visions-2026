"""Harness: dev split + local scoring using the official scorer functions.

The official `scorer.py` is imported unmodified; we only skip `scores_classif`,
which needs a `raw_annots` field that the released training data does not carry.
"""
import json
import pathlib
import random
import sys

import numpy as np

KIT = pathlib.Path(__file__).resolve().parents[1] / "kit" / "participant_kit"
DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "distrib"
sys.path.insert(0, str(KIT))
from scorer import score_cor, score_cor_lbl, score_iou  # noqa: E402

LANGS = ["en", "fr", "it", "zh"]
DEV_FRACTION = 0.2
SEED = 13

def load_train(lang):
    path = DATA / f"shroom-vision.train.{lang}.labeled.jsonl"
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]

def split(lang, seed=SEED):
    """Stratified train/dev split keeping the clean/hallucinated ratio intact."""
    rows = load_train(lang)
    clean = [r for r in rows if not r["labels"]]
    dirty = [r for r in rows if r["labels"]]
    rng = random.Random(seed)
    rng.shuffle(clean)
    rng.shuffle(dirty)
    n_clean = int(len(clean) * DEV_FRACTION)
    n_dirty = int(len(dirty) * DEV_FRACTION)
    dev = clean[:n_clean] + dirty[:n_dirty]
    train = clean[n_clean:] + dirty[n_dirty:]
    rng.shuffle(dev)
    rng.shuffle(train)
    return train, dev

def as_refs(rows):
    return [
        {"id": r["id"], "labels": r["labels"], "text_len": len(r["response"])}
        for r in sorted(rows, key=lambda r: r["id"])
    ]

def evaluate(refs, preds):
    """Return the metrics the official scorer writes out, plus IoU for reference."""
    preds = sorted(preds, key=lambda r: r["id"])
    refs = sorted(refs, key=lambda r: r["id"])
    assert [r["id"] for r in refs] == [p["id"] for p in preds], "id mismatch"
    cor = np.array([score_cor(r, p) for r, p in zip(refs, preds)])
    cor_lbl = np.array([score_cor_lbl(r, p) for r, p in zip(refs, preds)])
    iou = np.array([score_iou(r, p) for r, p in zip(refs, preds)])
    return {"Cor": cor.mean(), "Cor_lbl": cor_lbl.mean(), "IoU": iou.mean()}

def pred_mark_none(refs):
    return [{"id": r["id"], "labels": []} for r in refs]

def pred_mark_all(refs, label="mischaracterization"):
    return [
        {"id": r["id"],
         "labels": [{"start": 0, "end": r["text_len"], "prob": 1.0, "label": label}]}
        for r in refs
    ]

def pred_oracle_gate(refs, label="mischaracterization"):
    """Perfect clean/hallucinated decision, then mark the whole response.

    Isolates how much of the score comes from the binary gate alone, with no
    span localization at all.
    """
    out = []
    for r in refs:
        if not r["labels"]:
            out.append({"id": r["id"], "labels": []})
        else:
            out.append({"id": r["id"],
                        "labels": [{"start": 0, "end": r["text_len"],
                                    "prob": 1.0, "label": label}]})
    return out

def pred_oracle_gate_true_label(refs):
    """Oracle gate, and also the correct majority category per response."""
    out = []
    for r in refs:
        if not r["labels"]:
            out.append({"id": r["id"], "labels": []})
            continue
        counts = {}
        for span in r["labels"]:
            counts[span["label"]] = counts.get(span["label"], 0) + (span["end"] - span["start"])
        best = max(counts, key=counts.get)
        out.append({"id": r["id"],
                    "labels": [{"start": 0, "end": r["text_len"],
                                "prob": 1.0, "label": best}]})
    return out

def pred_oracle_spans(refs):
    """Gold spans copied verbatim — the attainable ceiling (sanity check)."""
    return [{"id": r["id"], "labels": r["labels"]} for r in refs]

if __name__ == "__main__":
    systems = {
        "mark_none": pred_mark_none,
        "mark_all": pred_mark_all,
        "oracle_gate": pred_oracle_gate,
        "oracle_gate+label": pred_oracle_gate_true_label,
        "oracle_spans": pred_oracle_spans,
    }
    header = f"{'system':<20}" + "".join(f"{l:>22}" for l in LANGS)
    print(header)
    print("-" * len(header))
    for name, fn in systems.items():
        cells = []
        for lang in LANGS:
            _, dev = split(lang)
            refs = as_refs(dev)
            scores = evaluate(refs, fn(refs))
            cells.append(f"{scores['Cor']:.3f}/{scores['Cor_lbl']:.3f}/{scores['IoU']:.3f}")
        print(f"{name:<20}" + "".join(f"{c:>22}" for c in cells))
    print("\ncells are Cor / Cor_lbl / IoU   (Cor and Cor_lbl are what the scorer reports)")
    for lang in LANGS:
        tr, dev = split(lang)
        print(f"{lang}: train={len(tr)} dev={len(dev)} dev_clean={sum(1 for r in dev if not r['labels'])}")
