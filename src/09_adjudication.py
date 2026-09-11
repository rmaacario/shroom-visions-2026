"""Can an LVLM mark hallucinated spans directly, rather than feeding features?

This is the approach that won the text-only edition of the task: show a model
the evidence and the response, ask which parts are unsupported, map the answer
back to character offsets. Our first attempt at it failed, but with a prompt
that asked for JSON and a parser that discarded anything malformed, so it was
never really tested.

Probe on 100 labelled training examples before committing to the full test set.
Scored with the official metric, so the number is directly comparable to the
0.42 our supervised tagger reaches.

    export WEC_API_KEY=...
    python day9_judge_probe.py                # 100 examples, ~7 min
    python day9_judge_probe.py 300            # more, if the first looks close
"""
import base64
import io
import json
import os
import pathlib
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kit" / "participant_kit"))
from scorer import score_cor, score_cor_lbl, score_iou  # noqa: E402

DISTRIB = ROOT / "data" / "distrib"
IMAGES = pathlib.Path.home() / "Downloads" / "shroom-vis-images"
BASE = "https://inference.wiline.com/v1/chat/completions"
MODEL = "gemma4"
KEY = os.environ.get("WEC_API_KEY")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
WORKERS = 4

CATEGORIES = ["invention", "mischaracterization", "OCR", "miscounting", "other"]

# Quoting substrings is far more robust than asking for character offsets, which
# models cannot count reliably. We match the quotes back ourselves.
PROMPT = """You are checking an AI-generated answer against the image it describes.

QUESTION: {question}

ANSWER: {response}

Some parts of the answer may not be supported by the image. List ONLY those parts.

Rules:
- Quote each unsupported part EXACTLY as it appears in the answer, character for character.
- Quote the shortest span that carries the error, not the whole sentence.
- Give each one a category: invention (something not in the image),
  mischaracterization (something described wrongly), OCR (text misread),
  miscounting (wrong quantity), other.
- If everything in the answer is supported by the image, output NONE.

Format, one per line, nothing else:
<exact quote> ||| <category>"""

def encode_image(name, max_side=1024):
    from PIL import Image
    img = Image.open(IMAGES / name).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def ask(row):
    body = json.dumps({
        "model": MODEL, "max_tokens": 400, "think": False,
        "num_ctx": 8192, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(question=row["prompt"],
                                                   response=row["response"])},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(row['image_name'])}"}}]}],
    }).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                BASE, data=body,
                headers={"Authorization": f"Bearer {KEY}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return row["id"], (json.load(r)["choices"][0]["message"]["content"] or "")
        except Exception as exc:
            if attempt == 2:
                return row["id"], ""
            time.sleep(2 ** attempt)

def parse(raw, response):
    """Map quoted substrings back to character offsets.

    Tries the exact quote first, then a whitespace-normalized match, since
    models routinely alter spacing when quoting.
    """
    spans, used = [], []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*0123456789. ")
        if not line or line.upper().startswith("NONE"):
            continue
        quote, _, cat = line.partition("|||")
        quote = quote.strip().strip('"“”')
        cat = cat.strip().lower()
        cat = next((c for c in CATEGORIES if c.lower() in cat), "mischaracterization")
        if len(quote) < 2:
            continue
        start = response.find(quote)
        if start < 0:
            flat = re.sub(r"\s+", " ", quote)
            m = re.search(re.escape(flat).replace(r"\ ", r"\s+"), response)
            if not m:
                continue
            start, end = m.start(), m.end()
        else:
            end = start + len(quote)
        if any(start < e and s < end for s, e in used):   # skip overlaps
            continue
        used.append((start, end))
        spans.append({"start": start, "end": end, "prob": 1.0, "label": cat})
    return sorted(spans, key=lambda s: s["start"])

if __name__ == "__main__":
    if not KEY:
        raise SystemExit("set WEC_API_KEY first")
    rows = [json.loads(l) for l in open(
        DISTRIB / "shroom-vision.train.en.labeled.jsonl", encoding="utf-8")][:N]
    print(f"{len(rows)} examples, model={MODEL}, {WORKERS} workers")

    raws, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(ask, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            rid, raw = fut.result()
            raws[rid] = raw
            if i % 20 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(rows)}  {rate:.2f}/s  "
                      f"eta {(len(rows)-i)/rate/60:.0f} min", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")

    refs, preds, empty, unmatched = [], [], 0, 0
    for row in rows:
        raw = raws.get(row["id"], "")
        if not raw.strip():
            empty += 1
        spans = parse(raw, row["response"])
        if raw.strip() and not spans and "NONE" not in raw.upper():
            unmatched += 1
        refs.append({"id": row["id"], "labels": row["labels"],
                     "text_len": len(row["response"])})
        preds.append({"id": row["id"], "labels": spans})

    cor = np.mean([score_cor(r, p) for r, p in zip(refs, preds)])
    cor_lbl = np.mean([score_cor_lbl(r, p) for r, p in zip(refs, preds)])
    iou = np.mean([score_iou(r, p) for r, p in zip(refs, preds)])
    mark_none = np.mean([score_cor(r, {"id": r["id"], "labels": []}) for r in refs])

    print(f"\n{'Cor':<10}{cor:.3f}")
    print(f"{'Cor_lbl':<10}{cor_lbl:.3f}")
    print(f"{'IoU':<10}{iou:.3f}")
    print(f"{'mark_none':<10}{mark_none:.3f}   (floor)")
    print(f"\nsupervised tagger reaches 0.34 (en test), 0.42 (it test)")
    print(f"empty replies: {empty}/{len(rows)}   "
          f"replies whose quotes matched nothing: {unmatched}")

    for row in rows[:3]:
        print("=" * 74)
        print("RAW  :", raws.get(row["id"], "")[:250].replace("\n", " | "))
        print("PRED :", [row["response"][s["start"]:s["end"]]
                         for s in parse(raws.get(row["id"], ""), row["response"])])
        print("GOLD :", [row["response"][s["start"]:s["end"]] for s in row["labels"]])
