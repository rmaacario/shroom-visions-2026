"""Verify each response sentence against its caption, and test whether the signal exists."""

import json, os, re, time, gc
import numpy as np
import torch

CAPTION_PATH = f"{OUT_DIR}/captions_gemma4.json"
NLI_DIM = 7
CHUNK_CHARS = 400

CAPTIONS = json.load(open(CAPTION_PATH, encoding="utf-8"))
CAPTIONS = {k: v for k, v in CAPTIONS.items() if v}
print(f"{len(CAPTIONS)} captions available")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

NLI_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
SIM_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

nli_tok = AutoTokenizer.from_pretrained(NLI_ID)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_ID).to(DEVICE).eval()
sim_model = SentenceTransformer(SIM_ID, device=DEVICE)

LABELS = {nli_model.config.id2label[i].lower(): i
          for i in range(nli_model.config.num_labels)}
I_ENT = LABELS.get("entailment", 0)
I_CON = LABELS.get("contradiction", 2)
print("label map:", LABELS)

SENT_END = re.compile(r"[.!?。！？\n]+")

def split_sentences(text):
    """Sentence spans with character offsets, for Latin and CJK punctuation."""
    spans, start = [], 0
    for m in SENT_END.finditer(text):
        end = m.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans

def chunk_caption(caption):
    """Paragraph-ish chunks, so the question-answering tail isn't truncated away."""
    parts, buf = [], ""
    for line in caption.split("\n"):
        if len(buf) + len(line) > CHUNK_CHARS and buf.strip():
            parts.append(buf.strip()); buf = line
        else:
            buf += "\n" + line
    if buf.strip():
        parts.append(buf.strip())
    return parts or [caption[:CHUNK_CHARS]]

# Excluded from faithfulness scoring in the white paper: conversational filler,
# markdown artifacts and bare links are not verifiable claims.
FILLER = re.compile(
    r"^\s*(here('s| is)|sure|certainly|of course|i hope|let me know|in summary|"
    r"voici|bien s[uû]r|certamente|ecco|当然|希望|总之)", re.I)

def is_filler(s):
    s = s.strip()
    if len(s.split()) < 5 and len(s) < 30:
        return True
    if FILLER.match(s) or re.fullmatch(r"[\s*#\-_=|`~\[\]()]*", s):
        return True
    return s.lower().startswith(("http", "source:", "!["))

@torch.no_grad()
def nli_probs(premises, hypotheses, batch_size=64):
    out = []
    for i in range(0, len(premises), batch_size):
        enc = nli_tok(premises[i:i + batch_size], hypotheses[i:i + batch_size],
                      return_tensors="pt", truncation=True, max_length=256,
                      padding=True).to(DEVICE)
        out.append(torch.softmax(nli_model(**enc).logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3))

def extract_nli(split, lang, limit=None):
    rows = load_rows(split, lang)
    if limit:
        rows = rows[:limit]
    rows = [r for r in rows if r["image_name"] in CAPTIONS]
    print(f"{split}/{lang}: {len(rows)} rows with captions")
    store, t0 = {}, time.time()
    for i, row in enumerate(rows):
        resp = row["response"]
        chunks = chunk_caption(CAPTIONS[row["image_name"]])
        spans = split_sentences(resp)
        if not spans:
            continue
        sents = [resp[a:b].strip() for a, b in spans]
        # every sentence against every chunk
        prem = [c for _ in sents for c in chunks]
        hyp = [s for s in sents for _ in chunks]
        p = nli_probs(prem, hyp).reshape(len(sents), len(chunks), -1)
        emb = sim_model.encode(chunks + sents, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
        cos = emb[len(chunks):] @ emb[:len(chunks)].T
        feats = np.zeros((len(spans), NLI_DIM), dtype=np.float32)
        for j, s in enumerate(sents):
            feats[j, 0] = p[j, :, I_ENT].max()     # best entailment over chunks
            feats[j, 1] = p[j, :, I_ENT].mean()
            feats[j, 2] = p[j, :, I_CON].max()     # any chunk contradicting it
            feats[j, 3] = cos[j].max()
            feats[j, 4] = float(is_filler(s))
            feats[j, 5] = j / max(len(sents) - 1, 1)
            feats[j, 6] = min(len(s) / 200.0, 1.0)
        store[row["id"] + "|off"] = np.array(spans, dtype=np.int32)
        store[row["id"] + "|f"] = feats
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(rows)} {rate:.1f}/s "
                  f"eta {(len(rows) - i - 1) / rate / 60:.0f} min", flush=True)
    path = f"{OUT_DIR}/nli_feats_{split}_{lang}.npz"
    np.savez_compressed(path, **store)
    print(f"{split}/{lang}: {len(store) // 2} saved, "
          f"{(time.time() - t0) / 60:.1f} min -> {path}", flush=True)

extract_nli("train", "en")
gc.collect(); torch.cuda.empty_cache()

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

d = np.load(f"{OUT_DIR}/nli_feats_train_en.npz")
gold_rows = {json.loads(l)["id"]: json.loads(l)
             for l in open(f"{DISTRIB}/shroom-vision.train.en.labeled.jsonl")}

ys, xs, groups = [], [], []
for key in [k for k in d.files if k.endswith("|f")]:
    rid = key[:-2]
    row, offs, feats = gold_rows[rid], d[rid + "|off"], d[key]
    gold = np.zeros(len(row["response"]))
    for s in row["labels"]:
        seg = gold[s["start"]:s["end"]]
        gold[s["start"]:s["end"]] = np.maximum(seg, s["prob"])
    for (a, b), f in zip(offs, feats):
        ys.append(gold[a:b].max() > 0 if b > a else False)
        xs.append(f); groups.append(rid)

ys, xs, groups = np.array(ys), np.array(xs), np.array(groups)
names = ["entail_max", "entail_mean", "contradict_max", "cosine_max",
         "filler", "position", "length"]
print(f"\n{ys.sum()} hallucinated / {len(ys)} sentences "
      f"({100 * ys.mean():.1f}%), {len(set(groups))} responses\n")
print(f"{'feature':<16}{'AUC':>8}")
for i, n in enumerate(names):
    a = roc_auc_score(ys, xs[:, i])
    print(f"{n:<16}{max(a, 1 - a):>8.3f}")

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
              .split(xs, ys, groups))
mu, sd = xs[tr].mean(0), xs[tr].std(0) + 1e-2
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit((xs[tr] - mu) / sd, ys[tr])
print(f"\nlogistic probe, grouped by response: "
      f"AUC {roc_auc_score(ys[te], clf.predict_proba((xs[te] - mu) / sd)[:, 1]):.3f}")
print("this is SENTENCE-level, so not directly comparable to the token-level "
      "0.636 from the visual features")
