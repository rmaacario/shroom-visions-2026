"""Caption-then-verify features.

The evidence is not retrieved, it is the image, so the image is turned into text
once and each response sentence is verified against that caption with NLI, with
a cosine-similarity fallback when NLI returns neutral. Images are captioned once
and reused across every example referencing them.
"""

import json, os, re, time, gc
import numpy as np
import torch
from PIL import Image

CAPTION_PATH = f"{OUT_DIR}/captions.json"
CAPTION_PROMPT = (
    "Describe this image in detail. State the objects present, their colours and "
    "attributes, how many of each there are, where they are relative to each "
    "other, and transcribe any text visible in the image. Only describe what you "
    "can actually see."
)

def all_image_names():
    names = set()
    for split in ["train", "test"]:
        for lang in LANGS:
            for row in load_rows(split, lang):
                names.add(row["image_name"])
    return sorted(names)

@torch.no_grad()
def caption_images():
    """One caption per unique image. Resumable — reloads whatever is on disk."""
    captions = {}
    if os.path.exists(CAPTION_PATH):
        captions = json.load(open(CAPTION_PATH, encoding="utf-8"))
        print(f"resuming with {len(captions)} captions")
    names = [n for n in all_image_names() if n not in captions]
    print(f"{len(names)} images left to caption")
    t0 = time.time()
    for i, name in enumerate(names):
        try:
            image = Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB")
        except Exception:
            captions[name] = ""
            continue
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(DEVICE)
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        captions[name] = processor.tokenizer.decode(gen, skip_special_tokens=True).strip()
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(names)}  {rate:.1f}/s  "
                  f"eta {(len(names) - i - 1) / rate / 60:.0f} min", flush=True)
            json.dump(captions, open(CAPTION_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False)
    json.dump(captions, open(CAPTION_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(captions)} captions -> {CAPTION_PATH}")
    return captions

CAPTIONS = caption_images()
print("\nexample:", list(CAPTIONS.values())[0][:300])

del model
gc.collect(); torch.cuda.empty_cache()
print(f"GPU free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Captions come out in English while responses are in four languages, so the
# NLI model has to work cross-lingually. XNLI-trained mDeBERTa is built for this.
NLI_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
SIM_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

nli_tok = AutoTokenizer.from_pretrained(NLI_ID)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_ID).to(DEVICE).eval()
sim_model = SentenceTransformer(SIM_ID, device=DEVICE)
NLI_LABELS = [nli_model.config.id2label[i].lower()
              for i in range(nli_model.config.num_labels)]
print("NLI labels:", NLI_LABELS)

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

FILLER = re.compile(
    r"^\s*(here('s| is)|sure|certainly|of course|i hope|let me know|in summary|"
    r"voici|bien s[uû]r|certamente|ecco|当然|希望|总之)", re.I)

def is_filler(sentence):
    """Excluded from faithfulness scoring in the WiLine framework: conversational
    filler, markdown artifacts and bare links are not verifiable claims."""
    s = sentence.strip()
    if len(s.split()) < 5 and len(s) < 30:
        return True
    if FILLER.match(s):
        return True
    if re.fullmatch(r"[\s*#\-_=|`~\[\]()]*", s):
        return True
    if s.lower().startswith(("http", "source:", "![")):
        return True
    return False

@torch.no_grad()
def nli_batch(premises, hypotheses, batch_size=32):
    probs = []
    for i in range(0, len(premises), batch_size):
        enc = nli_tok(premises[i:i + batch_size], hypotheses[i:i + batch_size],
                      return_tensors="pt", truncation=True, max_length=256,
                      padding=True).to(DEVICE)
        logits = nli_model(**enc).logits
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 3))

NLI_DIM = 6   # entail, neutral, contradict, cosine, filler flag, sentence position

def extract_nli(split, lang):
    rows = load_rows(split, lang)
    store, t0 = {}, time.time()
    for i, row in enumerate(rows):
        resp = row["response"]
        caption = CAPTIONS.get(row["image_name"], "")
        spans = split_sentences(resp)
        if not spans or not caption:
            store[row["id"] + "|off"] = np.zeros((0, 2), dtype=np.int32)
            store[row["id"] + "|f"] = np.zeros((0, NLI_DIM), dtype=np.float32)
            continue
        sents = [resp[a:b].strip() for a, b in spans]
        probs = nli_batch([caption] * len(sents), sents)
        emb = sim_model.encode([caption] + sents, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
        cos = emb[1:] @ emb[0]
        feats = np.zeros((len(spans), NLI_DIM), dtype=np.float32)
        for j, s in enumerate(sents):
            feats[j, :3] = probs[j]
            feats[j, 3] = cos[j]
            feats[j, 4] = float(is_filler(s))
            feats[j, 5] = j / max(len(sents) - 1, 1)
        store[row["id"] + "|off"] = np.array(spans, dtype=np.int32)
        store[row["id"] + "|f"] = feats
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {split}/{lang} {i + 1}/{len(rows)} {rate:.1f}/s", flush=True)
    path = f"{OUT_DIR}/nli_feats_{split}_{lang}.npz"
    np.savez_compressed(path, **store)
    print(f"{split}/{lang}: {len(store) // 2} saved, "
          f"{(time.time() - t0) / 60:.1f} min -> {path}", flush=True)

for split in ["test", "train"]:
    for lang in LANGS:
        extract_nli(split, lang)
        gc.collect(); torch.cuda.empty_cache()

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

d = np.load(f"{OUT_DIR}/nli_feats_train_en.npz")
rows = {json.loads(l)["id"]: json.loads(l)
        for l in open(f"{DISTRIB}/shroom-vision.train.en.labeled.jsonl")}

ys, xs, groups = [], [], []
for key in [k for k in d.files if k.endswith("|f")]:
    rid = key[:-2]
    if rid not in rows:
        continue
    row, offs, feats = rows[rid], d[rid + "|off"], d[key]
    gold = np.zeros(len(row["response"]))
    for s in row["labels"]:
        seg = gold[s["start"]:s["end"]]
        gold[s["start"]:s["end"]] = np.maximum(seg, s["prob"])
    for (a, b), f in zip(offs, feats):
        ys.append(gold[a:b].max() > 0 if b > a else False)
        xs.append(f); groups.append(rid)

ys, xs, groups = np.array(ys), np.array(xs), np.array(groups)
names = ["entail", "neutral", "contradict", "cosine", "filler", "position"]
print(f"{ys.sum()} hallucinated / {len(ys)} sentences ({100 * ys.mean():.1f}%)\n")
print(f"{'feature':<12}{'AUC':>8}")
for i, n in enumerate(names):
    a = roc_auc_score(ys, xs[:, i])
    print(f"{n:<12}{max(a, 1 - a):>8.3f}")

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
              .split(xs, ys, groups))
mu, sd = xs[tr].mean(0), xs[tr].std(0) + 1e-2
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit((xs[tr] - mu) / sd, ys[tr])
print(f"\nlogistic probe, grouped by response: "
      f"AUC {roc_auc_score(ys[te], clf.predict_proba((xs[te] - mu) / sd)[:, 1]):.3f}")
print("note: this is SENTENCE-level, so it is not comparable to the token-level "
      "numbers from the visual features")
