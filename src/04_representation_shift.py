"""Representation-shift features: the same two passes, internal states instead of logits.

Cosine similarity and Euclidean distance between the with- and without-image
states at three depths, plus a fixed 32-d random projection of the middle-layer
difference -- 38 features per token. Writes hidden_feats_{split}_{lang}.npz.
"""

import numpy as np, torch, json, os, time, gc

PROJ_DIM = 32
_rng = np.random.RandomState(0)
_proj = None   # built lazily once the hidden size is known

@torch.no_grad()
def score_response_hidden(image, prompt, response):
    global _proj
    resp_enc = tokenizer(response, return_offsets_mapping=True,
                         add_special_tokens=False)
    resp_ids = resp_enc["input_ids"]
    offsets = resp_enc["offset_mapping"]
    if not resp_ids:
        return None, None
    n_resp = len(resp_ids)

    def run(with_image):
        content = ([{"type": "image"}] if with_image else []) + \
                  [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        prefix = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [prefix + response], "return_tensors": "pt"}
        if with_image:
            kwargs["images"] = [image]
        inputs = processor(**kwargs).to(DEVICE)
        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states
        n = len(hs)
        picks = [n // 4, n // 2, (3 * n) // 4]
        # response tokens are the final n_resp positions in both passes
        return [hs[p][0, -n_resp:, :].float().cpu().numpy() for p in picks]

    a = run(True)
    b = run(False)
    if a[0].shape[0] != n_resp or b[0].shape[0] != n_resp:
        return None, None

    if _proj is None:
        _proj = _rng.randn(a[1].shape[1], PROJ_DIM).astype(np.float32) / np.sqrt(PROJ_DIM)

    cols = []
    for hi, hn in zip(a, b):
        num = (hi * hn).sum(axis=1)
        den = np.linalg.norm(hi, axis=1) * np.linalg.norm(hn, axis=1) + 1e-8
        cols.append(num / den)                        # cosine
        cols.append(np.linalg.norm(hi - hn, axis=1))  # distance
    feats = np.stack(cols, axis=1)
    proj = (a[1] - b[1]) @ _proj
    return np.array(offsets, dtype=np.int32), \
        np.concatenate([feats, proj], axis=1).astype(np.float32)

def extract_hidden(split, lang, limit=None):
    rows = load_rows(split, lang)
    if limit:
        rows = rows[:limit]
    store, skipped, t0 = {}, 0, time.time()
    for i, row in enumerate(rows):
        try:
            image = Image.open(os.path.join(IMAGE_DIR, row["image_name"])).convert("RGB")
        except Exception:
            skipped += 1
            continue
        offsets, feats = score_response_hidden(image, row["prompt"], row["response"])
        if feats is None:
            skipped += 1
            continue
        store[row["id"] + "|off"] = offsets
        store[row["id"] + "|f"] = feats
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(rows)} {rate:.1f}/s skipped {skipped}", flush=True)
    path = f"{OUT_DIR}/hidden_feats_{split}_{lang}.npz"
    np.savez_compressed(path, **store)
    print(f"{len(store) // 2} saved, {skipped} skipped, "
          f"{(time.time() - t0) / 60:.1f} min -> {path}", flush=True)

extract_hidden("train", "en", 300)
gc.collect(); torch.cuda.empty_cache()

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

d = np.load(f"{OUT_DIR}/hidden_feats_train_en.npz")
rows = {json.loads(l)["id"]: json.loads(l)
        for l in open(f"{DISTRIB}/shroom-vision.train.en.labeled.jsonl")}

ys, gs, xs = [], [], []
for key in [k for k in d.files if k.endswith("|f")]:
    rid = key[:-2]
    row, offs, feats = rows[rid], d[rid + "|off"], d[key]
    gold = np.zeros(len(row["response"]))
    for s in row["labels"]:
        seg = gold[s["start"]:s["end"]]
        gold[s["start"]:s["end"]] = np.maximum(seg, s["prob"])
    for (a, b), f in zip(offs, feats):
        g = float(gold[a:b].max()) if b > a else 0.0
        ys.append(g > 0); gs.append(g); xs.append(f)

ys, gs, xs = np.array(ys), np.array(gs), np.array(xs)
names = ["cos_early", "dist_early", "cos_mid", "dist_mid", "cos_late", "dist_late"]
print(f"{ys.sum()} hallucinated / {len(ys)} tokens ({100 * ys.mean():.1f}%)\n")
print(f"{'feature':<12}{'AUC':>8}{'Spearman':>11}")
for i, n in enumerate(names):
    # try both directions; a scalar can separate either way
    auc = roc_auc_score(ys, xs[:, i])
    print(f"{n:<12}{max(auc, 1 - auc):>8.3f}{spearmanr(gs, xs[:, i]).correlation:>11.3f}")

# can a classifier find structure in the full difference vector?
Xtr, Xte, ytr, yte = train_test_split(xs, ys, test_size=0.3, random_state=0, stratify=ys)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit((Xtr - mu) / sd, ytr)
probe = roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1])
print(f"\nlogistic probe on all {xs.shape[1]} features: AUC {probe:.3f}")
print("(0.50 = no signal; >0.60 means activations carry what logits did not)")
