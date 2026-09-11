"""Reviewer 4stj Q3: does restricting the probe to content tokens change it?

Re-runs the representation-shift probe on English, three ways: all tokens,
content tokens only, non-content only. Response-level splits, five seeds,
same protocol as the paper.
"""
import json, numpy as np, re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

FEATS = "/Users/rafaelmacariofernandes/Downloads/feats/hidden_feats_train_en.npz"
GOLD  = "/Users/rafaelmacariofernandes/Downloads/shroom-visions/data/distrib/shroom-vision.train.en.labeled.jsonl"

rows = {json.loads(l)["id"]: json.loads(l) for l in open(GOLD, encoding="utf-8")}
d = np.load(FEATS)

X, y, g, is_content = [], [], [], []
for key in [k for k in d.files if k.endswith("|f")]:
    rid = key[:-2]
    if rid not in rows: continue
    row, offs, feats = rows[rid], d[rid + "|off"], d[key]
    resp = row["response"]
    gold = np.zeros(len(resp))
    for s in row["labels"]:
        gold[s["start"]:s["end"]] = np.maximum(gold[s["start"]:s["end"]], s["prob"])
    for (a, b), f in zip(offs, feats):
        if b <= a: continue
        surf = resp[a:b]
        # content = an alphabetic run of 4+ chars that starts a word
        starts_word = a == 0 or not resp[a-1].isalnum()
        content = bool(re.fullmatch(r"[^\W\d_]{4,}", surf.strip())) and starts_word
        X.append(f); y.append(float(gold[a:b].max()) > 0)
        g.append(rid); is_content.append(content)

X = np.asarray(X, np.float32); y = np.asarray(y); g = np.asarray(g)
is_content = np.asarray(is_content)
print(f"tokens {len(y)}  positives {y.mean():.1%}  content {is_content.mean():.1%}\n")

def probe(mask, name):
    Xs, ys, gs = X[mask], y[mask], g[mask]
    if len(set(ys)) < 2: print(f"  {name}: degenerate"); return
    scores = []
    for seed in range(5):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                        random_state=seed).split(Xs, ys, gs))
        mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-8
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit((Xs[tr]-mu)/sd, ys[tr])
        scores.append(roc_auc_score(ys[te], clf.predict_proba((Xs[te]-mu)/sd)[:,1]))
    print(f"  {name:24} AUROC {np.mean(scores):.3f} +- {np.std(scores):.3f}   "
          f"(n={len(ys)}, {ys.mean():.1%} positive)")

probe(np.ones(len(y), bool), "all tokens")
probe(is_content,            "content tokens only")
probe(~is_content,           "non-content only")
