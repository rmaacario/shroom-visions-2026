"""Calcula os números das figuras da sonda e grava CSVs para o R.

Rode na pasta que contém feats/ e shroom-visions/data/distrib/.
Gera probe_categories.csv, probe_layers.csv e probe_recomputed.json.
Nada aqui é escrito à mão: as figuras passam a ser reproduzíveis.
"""
import json, warnings, numpy as np, csv
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

D = "shroom-visions/data/distrib"
CATS = ["invention", "mischaracterization", "OCR", "miscounting", "other"]
LANGS = ["en", "fr", "it", "zh"]
NSPLIT, TEST = 5, 0.3
# 38 columns: [cos_early, dist_early, cos_mid, dist_mid, cos_late, dist_late] + 32 proj
BLOCKS = {"early": [0, 1], "middle": [2, 3], "late": [4, 5],
          "all three depths": [0, 1, 2, 3, 4, 5],
          "32-d projection": list(range(6, 38)), "all 38 features": list(range(38))}

def load(lang):
    d = np.load(f"feats/hidden_feats_train_{lang}.npz")
    rows = {json.loads(l)["id"]: json.loads(l)
            for l in open(f"{D}/shroom-vision.train.{lang}.labeled.jsonl")}
    X, y, cat, grp = [], [], [], []
    for key in [k for k in d.files if k.endswith("|f")]:
        rid = key[:-2]
        if rid not in rows: continue
        row, offs, feats = rows[rid], d[rid + "|off"], d[key]
        n = len(row["response"]); gold = np.zeros(n); gcat = np.empty(n, dtype=object)
        for s in row["labels"]:
            seg = gold[s["start"]:s["end"]]; better = s["prob"] >= seg
            gold[s["start"]:s["end"]] = np.maximum(seg, s["prob"])
            gcat[np.arange(s["start"], s["end"])[better]] = s["label"]
        for (a, b), f in zip(offs, feats):
            g = float(gold[a:b].max()) if b > a else 0.0
            c = None
            if g > 0:
                sl = [x for x in gcat[a:b] if x is not None]
                c = sl[0] if sl else "other"
            X.append(f); y.append(g > 0); cat.append(c); grp.append(rid)
    return np.array(X), np.array(y), np.array(cat, dtype=object), np.array(grp)

def fit(Xtr, ytr, Xte, cols):
    mu, sd = Xtr[:, cols].mean(0), Xtr[:, cols].std(0) + 1e-8
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit((Xtr[:, cols] - mu) / sd, ytr)
    return clf.predict_proba((Xte[:, cols] - mu) / sd)[:, 1]

res, layers = {}, {b: {l: [] for l in LANGS} for b in BLOCKS}
for lang in LANGS:
    X, y, cat, grp = load(lang)
    per = {c: [] for c in CATS}; overall = []
    for seed in range(NSPLIT):
        tr, te = next(GroupShuffleSplit(1, test_size=TEST, random_state=seed).split(X, y, grp))
        for b, cols in BLOCKS.items():
            layers[b][lang].append(roc_auc_score(y[te], fit(X[tr], y[tr], X[te], cols)))
        s = fit(X[tr], y[tr], X[te], BLOCKS["all 38 features"])
        yte, cte, neg = y[te], cat[te], s[~y[te]]
        overall.append(roc_auc_score(yte, s))
        for c in CATS:
            m = yte & (cte == c)
            if m.sum() >= 10:
                per[c].append(roc_auc_score(
                    np.r_[np.ones(m.sum()), np.zeros(len(neg))], np.r_[s[m], neg]))
    ntok = {c: int((y & (cat == c)).sum()) for c in CATS}
    npos = max(1, int(y.sum()))
    st = {c: (float(np.mean(v)), float(np.std(v))) for c, v in per.items() if v}
    res[lang] = dict(overall=[float(np.mean(overall)), float(np.std(overall))],
                     cat=st, ntok=ntok, freq={c: ntok[c] / npos for c in CATS},
                     uniform=float(np.mean([m for m, _ in st.values()])),
                     recon=float(sum(ntok[c] / npos * st[c][0] for c in st)))
json.dump(res, open("probe_recomputed.json", "w"), indent=1)

with open("probe_categories.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["lang", "category", "tokens", "share", "auroc", "sd"])
    for l in LANGS:
        for c in CATS:
            if c in res[l]["cat"]:
                m, sd = res[l]["cat"][c]
                w.writerow([l, c, res[l]["ntok"][c], f"{res[l]['freq'][c]:.5f}",
                            f"{m:.5f}", f"{sd:.5f}"])
with open("probe_layers.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["lang", "block", "auroc", "sd"])
    for b in BLOCKS:
        for l in LANGS:
            v = layers[b][l]
            w.writerow([l, b, f"{np.mean(v):.5f}", f"{np.std(v):.5f}"])
with open("probe_overall.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["lang", "auroc", "sd", "recon", "uniform"])
    for l in LANGS:
        w.writerow([l, f"{res[l]['overall'][0]:.5f}", f"{res[l]['overall'][1]:.5f}",
                    f"{res[l]['recon']:.5f}", f"{res[l]['uniform']:.5f}"])

print("== camadas (English) vs. os valores hardcoded em make_figures.py ==")
HARD = {"early": 0.534, "middle": 0.519, "late": 0.593, "all three depths": 0.668,
        "32-d projection": 0.620, "all 38 features": 0.682}
for b in BLOCKS:
    v = layers[b]["en"]
    print(f"  {b:18s} en={np.mean(v):.3f}+-{np.std(v):.3f}   hardcoded={HARD[b]:.3f}"
          f"   diff={np.mean(v)-HARD[b]:+.3f}")
print("\nwrote probe_categories.csv, probe_layers.csv, probe_overall.csv, probe_recomputed.json")
