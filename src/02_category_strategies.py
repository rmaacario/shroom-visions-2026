"""Three ways of turning per-token category scores into span labels.

The span head stays exactly as trained; nothing is retrained.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def predict_char_full(model, rows):
    """Per-character hallucination probability plus the full category distribution.

    Returns (char_prob, char_cat_probs) per row, where char_cat_probs has shape
    (n_chars, len(CATEGORIES)). Keeping the distribution instead of an argmax is
    what lets us pool categories over a span or a whole response.
    """
    model.eval()
    dataset = SpanData(rows, labeled=False)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, collate_fn=collate)
    results, cursor = [], 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        span_logit, cat_logit = model(batch["input_ids"], batch["attention_mask"],
                                      batch["keep"], batch["keep_mask"])
        probs = torch.sigmoid(span_logit).cpu().numpy()
        cat_p = torch.softmax(cat_logit, dim=-1).cpu().numpy()
        for i in range(len(probs)):
            row = rows[cursor]
            enc, keep = dataset.cache[cursor]
            cursor += 1
            offs = [enc["offset_mapping"][k] for k in keep]
            n = len(row["response"])
            cp = np.zeros(n, dtype=np.float32)
            cd = np.zeros((n, len(CATEGORIES)), dtype=np.float32)
            for j, (a, b) in enumerate(offs):
                e = min(b, n)
                cp[a:e] = probs[i][j]
                cd[a:e] = cat_p[i][j]
            results.append((cp, cd))
    return results

def spans_with_strategy(char_prob, char_dist, threshold, strategy):
    """Build spans, assigning categories by one of three strategies.

    per_token : argmax at each character; a span breaks where the label changes
    per_span  : one label per contiguous run, pooled over its characters
    per_resp  : one label for the whole response, pooled over flagged characters
    """
    n = len(char_prob)
    above = char_prob >= threshold
    if not above.any():
        return []

    resp_label = int(char_dist[above].sum(axis=0).argmax())

    # contiguous runs of flagged characters
    runs, i = [], 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        runs.append((i, j))
        i = j

    spans = []
    for a, b in runs:
        if strategy == "per_resp":
            pieces = [(a, b, resp_label)]
        elif strategy == "per_span":
            pieces = [(a, b, int(char_dist[a:b].sum(axis=0).argmax()))]
        else:  # per_token — split the run wherever the argmax label changes
            labels = char_dist[a:b].argmax(axis=1)
            pieces, s = [], 0
            for k in range(1, len(labels) + 1):
                if k == len(labels) or labels[k] != labels[s]:
                    pieces.append((a + s, a + k, int(labels[s])))
                    s = k
        for s, e, lab in pieces:
            spans.append({"start": int(s), "end": int(e),
                          "prob": float(round(float(char_prob[s:e].mean()), 6)),
                          "label": CATEGORIES[lab]})
    return spans

def eval_strategy(rows, char_preds, threshold, strategy):
    refs = [{"id": r["id"], "labels": r["labels"], "text_len": len(r["response"])}
            for r in rows]
    preds = [{"id": r["id"],
              "labels": spans_with_strategy(cp, cd, threshold, strategy)}
             for r, (cp, cd) in zip(rows, char_preds)]
    scores = evaluate(refs, preds)
    counts = [len({s["label"] for s in p["labels"]}) for p in preds if p["labels"]]
    scores["cats_per_resp"] = float(np.mean(counts)) if counts else 0.0
    scores["n_flagged"] = len(counts)
    return scores

STRATEGIES = ["per_token", "per_span", "per_resp"]
best_strategy, best_thr = {}, {}

print(f"{'lang':<5}{'strategy':<12}{'thr':>6}{'Cor':>9}{'Cor_lbl':>10}{'cats/resp':>11}")
print("-" * 53)
for lang in LANGS:
    rows = [r for r in dev_rows if r["language"] == lang]
    preds_full = predict_char_full(model, rows)
    best = (None, None, -1)
    for strategy in STRATEGIES:
        top = (None, -1, None)
        for t in np.arange(0.10, 0.91, 0.05):
            s = eval_strategy(rows, preds_full, float(t), strategy)
            # Cor+Lbl is listed first on the leaderboard, so tune for it
            if s["Cor_lbl"] > top[1]:
                top = (float(t), s["Cor_lbl"], s)
        thr, cl, s = top
        print(f"{lang:<5}{strategy:<12}{thr:>6.2f}{s['Cor']:>9.3f}"
              f"{s['Cor_lbl']:>10.3f}{s['cats_per_resp']:>11.2f}")
        if cl > best[2]:
            best = (strategy, thr, cl)
    best_strategy[lang], best_thr[lang] = best[0], best[1]
    print(f"{'':5}-> best: {best[0]} @ {best[1]:.2f}  (Cor_lbl {best[2]:.3f})")
    print()

print("chosen:", {l: (best_strategy[l], round(best_thr[l], 2)) for l in LANGS})
