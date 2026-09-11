"""Character <-> token conversion for the span tagger.

The model predicts one score per token; the scorer wants one probability per
character. Everything the model sees or emits passes through here, so this is
the one place where information can silently leak away.
"""
import numpy as np

MAX_LEN = 512

def char_targets(row):
    """Gold labels -> per-character arrays: hallucination probability and category."""
    n = len(row["response"])
    prob = np.zeros(n, dtype=np.float32)
    cat = np.full(n, "", dtype=object)
    for span in row["labels"]:
        for i in range(span["start"], min(span["end"], n)):
            prob[i] = max(prob[i], span["prob"])
            cat[i] = span["label"]
    return prob, cat

def encode(tokenizer, response, prompt=None):
    """Tokenize the response, keeping character offsets.

    The prompt is prepended as a second segment so the model can see the
    question, but only response tokens carry offsets we score against.
    """
    enc = tokenizer(
        response,
        text_pair=prompt if prompt else None,
        return_offsets_mapping=True,
        truncation="only_first",
        max_length=MAX_LEN,
        return_tensors=None,
    )
    offsets = enc["offset_mapping"]
    seq_ids = enc.sequence_ids() if hasattr(enc, "sequence_ids") else None
    # keep only real response tokens: segment 0, non-empty span
    keep = [
        i for i, (a, b) in enumerate(offsets)
        if b > a and (seq_ids is None or seq_ids[i] == 0)
    ]
    return enc, [offsets[i] for i in keep], keep

def tokens_to_char_prob(offsets, token_prob, n_chars):
    """Scatter per-token scores back onto characters."""
    out = np.zeros(n_chars, dtype=np.float32)
    for (a, b), p in zip(offsets, token_prob):
        out[a:min(b, n_chars)] = p
    return out

def char_prob_to_spans(char_prob, char_cat, threshold=0.5, min_len=1):
    """Merge contiguous above-threshold characters into scorer-format spans.

    Adjacent runs are split when the predicted category changes, so each emitted
    span carries a single label.
    """
    spans = []
    n = len(char_prob)
    i = 0
    while i < n:
        if char_prob[i] < threshold:
            i += 1
            continue
        j = i
        cat = char_cat[i]
        while j < n and char_prob[j] >= threshold and char_cat[j] == cat:
            j += 1
        if j - i >= min_len:
            spans.append({
                "start": int(i),
                "end": int(j),
                "prob": float(round(np.mean(char_prob[i:j]), 6)),
                "label": cat if cat else "mischaracterization",
            })
        i = j
    return spans

def token_targets(offsets, prob_arr, cat_arr):
    """Per-token gold: mean character probability, and the majority category."""
    tok_prob, tok_cat = [], []
    for a, b in offsets:
        seg = prob_arr[a:b]
        tok_prob.append(float(seg.max()) if len(seg) else 0.0)
        cats = [c for c in cat_arr[a:b] if c]
        tok_cat.append(max(set(cats), key=cats.count) if cats else "")
    return np.array(tok_prob, dtype=np.float32), tok_cat
