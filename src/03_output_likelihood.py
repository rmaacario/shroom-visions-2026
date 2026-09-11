"""Output-likelihood features from Qwen2-VL-2B, teacher-forced with and without the image.

The image is removed, not noised: noise makes the model overlook regions at
random (Yin et al., arXiv 2504.10020), so the gap would measure what the noise
destroyed rather than what was grounded. Writes visual_feats_{split}_{lang}.npz:
character offsets plus six per-token features.
"""

SMOKE_TEST = True      # run 5 examples first; set False for the real run
MAX_EXAMPLES = None    # cap per (split, lang), or None for everything

import json, os, time, gc
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

DISTRIB   = "/kaggle/working/distrib"
IMAGE_DIR = "/kaggle/working/shroom-vis-images"
OUT_DIR   = "/kaggle/working"
MODEL_ID  = "Qwen/Qwen2-VL-2B-Instruct"
LANGS     = ["en", "fr", "it", "zh"]

# Cap visual tokens. Qwen2-VL uses dynamic resolution, and an unconstrained
# photo can produce >1500 visual tokens — the difference between a 3-hour run
# and a 12-hour one.
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", quantization_config=quant_cfg)
model.eval()
processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
tokenizer = processor.tokenizer
print("model loaded")

@torch.no_grad()
def score_response(image, prompt, response):
    """Teacher-force `response` twice and return per-token features.

    Returns (offsets, feats) where feats has one row per response token:
        [logp_img, ent_img, logp_noimg, ent_noimg, d_logp, d_ent]
    or (None, None) if the two tokenizations cannot be aligned.
    """
    # Character offsets come from tokenizing the response on its own.
    resp_enc = tokenizer(response, return_offsets_mapping=True,
                         add_special_tokens=False)
    resp_ids = resp_enc["input_ids"]
    offsets = resp_enc["offset_mapping"]
    if not resp_ids:
        return None, None

    def run(with_image):
        content = ([{"type": "image"}] if with_image else []) + \
                  [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        prefix = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [prefix + response], "return_tensors": "pt"}
        pre_kwargs = {"text": [prefix], "return_tensors": "pt"}
        if with_image:
            kwargs["images"] = [image]
            pre_kwargs["images"] = [image]
        inputs = processor(**kwargs).to(DEVICE)
        n_prefix = processor(**pre_kwargs)["input_ids"].shape[1]

        ids = inputs["input_ids"][0]
        # The response must tokenize identically in context, or offsets are wrong.
        if len(ids) - n_prefix != len(resp_ids):
            return None
        logits = model(**inputs).logits[0].float()
        # position i predicts token i+1
        step = logits[n_prefix - 1:-1]
        logprobs = torch.log_softmax(step, dim=-1)
        target = ids[n_prefix:]
        tok_logp = logprobs.gather(1, target.unsqueeze(1)).squeeze(1)
        entropy = -(logprobs.exp() * logprobs).sum(dim=-1)
        return tok_logp.cpu().numpy(), entropy.cpu().numpy()

    with_img = run(True)
    without_img = run(False)
    if with_img is None or without_img is None:
        return None, None

    logp_i, ent_i = with_img
    logp_n, ent_n = without_img
    feats = np.stack([logp_i, ent_i, logp_n, ent_n,
                      logp_i - logp_n, ent_i - ent_n], axis=1)
    return np.array(offsets, dtype=np.int32), feats.astype(np.float32)

def load_rows(split, lang):
    kind = "labeled" if split == "train" else "unlabeled"
    with open(f"{DISTRIB}/shroom-vision.{split}.{lang}.{kind}.jsonl",
              encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]

def extract(split, lang, limit=None):
    rows = load_rows(split, lang)
    if limit:
        rows = rows[:limit]
    out_path = f"{OUT_DIR}/visual_feats_{split}_{lang}.npz"

    store, skipped, t0 = {}, 0, time.time()
    for i, row in enumerate(rows):
        img_path = os.path.join(IMAGE_DIR, row["image_name"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                print(f"  image unreadable {row['image_name']}: {exc}")
            continue
        offsets, feats = score_response(image, row["prompt"], row["response"])
        if feats is None:
            skipped += 1
            continue
        store[row["id"] + "|off"] = offsets
        store[row["id"] + "|f"] = feats
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(rows) - i - 1) / rate / 60
            print(f"  {split}/{lang} {i + 1}/{len(rows)} "
                  f"{rate:.1f}/s  eta {eta:.0f} min  skipped {skipped}",
                  flush=True)

    np.savez_compressed(out_path, **store)
    done = len(store) // 2
    print(f"{split}/{lang}: {done} saved, {skipped} skipped, "
          f"{(time.time() - t0) / 60:.1f} min -> {out_path}", flush=True)
    return done, skipped

if SMOKE_TEST:
    rows = load_rows("train", "en")[:5]
    for row in rows:
        image = Image.open(os.path.join(IMAGE_DIR, row["image_name"])).convert("RGB")
        offsets, feats = score_response(image, row["prompt"], row["response"])
        if feats is None:
            print(f"{row['id']}: ALIGNMENT FAILED")
            continue
        # Sanity: characters covered by tokens should span the whole response.
        covered = offsets[-1][1] if len(offsets) else 0
        d = feats[:, 4]
        print(f"{row['id']}: {len(feats)} tokens, chars {covered}/{len(row['response'])}, "
              f"d_logp mean {d.mean():+.3f} min {d.min():+.3f} max {d.max():+.3f}")
        # Show which tokens the image mattered most / least for.
        order = np.argsort(d)
        toks = [row["response"][a:b] for a, b in offsets]
        print(f"   image mattered LEAST: {[toks[k] for k in order[:5]]}")
        print(f"   image mattered MOST : {[toks[k] for k in order[-5:]]}")
    print("\nSmoke test done. If alignment held and d_logp varies, set "
          "SMOKE_TEST = False and rerun.")
else:
    total = 0
    for split in ["test", "train"]:     # test first — it is what we must submit
        for lang in LANGS:
            done, _ = extract(split, lang, MAX_EXAMPLES)
            total += done
            gc.collect(); torch.cuda.empty_cache()
    print(f"\nall done: {total} examples")
