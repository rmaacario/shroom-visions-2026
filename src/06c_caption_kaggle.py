"""Batched captioning with Qwen2.5-VL-7B on the Kaggle GPU."""

SMOKE = True          # 8 images and stop
BATCH_IMAGES = 8
MAX_NEW = 400         # captions ran ~600 tokens on gemma4; 400 is enough for
                      # the description plus the question answers
MAX_QUESTIONS = 6

import json, os, time, gc
from collections import defaultdict
import numpy as np
import torch
from PIL import Image

CAPTION_PATH = f"{OUT_DIR}/captions_qwen7b.json"
VLM_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# 4-bit keeps a 7B inside the T4's 15 GB with room for batched activations.
import subprocess
subprocess.run("pip install -q -U bitsandbytes", shell=True)
from transformers import BitsAndBytesConfig, AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLMClass
except ImportError:  # older transformers
    from transformers import Qwen2VLForConditionalGeneration as VLMClass
    VLM_ID = "Qwen/Qwen2-VL-7B-Instruct"
    print("falling back to Qwen2-VL-7B")

quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
vlm = VLMClass.from_pretrained(VLM_ID, device_map="auto", quantization_config=quant)
vlm.eval()
vproc = AutoProcessor.from_pretrained(VLM_ID, min_pixels=128 * 28 * 28,
                                      max_pixels=512 * 28 * 28)
# left padding is required for batched generation, or short sequences get their
# generated tokens attached after the pad run instead of after the prompt
vproc.tokenizer.padding_side = "left"
print(f"{VLM_ID} loaded")

def load_rows(split, lang):
    kind = "labeled" if split == "train" else "unlabeled"
    with open(f"{DISTRIB}/shroom-vision.{split}.{lang}.{kind}.jsonl",
              encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]

def questions_by_image():
    out = defaultdict(list)
    for split in ["train", "test"]:
        for lang in LANGS:
            for row in load_rows(split, lang):
                q = row["prompt"].strip()
                if q not in out[row["image_name"]]:
                    out[row["image_name"]].append(q)
    return dict(out)

QUESTIONS = questions_by_image()
print(f"{len(QUESTIONS)} unique images")

def build_prompt(questions):
    listed = "\n".join(f"{i}. {q}" for i, q in enumerate(questions[:MAX_QUESTIONS], 1))
    return (
        "Describe this image factually and in detail. State the objects present, "
        "their colours, shapes and other visible attributes, how many of each "
        "there are, where they are relative to each other, and transcribe any "
        "text visible in the image.\n\n"
        "Your description must explicitly answer each of these questions:\n"
        f"{listed}\n\n"
        "Describe only what you can actually see. If something cannot be "
        "determined from the image, say so plainly rather than guessing."
    )

@torch.no_grad()
def caption_batch(names):
    images, texts, ok = [], [], []
    for name in names:
        try:
            img = Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB")
        except Exception:
            continue
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": build_prompt(QUESTIONS.get(name, []))}]}]
        texts.append(vproc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))
        images.append(img)
        ok.append(name)
    if not ok:
        return {}
    inputs = vproc(text=texts, images=images, return_tensors="pt",
                   padding=True).to(vlm.device)
    out = vlm.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    decoded = vproc.tokenizer.batch_decode(gen, skip_special_tokens=True)
    return {n: t.strip() for n, t in zip(ok, decoded)}

def caption_all():
    caps = ({} if not os.path.exists(CAPTION_PATH)
            else json.load(open(CAPTION_PATH, encoding="utf-8")))
    todo = [n for n in QUESTIONS if not caps.get(n)]
    print(f"{len(todo)} left to caption, batch {BATCH_IMAGES}")
    t0 = time.time()
    for i in range(0, len(todo), BATCH_IMAGES):
        caps.update(caption_batch(todo[i:i + BATCH_IMAGES]))
        done = i + BATCH_IMAGES
        if done % (BATCH_IMAGES * 10) == 0:
            rate = done / (time.time() - t0)
            print(f"  {min(done, len(todo))}/{len(todo)}  {rate:.2f}/s  "
                  f"eta {(len(todo) - done) / rate / 60:.0f} min", flush=True)
            json.dump(caps, open(CAPTION_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False)
    json.dump(caps, open(CAPTION_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(caps)} captions -> {CAPTION_PATH}")
    return caps

if SMOKE:
    # One batch, timed. Two things must hold: the captions must be distinct
    # (a padding bug shows up as identical or truncated outputs), and they must
    # answer the questions.
    names = [r["image_name"] for r in load_rows("train", "en")[:BATCH_IMAGES]]
    t0 = time.time()
    got = caption_batch(names)
    dt = time.time() - t0
    print(f"\n{len(got)} captions in {dt:.0f}s "
          f"-> {dt / max(len(got), 1):.1f}s each, "
          f"eta for 2476: {2476 * dt / max(len(got), 1) / 60:.0f} min\n")
    for name, cap in list(got.items())[:2]:
        print("=" * 74)
        print(f"QUESTIONS: {QUESTIONS.get(name, [])[:MAX_QUESTIONS]}")
        print(f"CAPTION ({len(cap)} chars):\n{cap[:800]}\n")
    lens = [len(c) for c in got.values()]
    print(f"caption lengths: {lens}")
    print("all distinct:", len(set(got.values())) == len(got))
else:
    CAPTIONS = caption_all()
