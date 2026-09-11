"""Caption every unique image with gemma4 via the WiLine endpoint."""

SMOKE = True          # caption 5 and stop; set False for the full run
WORKERS = 8
MAX_QUESTIONS = 8     # a few images carry up to 17; keep the prompt bounded

import base64, io, json, os, time, urllib.request, urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

BASE = "https://inference.wiline.com/v1/chat/completions"
CAPTION_MODEL = "gemma4"
CAPTION_PATH = f"{OUT_DIR}/captions_gemma4.json"

from kaggle_secrets import UserSecretsClient
API_KEY = UserSecretsClient().get_secret("WEC_API_KEY")

def questions_by_image():
    """image -> the distinct questions asked about it, across every split."""
    out = defaultdict(list)
    for split in ["train", "test"]:
        for lang in LANGS:
            for row in load_rows(split, lang):
                q = row["prompt"].strip()
                if q not in out[row["image_name"]]:
                    out[row["image_name"]].append(q)
    return dict(out)

QUESTIONS = questions_by_image()
print(f"{len(QUESTIONS)} unique images, "
      f"{sum(len(v) for v in QUESTIONS.values())} image-question pairs")

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

def _encode(name, max_side=1024):
    img = Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def caption_one(name):
    body = json.dumps({
        "model": CAPTION_MODEL,
        "max_tokens": 400,
        # Ollama defaults to 4096 whatever the model's native 131072, and an
        # image eats a large share of that. Without this, captions truncate.
        "num_ctx": 8192,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": build_prompt(QUESTIONS.get(name, []))},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{_encode(name)}"}}]}],
    }).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                BASE, data=body,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as r:
                msg = json.load(r)["choices"][0]["message"]["content"]
            return name, (msg or "").strip()
        except Exception as exc:
            if attempt == 3:
                print(f"  failed {name}: {str(exc)[:80]}", flush=True)
                return name, ""
            time.sleep(2 ** attempt)

def caption_all():
    caps = ({} if not os.path.exists(CAPTION_PATH)
            else json.load(open(CAPTION_PATH, encoding="utf-8")))
    todo = [n for n in QUESTIONS if not caps.get(n)]
    print(f"{len(todo)} left to caption")
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(caption_one, n) for n in todo]
        for fut in as_completed(futures):
            name, cap = fut.result()
            caps[name] = cap
            done += 1
            if done % 100 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(todo)}  {rate:.1f}/s  "
                      f"eta {(len(todo) - done) / rate / 60:.0f} min", flush=True)
                json.dump(caps, open(CAPTION_PATH, "w", encoding="utf-8"),
                          ensure_ascii=False)
    json.dump(caps, open(CAPTION_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    empty = sum(1 for v in caps.values() if not v)
    print(f"{len(caps)} captions, {empty} empty -> {CAPTION_PATH}")
    return caps

if SMOKE:
    # Read each caption against the response it is meant to ground, and against
    # the spans humans marked as hallucinated. If the caption does not contain
    # enough to see that those spans are unsupported, the feature cannot work
    # and there is no point running the other 2,471.
    for row in load_rows("train", "en")[:5]:
        _, cap = caption_one(row["image_name"])
        print("=" * 74)
        print(f"QUESTIONS ASKED OF THIS IMAGE: {QUESTIONS.get(row['image_name'], [])}")
        print(f"\nCAPTION :\n{cap[:900]}")
        print(f"\nRESPONSE: {row['response'][:300]}")
        if row["labels"]:
            print(f"GOLD HALLUCINATIONS: "
                  f"{[row['response'][s['start']:s['end']] for s in row['labels']]}")
    print("\nIf the captions answer the questions specifically and look accurate, "
          "set SMOKE = False and rerun.")
else:
    CAPTIONS = caption_all()
