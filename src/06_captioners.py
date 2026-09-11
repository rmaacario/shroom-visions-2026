"""Two captioners compared: Qwen2-VL-7B (open weights) and a GPT model via API."""

import json, os, time, base64, io, gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# The prompt targets the four hallucination categories directly: objects and
# attributes (invention, mischaracterization), counts (miscounting), and
# visible text (OCR).
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

IMAGE_NAMES = all_image_names()
print(f"{len(IMAGE_NAMES)} unique images")

def caption_with_qwen7b(out_path=None):
    out_path = out_path or f"{OUT_DIR}/captions_qwen7b.json"
    import torch
    from transformers import (Qwen2VLForConditionalGeneration, AutoProcessor,
                              BitsAndBytesConfig)

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    m = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", device_map="auto", quantization_config=quant)
    m.eval()
    p = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct",
                                      min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)

    caps = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    todo = [n for n in IMAGE_NAMES if n not in caps]
    print(f"qwen7b: {len(todo)} to go")
    t0 = time.time()
    for i, name in enumerate(todo):
        try:
            img = Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB")
        except Exception:
            caps[name] = ""
            continue
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}]}]
        text = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = p(text=[text], images=[img], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=140, do_sample=False)
        caps[name] = p.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(todo)} {rate:.2f}/s "
                  f"eta {(len(todo) - i - 1) / rate / 60:.0f} min", flush=True)
            json.dump(caps, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(caps, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    del m; gc.collect(); torch.cuda.empty_cache()
    print(f"{len(caps)} captions -> {out_path}")
    return caps

GPT_MODEL = "gpt-4o"     # swap for a newer vision model if you have access
GPT_WORKERS = 8

def _encode(name, max_side=1024):
    img = Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def caption_with_gpt(out_path=None):
    out_path = out_path or f"{OUT_DIR}/captions_gpt.json"
    from kaggle_secrets import UserSecretsClient
    from openai import OpenAI

    client = OpenAI(api_key=UserSecretsClient().get_secret("OPENAI_API_KEY"))
    caps = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    todo = [n for n in IMAGE_NAMES if n not in caps]
    print(f"gpt: {len(todo)} to go")

    def one(name):
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=GPT_MODEL, max_tokens=200,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": CAPTION_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{_encode(name)}"}}]}])
                return name, r.choices[0].message.content.strip(), r.usage
            except Exception as exc:
                if attempt == 2:
                    print(f"  failed {name}: {exc}")
                    return name, "", None
                time.sleep(2 ** attempt)

    t0, done, tok_in, tok_out = time.time(), 0, 0, 0
    with ThreadPoolExecutor(max_workers=GPT_WORKERS) as pool:
        futures = [pool.submit(one, n) for n in todo]
        for fut in as_completed(futures):
            name, cap, usage = fut.result()
            caps[name] = cap
            if usage:
                tok_in += usage.prompt_tokens; tok_out += usage.completion_tokens
            done += 1
            if done % 100 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(todo)} {rate:.1f}/s "
                      f"eta {(len(todo) - done) / rate / 60:.0f} min  "
                      f"tokens in/out {tok_in}/{tok_out}", flush=True)
                json.dump(caps, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(caps, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(caps)} captions -> {out_path}")
    print(f"tokens: {tok_in} in, {tok_out} out")
    return caps

def compare_captions(n=3):
    q = json.load(open(f"{OUT_DIR}/captions_qwen7b.json", encoding="utf-8"))
    g = json.load(open(f"{OUT_DIR}/captions_gpt.json", encoding="utf-8"))
    both = [k for k in q if k in g and q[k] and g[k]]
    print(f"{len(both)} images captioned by both\n")
    for name in both[:n]:
        print(f"--- {name}")
        print(f"  QWEN7B: {q[name][:400]}")
        print(f"  GPT   : {g[name][:400]}\n")
    ql = np.mean([len(q[k].split()) for k in both])
    gl = np.mean([len(g[k].split()) for k in both])
    print(f"mean caption length: qwen7b {ql:.0f} words, gpt {gl:.0f} words")
