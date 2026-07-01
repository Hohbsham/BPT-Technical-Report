"""Standalone Qwen-VL inspection - run me directly"""
import urllib.request, json, base64, os, sys, time

KEY = "sk-7fd674be9a5c49f3b278ccff317110ba"
RENDER_DIR = r"D:\ClothesNetData\inspection_renders"
OUTPUT_FILE = r"D:\ClothesNetData\inspection_qwen_results.json"

SAMPLES_TO_CHECK = 9999  # all samples

# Collect rendered samples
samples = []
for d in sorted(os.listdir(RENDER_DIR)):
    dpath = os.path.join(RENDER_DIR, d)
    if not os.path.isdir(dpath):
        continue
    views = [
        os.path.join(dpath, f"{d}_view1_front.png"),
        os.path.join(dpath, f"{d}_view2_side.png"),
        os.path.join(dpath, f"{d}_view3_top.png"),
    ]
    if all(os.path.exists(v) for v in views):
        samples.append((d, views))

print(f"Found {len(samples)} rendered samples, checking {SAMPLES_TO_CHECK}...")

PROMPT = """You are a 3D mesh quality inspector. These 3 views show a clothing mesh from different angles.

Analyze the mesh quality and return ONLY a JSON object:
{
  "score": <1-5>,
  "is_clean": <true/false>,
  "quad_quality": "<excellent|good|mediocre|poor>",
  "issues": ["<issue>"],
  "salvageable": <true/false>,
  "can_train": <true/false>,
  "reason": "<1 sentence>"
}

Scoring: 5=perfect clean quad mesh, 4=good mostly quads, 3=acceptable visible triangles, 2=problematic, 1=broken"""

results = {}
for i, (name, views) in enumerate(samples[:SAMPLES_TO_CHECK]):
    print(f"[{i+1}/{SAMPLES_TO_CHECK}] Inspecting {name}...")

    # Encode images
    content = []
    for v in views:
        with open(v, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": PROMPT})

    body = json.dumps({
        "model": "qwen-vl-max",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 500,
        "temperature": 0.1,
    }).encode()

    retries = 3
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
            )
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())
            text = data['choices'][0]['message']['content']
            # Parse JSON from response
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            result["sample"] = name
            results[name] = result
            print(f"  Score: {result.get('score')}/5 | Clean: {result.get('is_clean')} | {result.get('reason', '')[:80]}")
            break
        except json.JSONDecodeError:
            print(f"  JSON parse error, retry {attempt+1}")
            time.sleep(1)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(1)
    else:
        results[name] = {"sample": name, "error": "failed after retries"}
        print(f"  FAILED")

    # Incremental save after each sample
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    time.sleep(0.3)  # rate limit

# Final save (belt-and-suspenders)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to {OUTPUT_FILE}")
print(f"Summary:")
for name, r in results.items():
    score = r.get("score", "?")
    salvageable = "YES" if r.get("salvageable") else "NO"
    print(f"  {name}: score={score}/5, salvageable={salvageable}")
