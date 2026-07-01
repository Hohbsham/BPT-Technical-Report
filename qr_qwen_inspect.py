"""
Qwen-VL 多模态质量评估脚本
用法: python qr_qwen_inspect.py <render_dir> <sample_name>
输出: JSON 质量评估结果
"""
import os
import sys
import json
import base64
import urllib.request
import time
from pathlib import Path

# ============ 配置 ============

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "") or "sk-7fd674be9a5c49f3b278ccff317110ba"
MODEL = "qwen-vl-max"
MAX_RETRIES = 3
RETRY_DELAY = 2.0


# ============ API 调用 ============

def encode_image(image_path: str) -> str:
    """将图片编码为 base64 data URL"""
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/png;base64,{data}"


def call_qwen_vl(images: list[str], prompt: str) -> dict | None:
    """调用 Qwen-VL 多模态 API（OpenAI 兼容格式）"""
    if not API_KEY:
        print("  [ERROR] DASHSCOPE_API_KEY not set")
        return None

    # Build content array with images + text
    content = []
    for img_path in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image(img_path)}
        })
    content.append({"type": "text", "text": prompt})

    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2000,
        "temperature": 0.1,
    }).encode('utf-8')

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(API_URL, data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            })
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())
            return json.loads(data['choices'][0]['message']['content'])
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  [ERROR] API call failed (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None

    return None


# ============ 评估 Prompt ============

INSPECTION_PROMPT = """You are a 3D mesh quality inspector. You are shown 3 rendered views (front-isometric, side, top-down) of a clothing 3D mesh.

Analyze the mesh and return a JSON object with EXACTLY this structure (no other text):

{
  "overall_score": <1-5 integer>,
  "is_clean_mesh": <true/false>,
  "has_broken_geometry": <true/false>,
  "quad_quality": "<excellent|good|mediocre|poor>",
  "surface_issues": ["<issue1>", "<issue2>"],
  "topology_notes": "<brief description of mesh topology>",
  "salvageable": <true/false>,
  "can_use_as_training_data": <true/false>,
  "reason": "<1-2 sentence explanation>"
}

Scoring:
- 5: Perfect clean quad mesh, uniform topology, no visible defects
- 4: Good mesh, mostly quads, minor imperfections
- 3: Acceptable, visible triangles or uneven faces but usable
- 2: Problematic, significant defects, lots of triangles
- 1: Broken, unusable, severe geometry issues

Evaluate: mesh completeness, face regularity, edge quality, surface smoothness, whether the garment shape is intact.

Return ONLY the JSON, no markdown, no explanation outside the JSON."""


# ============ 主函数 ============

def inspect_sample(render_dir: str, sample_name: str) -> dict:
    """检查单个样本的渲染图片"""
    view_files = [
        os.path.join(render_dir, f"{sample_name}_view1_front.png"),
        os.path.join(render_dir, f"{sample_name}_view2_side.png"),
        os.path.join(render_dir, f"{sample_name}_view3_top.png"),
    ]

    # 验证所有图片存在
    for vf in view_files:
        if not os.path.exists(vf):
            return {
                "sample": sample_name,
                "error": f"Missing render: {os.path.basename(vf)}",
                "overall_score": 0,
                "salvageable": False,
            }

    print(f"  Sending {len(view_files)} views to {MODEL}...")
    result = call_qwen_vl(view_files, INSPECTION_PROMPT)

    if result is None:
        return {
            "sample": sample_name,
            "error": "API call failed after retries",
            "overall_score": 0,
            "salvageable": False,
        }

    result["sample"] = sample_name
    result["render_dir"] = render_dir
    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python qr_qwen_inspect.py <render_dir> <sample_name>")
        print("  render_dir: directory containing <sample_name>_view*.png")
        print("  sample_name: name of the sample")
        sys.exit(1)

    render_dir = sys.argv[1]
    sample_name = sys.argv[2]

    result = inspect_sample(render_dir, sample_name)

    # Output as JSON line for easy parsing
    print("\n--- RESULT ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
