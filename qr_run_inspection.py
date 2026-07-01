"""
数据集质量检查编排器
====================
遍历样本 → Blender渲染 → Qwen-VL评估 → 汇总报告

用法:
  python qr_run_inspection.py --test 5              # 抽样测试 5 个 excluded 样本
  python qr_run_inspection.py --test 5 --source all  # 从 excluded 全量中抽 5 个
  python qr_run_inspection.py --all --source excluded # 全量 excluded 1,228 样本
  python qr_run_inspection.py --resume               # 从断点续传
"""

import os
import sys
import json
import time
import shutil
import subprocess
import argparse
import random
from pathlib import Path

# ============ 配置 ============

BLENDER = r"D:\UserData\Blender\blender-4.2.19-windows-x64\blender.exe"
RENDER_SCRIPT = Path(__file__).parent / "qr_render_mesh.py"

EXCLUDED_DIRS = [
    Path(r"D:\ClothesNetData\supervised_data_excluded"),
    Path(r"D:\ClothesNetData\supervised_data_other_excluded"),
]
TRAIN_DIR = Path(r"D:\ClothesNetData\supervised_data")
RENDER_OUTPUT_DIR = Path(r"D:\ClothesNetData\inspection_renders")
RESULTS_FILE = Path(r"D:\ClothesNetData\inspection_results.json")
PROGRESS_FILE = Path(r"D:\ClothesNetData\inspection_progress.json")


# ============ 工具函数 ============

def collect_samples(source: str) -> list[tuple[Path, str]]:
    """收集待检查的样本"""
    samples = []
    if source in ("excluded", "all"):
        for ex_dir in EXCLUDED_DIRS:
            if not ex_dir.exists():
                continue
            for d in sorted(ex_dir.iterdir()):
                if d.is_dir() and (d / "ground_truth.obj").exists():
                    samples.append((d, d.name))
    if source in ("train", "all"):
        for d in sorted(TRAIN_DIR.iterdir()):
            if d.is_dir() and (d / "ground_truth.obj").exists():
                samples.append((d, d.name))
    return samples


def render_sample(sample_dir: Path, sample_name: str) -> bool:
    """用 Blender 渲染 3 个视角"""
    gt_obj = sample_dir / "ground_truth.obj"
    if not gt_obj.exists():
        print(f"  [ERROR] No ground_truth.obj in {sample_dir}")
        return False

    output_dir = RENDER_OUTPUT_DIR / sample_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 跳过已渲染的
    views_exist = all(
        (output_dir / f"{sample_name}_{v}.png").exists()
        for v in ["view1_front", "view2_side", "view3_top"]
    )
    if views_exist:
        print(f"  [SKIP] Already rendered")
        return True

    cmd = [
        BLENDER, "--background",
        "--python", str(RENDER_SCRIPT), "--",
        str(gt_obj), str(output_dir), sample_name
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  [ERROR] Blender exit {result.returncode}")
            # Show last few lines of output
            lines = result.stdout.strip().splitlines()
            for line in lines[-5:]:
                print(f"    {line}")
            return False
        print(f"  [OK] Rendered 3 views")
        return True
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Blender timeout")
        return False
    except Exception as e:
        print(f"  [ERROR] Blender: {e}")
        return False


def inspect_sample(sample_name: str) -> dict | None:
    """调用 Qwen-VL 评估渲染结果"""
    render_dir = RENDER_OUTPUT_DIR / sample_name

    # 检查是否有渲染图片
    views = [
        render_dir / f"{sample_name}_view1_front.png",
        render_dir / f"{sample_name}_view2_side.png",
        render_dir / f"{sample_name}_view3_top.png",
    ]
    if not all(v.exists() for v in views):
        print(f"  [ERROR] Missing renders for {sample_name}")
        return None

    # 调用 qr_qwen_inspect.py
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "qr_qwen_inspect.py"),
        str(render_dir), sample_name
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            print(f"  [ERROR] Qwen inspect exit {result.returncode}")
            print(f"  stderr: {result.stderr[-300:]}")
            return None

        # 从输出中提取 JSON
        output = result.stdout
        if "--- RESULT ---" in output:
            json_str = output.split("--- RESULT ---")[1].strip()
            return json.loads(json_str)
        else:
            # Try to parse the whole output as JSON
            return json.loads(output.strip())
    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON parse: {e}")
        print(f"  Raw output: {output[:300]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Qwen inspect timeout")
        return None
    except Exception as e:
        print(f"  [ERROR] Qwen inspect: {e}")
        return None


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "results": {}}


def save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2, ensure_ascii=False))


def generate_report(results: dict):
    """生成人类可读的质量报告"""
    lines = []
    lines.append("# Dataset Inspection Report")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Model: qwen-vl-max")
    lines.append("")

    # Stats
    scores = [r.get("overall_score", 0) for r in results.values() if "error" not in r]
    salvageable = [r for r in results.values() if r.get("salvageable")]
    good_for_training = [r for r in results.values() if r.get("can_use_as_training_data")]

    lines.append("## Summary")
    lines.append(f"- Total samples: {len(results)}")
    if scores:
        lines.append(f"- Mean score: {sum(scores)/len(scores):.1f}/5")
        lines.append(f"- Score distribution: 5★={scores.count(5)}  4★={scores.count(4)}  3★={scores.count(3)}  2★={scores.count(2)}  1★={scores.count(1)}")
    lines.append(f"- Salvageable: {len(salvageable)}")
    lines.append(f"- Can use as training data: {len(good_for_training)}")
    lines.append("")

    # Good samples (score >= 4)
    good = [(n, r) for n, r in results.items() if r.get("overall_score", 0) >= 4]
    if good:
        lines.append("## [Good] Samples (score >= 4) — Ready for training")
        lines.append("| Sample | Score | Quad Quality | Notes |")
        lines.append("|--------|-------|-------------|-------|")
        for name, r in sorted(good, key=lambda x: -x[1].get("overall_score", 0)):
            lines.append(f"| {name} | {r.get('overall_score')} | {r.get('quad_quality','?')} | {r.get('reason','')[:80]} |")
        lines.append("")

    # OK samples (score 3)
    ok = [(n, r) for n, r in results.items() if r.get("overall_score", 0) == 3]
    if ok:
        lines.append("## [OK] Samples (score = 3) — Usable with reservations")
        lines.append("| Sample | Score | Issues | Notes |")
        lines.append("|--------|-------|--------|-------|")
        for name, r in sorted(ok):
            issues = ", ".join(r.get("surface_issues", [])[:3])
            lines.append(f"| {name} | 3 | {issues} | {r.get('reason','')[:60]} |")
        lines.append("")

    # Bad samples (score <= 2)
    bad = [(n, r) for n, r in results.items() if r.get("overall_score", 0) <= 2]
    if bad:
        lines.append("## [Poor] Samples (score <= 2) — Not usable")
        lines.append("| Sample | Score | Issues | Reason |")
        lines.append("|--------|-------|--------|--------|")
        for name, r in sorted(bad, key=lambda x: x[1].get("overall_score", 0)):
            issues = ", ".join(r.get("surface_issues", [])[:3])
            lines.append(f"| {name} | {r.get('overall_score')} | {issues} | {r.get('reason','')[:60]} |")
        lines.append("")

    # Errors
    errors = [(n, r) for n, r in results.items() if "error" in r]
    if errors:
        lines.append("## Errors")
        for name, r in errors:
            lines.append(f"- **{name}**: {r['error']}")
        lines.append("")

    report_path = RENDER_OUTPUT_DIR / "inspection_report.md"
    report_path.write_text("\n".join(lines), encoding='utf-8')
    print(f"\nReport written to: {report_path}")
    return report_path


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="Dataset quality inspection pipeline")
    parser.add_argument("--test", type=int, default=0, help="Test mode: inspect N samples")
    parser.add_argument("--all", action="store_true", help="Process all samples")
    parser.add_argument("--source", type=str, default="excluded",
                        choices=["excluded", "train", "all"],
                        help="Data source")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--render-only", action="store_true", help="Only render, skip Qwen")
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect, skip render")
    args = parser.parse_args()

    # 验证环境
    if not args.inspect_only:
        if not Path(BLENDER).exists():
            print(f"[FATAL] Blender not found: {BLENDER}")
            sys.exit(1)
    if not args.render_only:
        if not os.environ.get("DASHSCOPE_API_KEY"):
            print("[FATAL] DASHSCOPE_API_KEY not set")
            sys.exit(1)

    RENDER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 收集样本
    all_samples = collect_samples(args.source)
    print(f"Total {args.source} samples: {len(all_samples)}")

    # 进度
    progress = load_progress()
    completed = set(progress.get("completed", []))

    # 选择样本
    if args.test > 0:
        # 排除已完成，随机抽样
        pending = [(d, n) for d, n in all_samples if n not in completed]
        random.seed(42)
        random.shuffle(pending)
        samples = pending[:args.test]
        print(f"Test mode: {len(samples)} samples (from {len(pending)} unprocessed)")
    elif args.all:
        samples = [(d, n) for d, n in all_samples if n not in completed]
        print(f"Full run: {len(samples)} samples remaining")
    else:
        print("Use --test N or --all")
        sys.exit(1)

    if not samples:
        print("All samples already processed!")
        generate_report(progress.get("results", {}))
        return

    # 逐个处理
    start_time = time.time()
    for i, (sample_dir, sample_name) in enumerate(samples):
        elapsed = time.time() - start_time
        eta = (elapsed / max(i, 1)) * (len(samples) - i) if i > 0 else 0
        print(f"\n[{i+1}/{len(samples)}] {sample_name}  (elapsed: {elapsed/60:.1f}m, eta: {eta/60:.1f}m)")

        # Step 1: Render
        if not args.inspect_only:
            ok = render_sample(sample_dir, sample_name)
        else:
            ok = True

        # Step 2: Inspect with Qwen-VL
        if ok and not args.render_only:
            result = inspect_sample(sample_name)
            if result:
                progress["results"][sample_name] = result
                score = result.get("overall_score", "?")
                salvageable = "SALVAGEABLE" if result.get("salvageable") else "BROKEN"
                print(f"  Score: {score}/5 | {salvageable} | {result.get('reason', '')[:80]}")
            else:
                progress["results"][sample_name] = {"sample": sample_name, "error": "inspect failed"}
                print(f"  [FAILED] Inspection failed")
        elif not ok:
            progress["results"][sample_name] = {"sample": sample_name, "error": "render failed"}

        progress["completed"].append(sample_name)
        save_progress(progress)

        # API rate limiting
        if not args.render_only and ok:
            time.sleep(0.5)

    # 报告
    total_time = (time.time() - start_time) / 60
    print(f"\nDone! Total time: {total_time:.1f}m")
    print(f"Processed: {len(progress['completed'])} samples")
    generate_report(progress.get("results", {}))


if __name__ == "__main__":
    main()
