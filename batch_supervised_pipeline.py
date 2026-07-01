"""
Batch supervised data pipeline with per-model timeout.
Launches each model in its own Blender subprocess so QuadriFlow hangs
are automatically killed and skipped.

Usage:
  python batch_supervised_pipeline.py [--data_dir DIR] [--output_dir DIR] ...

On Windows, Python must be available. Each model gets a dedicated Blender
background process with a timeout (default 180s per model).
"""

import subprocess
import sys
import os
import argparse
import time
from pathlib import Path

BLENDER = r"C:\Users\YANGZ\blender\blender-4.2.1-windows-x64\blender-4.2.1-windows-x64\blender.exe"
PROCESS_SCRIPT = Path(__file__).parent / "process_one_model.py"


def find_all_objs(data_dir, category="all"):
    data_dir = Path(data_dir)
    objs = []
    pattern = f"{category}/**/*.obj" if category != "all" else "**/*.obj"
    for obj_path in data_dir.glob(pattern):
        if obj_path.name == "border.obj":
            continue
        objs.append(obj_path)
    return sorted(objs)


def is_done(output_dir, model_name):
    """Check if a model already has all 3 output files."""
    out_dir = Path(output_dir) / model_name
    return (out_dir / "ground_truth.obj").exists() and \
           (out_dir / "input_damaged.obj").exists() and \
           (out_dir / "input_pointcloud.ply").exists()


def process_one(obj_path, output_dir, target_faces, damage_level, timeout):
    """Run process_one_model.py in a Blender subprocess with timeout."""
    cmd = [
        BLENDER, "--background",
        "--python", str(PROCESS_SCRIPT), "--",
        str(obj_path), str(output_dir),
        "--target_faces", str(target_faces),
        "--damage_level", damage_level,
    ]

    model_name = obj_path.parent.name
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stdout, _ = proc.communicate(timeout=timeout)
        if proc.returncode == 0 and "DONE" in stdout:
            return "ok", stdout
        else:
            return "failed", stdout
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return "timeout", f"TIMEOUT after {timeout}s"
    except Exception as e:
        return "error", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"D:/ClothesNetData/ClothesNetM")
    parser.add_argument("--output_dir", default=r"D:/ClothesNetData/supervised_data")
    parser.add_argument("--target_faces", type=int, default=2500)
    parser.add_argument("--category", default="all")
    parser.add_argument("--damage_level", default="medium",
                        choices=["light", "medium", "heavy"])
    parser.add_argument("--timeout", type=int, default=180,
                        help="Per-model timeout in seconds")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed models")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    obj_paths = find_all_objs(args.data_dir, args.category)
    print(f"Found {len(obj_paths)} OBJ files")
    print(f"Blender: {BLENDER}")
    print(f"Timeout: {args.timeout}s per model")
    if args.resume:
        print("Resume mode: skipping already-processed models")

    if args.limit > 0:
        obj_paths = obj_paths[:args.limit]
        print(f"Limited to {args.limit} models")

    if args.dry_run:
        for p in obj_paths:
            print(f"  {p.relative_to(args.data_dir)}")
        return

    success = 0
    failed = 0
    timeout_count = 0
    skipped = 0
    failed_models = []

    total = len(obj_paths)
    for i, obj_path in enumerate(obj_paths):
        model_name = obj_path.parent.name

        if args.resume and is_done(args.output_dir, model_name):
            skipped += 1
            print(f"[{i+1}/{total}] SKIP (done): {model_name}")
            continue

        print(f"[{i+1}/{total}] {model_name} ...", end=" ", flush=True)
        start = time.time()

        result, output = process_one(
            obj_path, args.output_dir,
            args.target_faces, args.damage_level, args.timeout
        )

        elapsed = time.time() - start

        if result == "ok":
            success += 1
            print(f"OK ({elapsed:.0f}s)")
        elif result == "timeout":
            timeout_count += 1
            failed_models.append(f"{model_name}: TIMEOUT")
            print(f"TIMEOUT ({elapsed:.0f}s) -> skipping")
        else:
            failed += 1
            failed_models.append(f"{model_name}: FAILED")
            print(f"FAILED ({elapsed:.0f}s)")
            # Print last few lines of output for debugging
            for line in output.strip().split("\n")[-3:]:
                if line.strip():
                    print(f"  | {line.strip()[:120]}")

    print(f"\n{'='*60}")
    print(f"DONE: {success} ok, {timeout_count} timeout, {failed} failed, "
          f"{skipped} skipped")
    if failed_models:
        print(f"Failed/timeout models:")
        for m in failed_models:
            print(f"  - {m}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
