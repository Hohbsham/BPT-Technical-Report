"""
Batch standardize mesh face counts in supervised_data.

Categories:
  Small (target=1000): GL, HA, MA, SOL, SOM, SOS, ST
  Large (target=2500): all others

Usage:
  python standardize_faces.py [--dry_run] [--limit N] [--data_dir DIR]
"""
import subprocess
import sys
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict

BLENDER = r"C:\Users\YANGZ\blender\blender-4.2.1-windows-x64\blender-4.2.1-windows-x64\blender.exe"
REMESH_SCRIPT = Path(__file__).parent / "remesh_to_target.py"

# Small garment prefixes -> target 1000 quads
SMALL_PREFIXES = {"GL", "HA", "MA", "SOL", "SOM", "SOS", "ST"}

# Acceptable ranges: [min, max] for each category
SMALL_RANGE = (700, 1300)
LARGE_RANGE = (1800, 3200)


def get_category(sample_name):
    """Determine large/small from sample directory name prefix."""
    parts = sample_name.split("_")
    prefix = parts[0] if parts else sample_name
    return "small" if prefix in SMALL_PREFIXES else "large"


def count_quads(gt_path):
    """Count quad faces in an OBJ file."""
    quad_count = 0
    tri_count = 0
    with open(str(gt_path), 'r') as f:
        for line in f:
            if line.startswith("f "):
                parts = line.split()
                if len(parts) == 5:
                    quad_count += 1
                elif len(parts) == 4:
                    tri_count += 1
    return quad_count + tri_count


def needs_remesh(face_count, category):
    """Check if face count is outside acceptable range for category."""
    if face_count == 0:
        return False  # broken sample, skip
    lo, hi = SMALL_RANGE if category == "small" else LARGE_RANGE
    return face_count < lo or face_count > hi


def get_target_faces(category):
    return 1000 if category == "small" else 2500


def process_one(sample_dir, target_faces, timeout):
    """Run remesh_to_target.py in Blender subprocess."""
    cmd = [
        BLENDER, "--background",
        "--python", str(REMESH_SCRIPT), "--",
        str(sample_dir), str(target_faces),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stdout, _ = proc.communicate(timeout=timeout)
        if proc.returncode == 0 and "RESULT ok" in stdout:
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
    parser.add_argument("--data_dir", default=r"D:/ClothesNetData/supervised_data")
    parser.add_argument("--cache_dir", default=r"D:/ClothesNetData/bpt/cached_embeddings")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)

    # Scan all samples
    samples = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name == "Cube":
            continue
        gt = d / "ground_truth.obj"
        if not gt.exists():
            continue
        fc = count_quads(gt)
        cat = get_category(d.name)
        target = get_target_faces(cat)
        lo, hi = SMALL_RANGE if cat == "small" else LARGE_RANGE
        in_range = lo <= fc <= hi
        samples.append({
            "name": d.name,
            "dir": d,
            "face_count": fc,
            "category": cat,
            "target": target,
            "in_range": in_range,
        })

    # Summary
    large_samples = [s for s in samples if s["category"] == "large"]
    small_samples = [s for s in samples if s["category"] == "small"]
    print(f"Total: {len(samples)} samples")
    print(f"  Large (target=2500, 2000-3000): {len(large_samples)} | "
          f"in_range={sum(1 for s in large_samples if s['in_range'])} | "
          f"out={sum(1 for s in large_samples if not s['in_range'] and s['face_count'] > 0)} | "
          f"broken={sum(1 for s in large_samples if s['face_count'] == 0)}")
    print(f"  Small (target=1000, 800-1200):  {len(small_samples)} | "
          f"in_range={sum(1 for s in small_samples if s['in_range'])} | "
          f"out={sum(1 for s in small_samples if not s['in_range'] and s['face_count'] > 0)} | "
          f"broken={sum(1 for s in small_samples if s['face_count'] == 0)}")

    to_process = [s for s in samples if not s["in_range"] and s["face_count"] > 0]
    broken = [s for s in samples if s["face_count"] == 0]

    print(f"\nTo remesh: {len(to_process)} samples")
    print(f"Broken (0 faces, skip): {len(broken)} samples")

    if broken:
        print("Broken samples:")
        for s in broken:
            print(f"  - {s['name']}")

    if args.limit > 0:
        to_process = to_process[:args.limit]
        print(f"Limited to {args.limit} samples")

    if args.dry_run:
        print("\nDry run — would remesh:")
        for s in to_process:
            print(f"  {s['name']}: {s['face_count']} -> {s['target']} ({s['category']})")
        return

    if not to_process:
        print("\nAll samples already in range. Nothing to do.")
        return

    print(f"\n{'='*60}")
    print(f"Processing {len(to_process)} samples...")
    print(f"Blender: {BLENDER}")
    print(f"Timeout: {args.timeout}s per model")
    print(f"{'='*60}\n")

    success = 0
    failed = 0
    timeout_count = 0
    failed_list = []

    for i, s in enumerate(to_process):
        print(f"[{i+1}/{len(to_process)}] {s['name']} "
              f"({s['face_count']}->{s['target']}, {s['category']}) ...",
              end=" ", flush=True)
        start = time.time()

        result, output = process_one(s["dir"], s["target"], args.timeout)
        elapsed = time.time() - start

        if result == "ok":
            success += 1
            # Delete cached embedding if it exists
            cache_file = cache_dir / f"{s['name']}.pt"
            if cache_file.exists():
                cache_file.unlink()
            print(f"OK ({elapsed:.0f}s)")
        elif result == "timeout":
            timeout_count += 1
            failed_list.append(f"{s['name']}: TIMEOUT")
            print(f"TIMEOUT ({elapsed:.0f}s)")
        else:
            failed += 1
            failed_list.append(f"{s['name']}: FAILED")
            print(f"FAILED ({elapsed:.0f}s)")
            for line in output.strip().split("\n")[-3:]:
                if line.strip():
                    print(f"  | {line.strip()[:120]}")

    print(f"\n{'='*60}")
    print(f"DONE: {success} ok, {timeout_count} timeout, {failed} failed")
    if failed_list:
        print(f"Failed/timeout:")
        for m in failed_list:
            print(f"  - {m}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
