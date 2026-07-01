"""
Second pass: re-remesh remaining outliers with adjusted targets.
Usage:
  python standardize_faces_pass2.py
"""
import subprocess
import sys
import argparse
import time
from pathlib import Path

BLENDER = r"C:\Users\YANGZ\blender\blender-4.2.1-windows-x64\blender-4.2.1-windows-x64\blender.exe"
REMESH_SCRIPT = Path(__file__).parent / "remesh_to_target.py"

SMALL_PREFIXES = {"GL", "HA", "MA", "SOL", "SOM", "SOS", "ST"}
LARGE_RANGE = (1800, 3200)
SMALL_RANGE = (700, 1300)


def get_category(name):
    parts = name.split("_")
    prefix = parts[0] if parts else name
    return "small" if prefix in SMALL_PREFIXES else "large"


def count_faces(gt_path):
    quad = tri = 0
    with open(str(gt_path)) as f:
        for line in f:
            if not line.startswith("f "):
                continue
            parts = line.split()
            if len(parts) == 5:
                quad += 1
            elif len(parts) == 4:
                tri += 1
    return quad + tri


def process_one(sample_dir, target_faces, timeout=180):
    cmd = [
        BLENDER, "--background",
        "--python", str(REMESH_SCRIPT), "--",
        str(sample_dir), str(target_faces),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        stdout, _ = proc.communicate(timeout=timeout)
        return ("ok" if proc.returncode == 0 and "RESULT ok" in stdout else "failed"), stdout
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return "timeout", f"TIMEOUT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"D:/ClothesNetData/supervised_data")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Find outliers
    outliers = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name == "Cube":
            continue
        gt = d / "ground_truth.obj"
        if not gt.exists():
            continue
        fc = count_faces(gt)
        if fc == 0:
            continue
        cat = get_category(d.name)
        lo, hi = SMALL_RANGE if cat == "small" else LARGE_RANGE
        if fc < lo or fc > hi:
            # Determine adjusted target
            if cat == "large":
                target = 2000 if fc > hi else 2800
            else:
                target = 800 if fc > hi else 1200
            outliers.append({"name": d.name, "dir": d, "face_count": fc,
                           "category": cat, "target": target})

    print(f"Remaining outliers: {len(outliers)}")
    if args.limit > 0:
        outliers = outliers[:args.limit]
        print(f"Limited to {args.limit}")

    if args.dry_run:
        for s in outliers:
            print(f"  {s['name']}: {s['face_count']} -> {s['target']} ({s['category']})")
        return

    if not outliers:
        print("No outliers to process.")
        return

    to_process = outliers
    print(f"Processing {len(to_process)}...")

    success = failed = timeout_count = 0
    failed_list = []

    for i, s in enumerate(to_process):
        print(f"[{i+1}/{len(to_process)}] {s['name']} "
              f"({s['face_count']}->{s['target']}, {s['category']}) ...",
              end=" ", flush=True)
        start = time.time()
        result, output = process_one(s["dir"], s["target"])
        elapsed = time.time() - start

        if result == "ok":
            success += 1
            # Re-check result
            new_fc = count_faces(s["dir"] / "ground_truth.obj")
            lo, hi = SMALL_RANGE if s["category"] == "small" else LARGE_RANGE
            in_range = lo <= new_fc <= hi
            tag = "IN" if in_range else f"still={new_fc}"
            print(f"OK ({elapsed:.0f}s) {tag}")
            # Delete cache
            cache = Path(r"D:/ClothesNetData/bpt/cached_embeddings") / f"{s['name']}.pt"
            if cache.exists():
                cache.unlink()
        elif result == "timeout":
            timeout_count += 1
            failed_list.append(f"{s['name']}: TIMEOUT")
            print(f"TIMEOUT")
        else:
            failed += 1
            failed_list.append(f"{s['name']}: FAILED")
            print(f"FAILED")

    # Final summary
    final_out = 0
    for d in data_dir.iterdir():
        if not d.is_dir() or d.name == "Cube":
            continue
        gt = d / "ground_truth.obj"
        if not gt.exists():
            continue
        fc = count_faces(gt)
        if fc == 0:
            continue
        cat = get_category(d.name)
        lo, hi = SMALL_RANGE if cat == "small" else LARGE_RANGE
        if fc < lo or fc > hi:
            final_out += 1

    total = sum(1 for d in data_dir.iterdir() if d.is_dir() and d.name != "Cube")
    print(f"\n{'='*60}")
    print(f"PASS 2: {success} ok, {timeout_count} timeout, {failed} failed")
    print(f"Remaining outliers: {final_out}/{total} ({(1-final_out/total)*100:.1f}% in range)")
    if failed_list:
        print(f"Failed:")
        for m in failed_list[:10]:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
