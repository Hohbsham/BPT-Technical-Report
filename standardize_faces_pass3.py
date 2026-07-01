"""
Third pass: retry QuadriFlow on triangle-only meshes to see if any convert to quads.
"""
import subprocess
import sys
import time
from pathlib import Path

BLENDER = r"C:\Users\YANGZ\blender\blender-4.2.1-windows-x64\blender-4.2.1-windows-x64\blender.exe"
REMESH_SCRIPT = Path(__file__).parent / "remesh_to_target.py"

SMALL_PREFIXES = {"GL", "HA", "MA", "SOL", "SOM", "SOS", "ST"}


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
    return quad + tri, quad, tri


def main():
    data_dir = Path(r"D:/ClothesNetData/supervised_data")

    tri_samples = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name == "Cube":
            continue
        gt = d / "ground_truth.obj"
        if not gt.exists():
            continue
        total, quad, tri = count_faces(gt)
        if total == 0:
            continue
        if quad == 0 and tri > 0:
            parts = d.name.split("_")
            prefix = parts[0] if parts else d.name
            cat = "small" if prefix in SMALL_PREFIXES else "large"
            target = 1000 if cat == "small" else 2500
            tri_samples.append({"name": d.name, "dir": d, "face_count": total,
                              "category": cat, "target": target})

    print(f"Triangle-only samples to retry: {len(tri_samples)}")

    converted = 0
    still_tri = 0
    failed = 0

    for i, s in enumerate(tri_samples):
        print(f"[{i+1}/{len(tri_samples)}] {s['name']} "
              f"({s['face_count']} tris -> target={s['target']}, {s['category']}) ...",
              end=" ", flush=True)

        cmd = [
            BLENDER, "--background",
            "--python", str(REMESH_SCRIPT), "--",
            str(s["dir"]), str(s["target"]),
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            stdout, _ = proc.communicate(timeout=180)
            ok = proc.returncode == 0 and "RESULT ok" in stdout
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            ok = False

        if ok:
            total, quad, tri = count_faces(s["dir"] / "ground_truth.obj")
            if quad > 0:
                converted += 1
                print(f"CONVERTED: {quad}q + {tri}t = {total}")
                # Delete cache for re-precompute
                cache = Path(r"D:/ClothesNetData/bpt/cached_embeddings") / f"{s['name']}.pt"
                if cache.exists():
                    cache.unlink()
            else:
                still_tri += 1
                print(f"STILL TRIS: {tri}t")
        else:
            failed += 1
            print(f"FAILED")

    print(f"\n{'='*60}")
    print(f"Retry results: {converted} converted to quads, "
          f"{still_tri} still triangle, {failed} failed")


if __name__ == "__main__":
    main()
