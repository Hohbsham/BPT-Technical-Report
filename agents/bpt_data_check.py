"""
BPT_Data_check Agent — Automated Data Quality Pipeline
Handles: raw OBJ → QuadriFlow remesh → quality check → boundary/keypoint extraction → supervised_data

Register on A2A platform with capabilities:
  data_quality, mesh_inspection, quad_check, boundary_extract, kp_extract, reprocess_failed
"""
import os, sys, json, subprocess, shutil
from pathlib import Path
from collections import Counter
import trimesh
import numpy as np

BLENDER = r"C:\Users\YANGZ\blender\blender-4.2.1-windows-x64\blender-4.2.1-windows-x64\blender.exe"
PROCESS_SCRIPT = Path(__file__).parent.parent / "process_one_model.py"
DATA_ROOT = Path(r"D:\ClothesNetData")


class BPTDataChecker:
    """Quality inspection agent for garment mesh data."""

    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else DATA_ROOT / "supervised_data_new"
        self.output_dir.mkdir(exist_ok=True)
        self.report = {"total": 0, "ok": 0, "failed": 0, "boundary_issues": 0, "face_outliers": 0}

    def find_raw_objs(self, source="ClothesNetM"):
        """Find all OBJ files in source dataset."""
        objs = []
        for p in (DATA_ROOT / source).glob("**/*.obj"):
            if p.name != "border.obj":
                objs.append(p)
        return sorted(objs)

    def check_quad_quality(self, obj_path):
        """Inspect a single OBJ: quad ratio, face count, boundary edges."""
        mesh = trimesh.load(str(obj_path), force='mesh', process=False)
        faces = mesh.faces
        quads = sum(1 for f in faces if len(f) == 4)
        tris = sum(1 for f in faces if len(f) == 3)
        total = quads + tris
        quad_ratio = 100 * quads / max(1, total)

        # Boundary edges: edges appearing in only 1 face → garment openings
        edge_counts = Counter()
        for f in faces:
            n = len(f)
            for i in range(n):
                e = tuple(sorted([f[i], f[(i+1) % n]]))
                edge_counts[e] += 1
        boundary_edges = sum(1 for v in edge_counts.values() if v == 1)

        # Quality assessment
        issues = []
        if quad_ratio < 80:
            issues.append(f"low_quad:{quad_ratio:.0f}%")
        if total > 5000:
            issues.append(f"high_faces:{total}")
        if total < 300:
            issues.append(f"low_faces:{total}")
        if boundary_edges > 500:
            issues.append(f"too_many_boundary:{boundary_edges}")

        quality = "OK" if not issues else "FAIL"
        return {
            "path": str(obj_path),
            "name": obj_path.parent.name if obj_path.parent.name != "Cube" else obj_path.stem,
            "vertices": len(mesh.vertices),
            "faces": total,
            "quads": quads,
            "tris": tris,
            "quad_ratio": round(quad_ratio, 1),
            "boundary_edges": boundary_edges,
            "issues": issues,
            "quality": quality,
        }

    def extract_boundary_vertices(self, obj_path):
        """Extract boundary vertex indices (garment openings)."""
        mesh = trimesh.load(str(obj_path), force='mesh', process=False)
        edge_counts = Counter()
        for f in mesh.faces:
            n = len(f)
            for i in range(n):
                e = tuple(sorted([f[i], f[(i+1) % n]]))
                edge_counts[e] += 1
        # Boundary edges appear only once
        boundary_verts = set()
        for e, count in edge_counts.items():
            if count == 1:
                boundary_verts.add(e[0])
                boundary_verts.add(e[1])
        return list(boundary_verts)

    def extract_keypoints(self, obj_path):
        """Extract keypoints from corresponding kp_*.pcd file."""
        pcd_path = obj_path.parent / f"kp_{obj_path.parent.name}.pcd"
        if not pcd_path.exists():
            return None
        pts = []
        with open(str(pcd_path), 'r') as f:
            for line in f:
                if line.startswith('#') or line.startswith('VERSION') or line.startswith('FIELDS'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
        return np.array(pts) if pts else None

    def process_failed(self, obj_path, retries=3):
        """Re-process a failed sample with QuadriFlow (Blender background)."""
        for attempt in range(retries):
            print(f"  Retry {attempt+1}/{retries} for {obj_path.name}...")
            cmd = [
                BLENDER, "--background",
                "--python", str(PROCESS_SCRIPT), "--",
                str(obj_path), str(self.output_dir),
                "--target_faces", "2500",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                if "RESULT ok" in result.stdout:
                    return True
            except subprocess.TimeoutExpired:
                continue
        return False

    def run_full_check(self, source="ClothesNetM", max_samples=None):
        """Run complete quality check on all samples."""
        objs = self.find_raw_objs(source)
        if max_samples:
            objs = objs[:max_samples]

        print(f"BPT_Data_check: Scanning {len(objs)} samples from {source}...")
        results = {"OK": [], "FAIL": [], "BOUNDARY_ISSUE": []}

        for obj in objs:
            q = self.check_quad_quality(obj)
            self.report["total"] += 1

            if q["quality"] == "OK":
                results["OK"].append(q)
                self.report["ok"] += 1
            else:
                results["FAIL"].append(q)
                self.report["failed"] += 1
                if any("boundary" in i for i in q["issues"]):
                    results["BOUNDARY_ISSUE"].append(q)
                    self.report["boundary_issues"] += 1
                if any("high_faces" in i for i in q["issues"]):
                    self.report["face_outliers"] += 1

        # Summary
        print(f"\n=== Quality Report ===")
        print(f"  Total: {self.report['total']}")
        print(f"  OK: {self.report['ok']} ({100*self.report['ok']/max(1,self.report['total']):.0f}%)")
        print(f"  Failed: {self.report['failed']}")
        print(f"  Boundary issues: {self.report['boundary_issues']}")
        print(f"  Face outliers: {self.report['face_outliers']}")

        # Save report
        report_path = self.output_dir / "quality_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                "summary": self.report,
                "ok_samples": [r["name"] for r in results["OK"]],
                "failed_samples": [{"name": r["name"], "issues": r["issues"]} for r in results["FAIL"]],
                "boundary_issues": [{"name": r["name"], "boundary_edges": r["boundary_edges"]} for r in results["BOUNDARY_ISSUE"]],
            }, f, indent=2)
        print(f"Report saved: {report_path}")
        return results

    def generate_training_data(self, obj_path):
        """Generate supervised_data entry from a clean OBJ."""
        # This would call process_one_model.py in Blender
        # Includes: QuadriFlow remesh → point cloud sampling → noise injection
        pass


if __name__ == "__main__":
    checker = BPTDataChecker()
    checker.run_full_check(source="Other_clothes", max_samples=20)
