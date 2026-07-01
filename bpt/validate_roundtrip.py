"""
BPT Roundtrip Validation: mesh → tokens → mesh
Measures reconstruction fidelity on clothing quad meshes.

Tests both triangle and quad serialization modes.
"""

import os
import sys
import types
import argparse
from pathlib import Path

# ---- Windows patches ----
class _DeepSpeedLoader:
    def find_module(self, fullname, path=None):
        return self if fullname.startswith('deepspeed') else None
    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = types.ModuleType(fullname)
        mod.__path__ = []; mod.__file__ = '(deepspeed-dummy)'
        class _AutoMod(types.ModuleType):
            def __getattr__(s, name):
                if name.startswith('_'): raise AttributeError(name)
                cls = type(name, (), {
                    '__new__': lambda cls, *a, **kw: object.__new__(cls),
                    '__init__': lambda self, *a, **kw: None,
                })
                setattr(s, name, cls)
                return cls
        mod.__class__ = _AutoMod
        sys.modules[fullname] = mod
        return mod
sys.meta_path.insert(0, _DeepSpeedLoader())

import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

import numpy as np
import trimesh
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model.serializaiton import BPT_serialize, BPT_deserialize, patchified_mesh, decode_block
from model.data_utils import to_mesh, discretize, undiscretize
from utils import apply_normalize


def chamfer_distance(pts1, pts2):
    """Chamfer distance between two point sets."""
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    d1, _ = tree1.query(pts2, k=1)
    d2, _ = tree2.query(pts1, k=1)
    return np.mean(d1) + np.mean(d2)


def hausdorff_distance(pts1, pts2):
    """Hausdorff distance between two point sets."""
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    d1, _ = tree1.query(pts2, k=1)
    d2, _ = tree2.query(pts1, k=1)
    return max(np.max(d1), np.max(d2))


class _QuadMesh:
    """Minimal mesh-like object with quad faces for BPT quad serialization."""

    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self._compute_derived()

    def _compute_derived(self):
        n_verts = len(self.vertices)
        self.vertex_degree = np.zeros(n_verts, dtype=np.int32)
        self.vertex_faces = [set() for _ in range(n_verts)]
        for fi, face in enumerate(self.faces):
            for vi in face:
                if 0 <= vi < n_verts:
                    self.vertex_degree[vi] += 1
                    self.vertex_faces[vi].add(fi)
        for i in range(n_verts):
            self.vertex_faces[i] = np.array(sorted(self.vertex_faces[i]), dtype=np.int64)
        self.bounds = np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    def apply_translation(self, vec):
        self.vertices = self.vertices + np.asarray(vec)
        self.bounds = np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    def apply_scale(self, s):
        self.vertices = self.vertices * s
        self.bounds = np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    def sample(self, n):
        """Sample points on quad faces by splitting into 2 triangles."""
        tris = []
        for f in self.faces:
            if len(f) == 4:
                tris.append([f[0], f[1], f[2]])
                tris.append([f[0], f[2], f[3]])
            elif len(f) == 3:
                tris.append(f)
        if not tris:
            return self.vertices
        tris = np.array(tris, dtype=np.int64)
        tri_mesh = trimesh.Trimesh(vertices=self.vertices, faces=tris, process=False)
        return tri_mesh.sample(n)


def _load_quad_mesh(obj_path):
    """Load OBJ preserving quad faces. Returns _QuadMesh or None if not quads."""
    verts = []
    faces = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]
                faces.append(face)
    if not faces:
        return None
    face_sizes = [len(f) for f in faces]
    if sum(face_sizes) / len(face_sizes) < 3.8:
        return None  # Not a quad mesh
    verts = np.array(verts, dtype=np.float64)
    return _QuadMesh(verts, faces)


def validate_roundtrip(mesh_path, output_dir, mesh_type="triangle", visualize=True):
    """Test mesh → tokens → mesh roundtrip."""
    name = Path(mesh_path).parent.name if mesh_path.endswith('.obj') else Path(mesh_path).name
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect quad mesh from OBJ file directly
    quad_mesh = _load_quad_mesh(mesh_path)
    is_quad = quad_mesh is not None

    # Auto-detect: skip quad mode if mesh is triangles, skip triangle mode if mesh is quads
    if mesh_type == "quad" and not is_quad:
        print(f"\n  [SKIP] Mesh has triangles, can't test quad mode")
        return None
    if mesh_type == "triangle" and is_quad:
        print(f"\n  [SKIP] Mesh has quads, can't test triangle mode")
        return None

    # Load and normalize
    if mesh_type == "quad":
        mesh = quad_mesh
        n_faces_orig = len(mesh.faces)
        n_verts_orig = len(mesh.vertices)
        avg_face_size = 4.0
    else:
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        n_faces_orig = len(mesh.faces)
        n_verts_orig = len(mesh.vertices)
        face_sizes = [len(f) for f in mesh.faces]
        avg_face_size = sum(face_sizes) / max(len(face_sizes), 1)

    mesh = apply_normalize(mesh)

    print(f"\n{'='*60}")
    print(f"Mesh: {name}")
    print(f"  Faces: {n_faces_orig} (avg {avg_face_size:.1f} verts/face → {mesh_type})")
    print(f"  Vertices: {n_verts_orig}")

    # Step 1: patchify (vertex sequence with special tokens)
    patch_seq = patchified_mesh(mesh, special_token=-2, mesh_type=mesh_type)
    n_special = (patch_seq == -2).all(axis=-1).sum()
    n_verts_in_seq = len(patch_seq) - n_special
    print(f"\n[1] Patchified: {len(patch_seq)} entries ({n_verts_in_seq} verts, {n_special} special)")

    # Step 2: block/offset encoding
    codes = BPT_serialize(mesh, mesh_type=mesh_type)
    n_tokens = len(codes)
    print(f"[2] BPT tokens: {n_tokens} tokens")

    # Step 3: deserialize
    recon_verts = BPT_deserialize(codes, mesh_type=mesh_type)
    n_recon_verts = len(recon_verts)
    print(f"[3] Reconstructed vertices: {n_recon_verts}")

    # Step 4: build faces and mesh
    face_width = 4 if mesh_type == "quad" else 3
    n_faces_recon = n_recon_verts // face_width

    # Only use complete faces
    usable_verts = (n_recon_verts // face_width) * face_width
    recon_verts_trimmed = recon_verts[:usable_verts]

    if mesh_type == "quad":
        faces = []
        for j in range(0, usable_verts, 4):
            faces.append([j, j+1, j+2, j+3])
    else:
        faces = []
        for j in range(0, usable_verts, 3):
            faces.append([j, j+1, j+2])

    # Remove NaN faces
    faces = [f for f in faces if not np.isnan(recon_verts_trimmed[f]).any()]

    faces = np.array(faces, dtype=np.int32)

    if len(faces) == 0:
        print("[FAIL] No valid faces in reconstruction!")
        return None

    recon_mesh = trimesh.Trimesh(vertices=recon_verts_trimmed, faces=faces, process=False)

    # Step 5: sample points from both meshes for comparison
    try:
        orig_pts = mesh.sample(min(10000, n_verts_orig * 10))
        recon_pts = recon_mesh.sample(min(10000, n_recon_verts * 10))
    except Exception:
        orig_pts = mesh.vertices
        recon_pts = recon_mesh.vertices[:min(len(orig_pts), len(recon_verts))]

    cd = chamfer_distance(orig_pts, recon_pts)
    hd = hausdorff_distance(orig_pts, recon_pts)

    # Step 6: face coverage
    face_coverage = n_faces_recon / n_faces_orig * 100
    vert_coverage = usable_verts / n_faces_orig / face_width * 100

    # Bounding box size for normalization
    bbox_size = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

    print(f"\n{'─'*40}")
    print(f"RESULTS ({mesh_type} mode):")
    print(f"  Reconstruction faces:  {n_faces_recon} (original: {n_faces_orig})")
    print(f"  Face coverage:         {face_coverage:.1f}%")
    print(f"  Chamfer distance:      {cd:.6f} ({cd/bbox_size*100:.3f}% of bbox)")
    print(f"  Hausdorff distance:    {hd:.6f} ({hd/bbox_size*100:.3f}% of bbox)")
    print(f"  Token compression:     {n_tokens} tokens for {n_faces_orig} faces ({n_tokens/n_faces_orig:.1f} tokens/face)")

    # Quality verdict
    if cd / bbox_size < 0.01:
        quality = "EXCELLENT"
    elif cd / bbox_size < 0.05:
        quality = "GOOD"
    elif cd / bbox_size < 0.10:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"
    print(f"  Quality:               {quality}")

    if visualize:
        orig_path = str(out_dir / f"{name}_original.obj")
        recon_path = str(out_dir / f"{name}_reconstructed_{mesh_type}.obj")
        if mesh_type == "quad":
            # Triangulate quad mesh for OBJ export
            tri_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=np.array([(f[0], f[1], f[2]) for f in mesh.faces]
                               + [(f[0], f[2], f[3]) for f in mesh.faces], dtype=np.int64),
                process=False)
            tri_mesh.export(orig_path)
        else:
            mesh.export(orig_path)
        recon_mesh.export(recon_path)
        print(f"  Saved: {orig_path}")
        print(f"  Saved: {recon_path}")

    return {
        "name": name,
        "mesh_type": mesh_type,
        "n_faces_orig": n_faces_orig,
        "n_verts_orig": n_verts_orig,
        "n_tokens": n_tokens,
        "n_faces_recon": n_faces_recon,
        "face_coverage": face_coverage,
        "chamfer": cd,
        "hausdorff": hd,
        "chamfer_pct": cd / bbox_size * 100,
        "hausdorff_pct": hd / bbox_size * 100,
        "quality": quality,
        "avg_face_size": avg_face_size,
        "n_special": n_special,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate BPT roundtrip on clothing meshes")
    parser.add_argument("--data_dir", type=str, default=r"D:\ClothesNetData\supervised_data")
    parser.add_argument("--output_dir", type=str, default="./validation_results")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of random samples to validate")
    parser.add_argument("--mesh_path", type=str, default=None,
                        help="Validate a single mesh file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.mesh_path:
        validate_roundtrip(args.mesh_path, args.output_dir, mesh_type="triangle")
        validate_roundtrip(args.mesh_path, args.output_dir, mesh_type="quad")
        return

    # Find samples
    data_dir = Path(args.data_dir)
    samples = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name == "Cube":
            continue
        gt = d / "ground_truth.obj"
        if gt.exists():
            samples.append(gt)

    # Select diverse samples by face count
    sample_info = []
    for s in samples:
        m = trimesh.load(str(s), force='mesh', process=False)
        sample_info.append((s, len(m.faces)))

    sample_info.sort(key=lambda x: x[1])

    # Pick min, median, max, and random
    selected = [sample_info[0], sample_info[len(sample_info)//2], sample_info[-1]]
    rng = np.random.default_rng(args.seed)
    rest = [s for s in sample_info if s not in selected]
    selected.extend(rng.choice(rest, min(args.n_samples - 3, len(rest)), replace=False).tolist())

    print(f"Validating {len(selected)} samples...")

    all_results = []
    for mesh_path, n_faces in tqdm(selected, desc="Validating"):
        # Test both modes on first 3, triangle only on rest
        for mode in ["triangle", "quad"]:
            if mode == "quad" and len(all_results) > 6:
                continue  # Only test quad on a few samples (slow)
            result = validate_roundtrip(str(mesh_path), args.output_dir, mesh_type=mode)
            if result:
                all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(all_results)} validations)")
    print(f"{'='*60}")

    tri_results = [r for r in all_results if r["mesh_type"] == "triangle"]
    quad_results = [r for r in all_results if r["mesh_type"] == "quad"]

    for label, results in [("Triangle", tri_results), ("Quad", quad_results)]:
        if not results:
            continue
        cd_vals = [r["chamfer_pct"] for r in results]
        fc_vals = [r["face_coverage"] for r in results]
        tok_vals = [r["n_tokens"] / r["n_faces_orig"] for r in results]
        print(f"\n{label} mode ({len(results)} samples):")
        print(f"  Chamfer (%bbox):  mean={np.mean(cd_vals):.3f}  min={np.min(cd_vals):.3f}  max={np.max(cd_vals):.3f}")
        print(f"  Face coverage:    mean={np.mean(fc_vals):.1f}%  min={np.min(fc_vals):.1f}%  max={np.max(fc_vals):.1f}%")
        print(f"  Tokens/face:      mean={np.mean(tok_vals):.1f}  min={np.min(tok_vals):.1f}  max={np.max(tok_vals):.1f}")
        qualities = [r["quality"] for r in results]
        for q in ["EXCELLENT", "GOOD", "ACCEPTABLE", "POOR"]:
            if qualities.count(q) > 0:
                print(f"  {q}: {qualities.count(q)}")


if __name__ == "__main__":
    main()
