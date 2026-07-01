"""
Supervised Data Pipeline for Cloth Point Cloud Regularizer.

For each OBJ in ClothesNetM:
  1. Import OBJ
  2. Run QuadriFlow remesh → GROUND TRUTH (clean quad mesh)
  3. Generate damaged version → INPUT (noise + holes + edge erosion)
  4. Export (input_damaged.obj, ground_truth.obj, input_pointcloud.ply)

Usage:
  blender --background --python supervised_data_pipeline.py -- [options]

Options:
  --data_dir     Path to ClothesNetM data (default: D:/ClothesNetData/ClothesNetM)
  --output_dir   Output directory (default: D:/ClothesNetData/supervised_data)
  --target_faces Target face count for remeshing (default: 8000)
  --category     Process single category (e.g. "Dress"), or "all" (default: all)
  --damage_level Light/medium/heavy damage preset (default: medium)
  --dry_run      Print what would be done without processing
"""

import bpy
import sys
import os
import argparse
import random
import math
from pathlib import Path
from mathutils import Vector, noise

# ============================================================
# Configuration
# ============================================================

# Models known to hang QuadriFlow — use original mesh as GT instead
QUADRIFLOW_BLACKLIST = {
    "DLG_Dress105",  # QuadriFlow hangs indefinitely on this mesh
}


DAMAGE_PRESETS = {
    "light": {
        "noise_scale": 0.005,   # 0.5% of bbox
        "hole_ratio": 0.05,     # 5% faces removed
        "edge_erosion": 0.05,   # 5% boundary erosion
    },
    "medium": {
        "noise_scale": 0.01,    # 1%
        "hole_ratio": 0.10,     # 10%
        "edge_erosion": 0.10,   # 10%
    },
    "heavy": {
        "noise_scale": 0.02,    # 2%
        "hole_ratio": 0.20,     # 20%
        "edge_erosion": 0.15,   # 15%
    },
}


# ============================================================
# Mesh Processing Functions
# ============================================================

def import_obj(filepath):
    """Import OBJ file into Blender, return the imported object."""
    bpy.ops.wm.obj_import(filepath=str(filepath))
    objs = bpy.context.selected_objects
    if not objs:
        raise RuntimeError(f"No objects imported from {filepath}")
    obj = objs[0]
    obj.name = Path(filepath).stem
    return obj


def clean_scene():
    """Remove all mesh objects from scene."""
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    # Clean orphan data
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)


def set_active(obj):
    """Set object as active and selected."""
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(o == obj)


def quadriflow_remesh(obj, target_faces=8000):
    """Run QuadriFlow remeshing on the object. Returns success bool."""
    set_active(obj)
    try:
        bpy.ops.object.quadriflow_remesh(target_faces=target_faces)
        return True
    except Exception as e:
        print(f"  QuadriFlow failed: {e}")
        return False


def get_bbox(obj):
    """Return (min_corner, max_corner, center, size) for object."""
    bbox = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    mins = Vector([min(v[i] for v in bbox) for i in range(3)])
    maxs = Vector([max(v[i] for v in bbox) for i in range(3)])
    center = (mins + maxs) / 2
    size = (maxs - mins).length
    return mins, maxs, center, size


def add_vertex_noise(obj, noise_scale):
    """Add Gaussian noise to vertices."""
    _, _, _, size = get_bbox(obj)
    sigma = size * noise_scale

    mesh = obj.data
    for vert in mesh.vertices:
        offset = Vector((
            random.gauss(0, sigma),
            random.gauss(0, sigma),
            random.gauss(0, sigma),
        ))
        vert.co += offset

    mesh.update()
    print(f"  Added noise: sigma={sigma:.4f}")
    return obj


def remove_random_faces(obj, ratio):
    """Randomly delete a ratio of faces."""
    import bmesh

    set_active(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)

    faces_to_remove = []
    for face in bm.faces:
        if random.random() < ratio:
            faces_to_remove.append(face)

    for face in faces_to_remove:
        bm.faces.remove(face)

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    n_faces = len(obj.data.polygons)
    print(f"  Removed {len(faces_to_remove)} faces ({ratio*100:.0f}%), remaining: {n_faces}")
    return obj


def erode_boundary_edges(obj, ratio):
    """Delete boundary edges to simulate ripped edges."""
    import bmesh

    set_active(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()

    # Find boundary edges (edges with only 1 face)
    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]

    # Select a ratio of them to remove
    n_remove = max(1, int(len(boundary_edges) * ratio))
    edges_to_remove = random.sample(boundary_edges, min(n_remove, len(boundary_edges)))

    faces_to_remove = set()
    for edge in edges_to_remove:
        for face in edge.link_faces:
            faces_to_remove.add(face)

    for face in faces_to_remove:
        bm.faces.remove(face)

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"  Eroded {len(edges_to_remove)} boundary edges, removed {len(faces_to_remove)} faces")
    return obj


def generate_damaged_version(obj, damage_cfg):
    """Apply damage to create input data. Returns modified obj."""
    # 1. Add vertex noise
    add_vertex_noise(obj, damage_cfg["noise_scale"])

    # 2. Remove random faces (holes)
    if damage_cfg["hole_ratio"] > 0:
        remove_random_faces(obj, damage_cfg["hole_ratio"])

    # 3. Erode boundary edges
    if damage_cfg["edge_erosion"] > 0:
        erode_boundary_edges(obj, damage_cfg["edge_erosion"])

    return obj


def export_obj(obj, filepath):
    """Export object as OBJ."""
    set_active(obj)
    filepath = str(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_materials=False,
        export_uv=False,
        export_normals=True,
    )
    print(f"  Exported: {filepath}")


def export_pointcloud(obj, filepath, num_points=4096):
    """Export object vertices as PLY point cloud."""
    filepath = str(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if len(verts) > num_points:
        verts = random.sample(verts, num_points)

    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v.x:.6f} {v.y:.6f} {v.z:.6f}\n")

    print(f"  Exported point cloud: {filepath} ({len(verts)} points)")


# ============================================================
# Main Pipeline
# ============================================================

def process_single_model(obj_path, output_dir, target_faces, damage_cfg, resume=False):
    """Process a single OBJ file: remesh → damage → export pair."""
    model_name = Path(obj_path).parent.name
    out_dir = Path(output_dir) / model_name

    # Skip if already processed
    gt_path = out_dir / "ground_truth.obj"
    dam_path = out_dir / "input_damaged.obj"
    pc_path = out_dir / "input_pointcloud.ply"
    if resume and gt_path.exists() and dam_path.exists() and pc_path.exists():
        print(f"  SKIP: {model_name} already processed")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"  Source: {obj_path}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    clean_scene()

    # Step 1: Import
    print("\n[1/5] Importing...")
    obj = import_obj(obj_path)
    print(f"  Imported: {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces")

    # Step 2: QuadriFlow remesh → GT
    print(f"\n[2/5] QuadriFlow remesh (target={target_faces})...")
    if model_name in QUADRIFLOW_BLACKLIST:
        print(f"  SKIP: {model_name} is blacklisted, using original mesh as GT")
    elif not quadriflow_remesh(obj, target_faces):
        print("  WARNING: QuadriFlow failed, using original mesh as GT")
        clean_scene()
        obj = import_obj(obj_path)

    quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
    n_polys = len(obj.data.polygons)
    if n_polys > 0:
        print(f"  Result: {len(obj.data.vertices)} verts, {n_polys} faces, "
              f"{100*quads/n_polys:.1f}% quads")
    else:
        print(f"  Result: {len(obj.data.vertices)} verts, {n_polys} faces")

    # Step 3: Save GT
    print(f"\n[3/5] Saving ground truth...")
    export_obj(obj, gt_path)

    # Step 4: Generate damaged version
    print(f"\n[4/5] Generating damaged version...")
    generate_damaged_version(obj, damage_cfg)
    export_obj(obj, dam_path)

    # Step 5: Export point cloud
    print(f"\n[5/5] Exporting point cloud...")
    export_pointcloud(obj, pc_path)

    return True


def find_all_objs(data_dir, category="all"):
    """Find all OBJ files in the dataset."""
    data_dir = Path(data_dir)
    objs = []

    if category == "all":
        pattern = "**/*.obj"
    else:
        pattern = f"{category}/**/*.obj"

    for obj_path in data_dir.glob(pattern):
        # Skip border.obj files
        if obj_path.name == "border.obj":
            continue
        objs.append(obj_path)

    return sorted(objs)


def parse_args():
    """Parse command line arguments after --."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Supervised Data Pipeline")
    parser.add_argument("--data_dir", default=r"D:/ClothesNetData/ClothesNetM")
    parser.add_argument("--output_dir", default=r"D:/ClothesNetData/supervised_data")
    parser.add_argument("--target_faces", type=int, default=8000)
    parser.add_argument("--category", default="all")
    parser.add_argument("--damage_level", default="medium",
                        choices=["light", "medium", "heavy"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of models to process")
    parser.add_argument("--resume", action="store_true",
                        help="Skip models that already have output files")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    damage_cfg = DAMAGE_PRESETS[args.damage_level]

    print("=" * 60)
    print("SUPERVISED DATA PIPELINE")
    print("=" * 60)
    print(f"Data dir:     {args.data_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Target faces: {args.target_faces}")
    print(f"Damage level: {args.damage_level} (noise={damage_cfg['noise_scale']}, "
          f"holes={damage_cfg['hole_ratio']}, erosion={damage_cfg['edge_erosion']})")
    print(f"Category:     {args.category}")

    # Find models
    obj_paths = find_all_objs(args.data_dir, args.category)
    print(f"\nFound {len(obj_paths)} OBJ files")

    if args.limit > 0:
        obj_paths = obj_paths[:args.limit]
        print(f"Limited to {args.limit} models")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for p in obj_paths:
            print(f"  {p.relative_to(args.data_dir)}")
        return

    # Process each model
    success = 0
    failed = 0
    skipped = 0
    failed_models = []
    for i, obj_path in enumerate(obj_paths):
        print(f"\n[{i+1}/{len(obj_paths)}]")
        try:
            result = process_single_model(obj_path, args.output_dir,
                                          args.target_faces, damage_cfg,
                                          resume=args.resume)
            if result:
                success += 1
            else:
                failed += 1
                failed_models.append(str(obj_path))
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            failed_models.append(f"{obj_path}: {e}")

    print(f"\n{'='*60}")
    print(f"DONE: {success} succeeded, {skipped} skipped, {failed} failed")
    if failed_models:
        print(f"Failed models:")
        for m in failed_models:
            print(f"  - {m}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
