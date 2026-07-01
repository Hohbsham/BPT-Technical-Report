"""
Process a single OBJ model: remesh -> damage -> export.
Called by batch_supervised_pipeline.py as a subprocess.

Usage:
  blender --background --python process_one_model.py -- <model_path> <output_dir> [--target_faces N] [--damage_level LEVEL]

Damage approach (NEW):
  - GT: clean quad mesh (2000-3000 faces)
  - Input: sparse point cloud sampled from mesh + edge noise (simulates bad scanning)
  - No more hole punching / vertex noise everywhere
"""
import bpy
import sys
import os
import random
import math
import argparse
from pathlib import Path
from mathutils import Vector

DAMAGE_PRESETS = {
    "light":    {"pc_samples": 2048, "pc_noise": 0.005, "edge_noise_extra": 0.01},
    "medium":   {"pc_samples": 2048, "pc_noise": 0.010, "edge_noise_extra": 0.02},
    "heavy":    {"pc_samples": 1024, "pc_noise": 0.020, "edge_noise_extra": 0.03},
}

# Default 2500 faces (midpoint of 2000-3000)
DEFAULT_TARGET_FACES = 2500


def import_obj(filepath):
    bpy.ops.wm.obj_import(filepath=str(filepath))
    objs = bpy.context.selected_objects
    if not objs:
        raise RuntimeError(f"No objects imported from {filepath}")
    obj = objs[0]
    obj.name = Path(filepath).stem
    return obj


def clean_scene():
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def set_active(obj):
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.view_layer.objects:
        o.select_set(o == obj)


def quadriflow_remesh(obj, target_faces=DEFAULT_TARGET_FACES):
    set_active(obj)
    bpy.ops.object.quadriflow_remesh(target_faces=target_faces)


def get_bbox(obj):
    bbox = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    mins = Vector([min(v[i] for v in bbox) for i in range(3)])
    maxs = Vector([max(v[i] for v in bbox) for i in range(3)])
    center = (mins + maxs) / 2
    size = (maxs - mins).length
    return mins, maxs, center, size


def get_boundary_vertex_indices(obj):
    """Return set of vertex indices that lie on boundary edges."""
    import bmesh
    set_active(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    boundary = set()
    for e in bm.edges:
        if len(e.link_faces) == 1:
            boundary.add(e.verts[0].index)
            boundary.add(e.verts[1].index)
    bpy.ops.object.mode_set(mode='OBJECT')
    return boundary


def add_edge_vertex_noise(obj, noise_scale):
    """Jitter only boundary/edge vertices to simulate bad edge reconstruction."""
    _, _, _, size = get_bbox(obj)
    sigma = size * noise_scale
    boundary = get_boundary_vertex_indices(obj)
    mesh = obj.data
    count = 0
    for i, vert in enumerate(mesh.vertices):
        if i in boundary:
            vert.co += Vector((
                random.gauss(0, sigma),
                random.gauss(0, sigma),
                random.gauss(0, sigma),
            ))
            count += 1
    mesh.update()
    print(f"  Edge noise: {count} boundary verts jittered (sigma={sigma:.4f})")


def export_obj(obj, filepath):
    set_active(obj)
    filepath = str(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bpy.ops.wm.obj_export(filepath=filepath, export_materials=False,
                           export_uv=False, export_normals=True)


def export_sparse_pointcloud(obj, filepath, num_points=2048, noise_scale=0.01,
                             edge_noise_extra=0.02):
    """
    Sample sparse point cloud from mesh surface with noise.
    Edge-region points get extra noise to simulate scanning artifacts.
    """
    import bmesh
    filepath = str(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    _, _, _, size = get_bbox(obj)
    base_sigma = size * noise_scale
    edge_sigma = size * edge_noise_extra

    # Get boundary vertices
    boundary = get_boundary_vertex_indices(obj)

    # Sample faces by area to get surface points
    set_active(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Build face area CDF for weighted sampling
    faces = list(bm.faces)
    areas = [f.calc_area() for f in faces]
    total_area = sum(areas)
    if total_area == 0:
        areas = [1.0 / len(faces)] * len(faces)
        total_area = 1.0

    # Sample points from faces (proportional to area)
    verts_out = []
    for _ in range(num_points):
        # Pick face by area-weighted random
        r = random.random() * total_area
        cum = 0
        face = faces[-1]
        for f, a in zip(faces, areas):
            cum += a
            if cum >= r:
                face = f
                break

        # Random barycentric point on face
        u = random.random()
        v = random.random() * (1 - u)
        w = 1 - u - v
        fv = face.verts
        pt = fv[0].co * u + fv[1].co * v + fv[2].co * w
        if len(fv) == 4:
            # Quad: two triangles, pick one
            u2 = random.random()
            v2 = random.random() * (1 - u2)
            w2 = 1 - u2 - v2
            if random.random() < 0.5:
                pt = fv[0].co * u2 + fv[1].co * v2 + fv[2].co * w2
            else:
                pt = fv[0].co * u2 + fv[2].co * v2 + fv[3].co * w2

        # Check if this point is near a boundary vertex
        near_boundary = any(
            v.index in boundary for v in face.verts
        )

        # Add noise (more for boundary-adjacent points)
        sigma = edge_sigma if near_boundary else base_sigma
        pt += Vector((
            random.gauss(0, sigma),
            random.gauss(0, sigma),
            random.gauss(0, sigma),
        ))

        # Transform to world
        pt = obj.matrix_world @ pt
        verts_out.append(pt)

    bpy.ops.object.mode_set(mode='OBJECT')

    # Write PLY
    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts_out)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for v in verts_out:
            f.write(f"{v.x:.6f} {v.y:.6f} {v.z:.6f}\n")

    n_edge = sum(1 for v in verts_out if True)  # approximate
    print(f"  Point cloud: {len(verts_out)} pts (noise: base={base_sigma:.4f}, edge={edge_sigma:.4f})")


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("output_dir")
    parser.add_argument("--target_faces", type=int, default=DEFAULT_TARGET_FACES)
    parser.add_argument("--damage_level", default="medium")
    args = parser.parse_args(argv)

    damage_cfg = DAMAGE_PRESETS[args.damage_level]
    model_path = Path(args.model_path)
    model_name = model_path.parent.name
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = out_dir / "ground_truth.obj"
    pc_path = out_dir / "input_pointcloud.ply"

    print(f"MODEL: {model_name}")
    clean_scene()

    # 1. Import
    print("[1/4] Import...")
    obj = import_obj(model_path)
    print(f"  {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces")

    # 2. QuadriFlow remesh -> clean quad GT
    print(f"[2/4] QuadriFlow (target={args.target_faces})...")
    quadriflow_remesh(obj, args.target_faces)
    quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
    print(f"  {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces, "
          f"{100*quads/max(1,len(obj.data.polygons)):.1f}% quads")

    # 3. Save GT (clean quad mesh)
    print("[3/4] Save GT...")
    export_obj(obj, gt_path)

    # 4. Generate input: sparse point cloud with noise (edge-heavy)
    print("[4/4] Generate input point cloud...")
    export_sparse_pointcloud(
        obj, pc_path,
        num_points=damage_cfg["pc_samples"],
        noise_scale=damage_cfg["pc_noise"],
        edge_noise_extra=damage_cfg["edge_noise_extra"],
    )

    # Also save a damaged OBJ: edge-perturbed version of GT
    print("[4/4 bonus] Save edge-damaged OBJ...")
    add_edge_vertex_noise(obj, damage_cfg["edge_noise_extra"])
    dam_path = out_dir / "input_damaged.obj"
    export_obj(obj, dam_path)

    print("DONE")


if __name__ == "__main__":
    main()
