"""
Blender background script: re-remesh a single ground_truth.obj to target face count.
Regenerates input_pointcloud.ply from the new mesh.

Usage:
  blender --background --python remesh_to_target.py -- <sample_dir> <target_faces>
"""
import bpy
import sys
import os
import random
import math
import argparse
from pathlib import Path
from mathutils import Vector

DAMAGE_CFG = {"pc_samples": 2048, "pc_noise": 0.010, "edge_noise_extra": 0.02}


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


def quadriflow_remesh(obj, target_faces):
    set_active(obj)
    bpy.ops.object.quadriflow_remesh(target_faces=target_faces)


def get_bbox(obj):
    bbox = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    mins = Vector([min(v[i] for v in bbox) for i in range(3)])
    maxs = Vector([max(v[i] for v in bbox) for i in range(3)])
    size = (maxs - mins).length
    return size


def get_boundary_vertex_indices(obj):
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


def export_obj(obj, filepath):
    set_active(obj)
    filepath = str(filepath)
    bpy.ops.wm.obj_export(filepath=filepath, export_materials=False,
                           export_uv=False, export_normals=True)


def export_sparse_pointcloud(obj, filepath, num_points=2048, noise_scale=0.01,
                             edge_noise_extra=0.02):
    import bmesh
    filepath = str(filepath)

    size = get_bbox(obj)
    base_sigma = size * noise_scale
    edge_sigma = size * edge_noise_extra

    boundary = get_boundary_vertex_indices(obj)

    set_active(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    faces = list(bm.faces)
    areas = [f.calc_area() for f in faces]
    total_area = sum(areas)
    if total_area == 0:
        areas = [1.0 / len(faces)] * len(faces)
        total_area = 1.0

    verts_out = []
    for _ in range(num_points):
        r = random.random() * total_area
        cum = 0
        face = faces[-1]
        for f, a in zip(faces, areas):
            cum += a
            if cum >= r:
                face = f
                break

        fv = face.verts
        u = random.random()
        v = random.random() * (1 - u)
        w = 1 - u - v
        pt = fv[0].co * u + fv[1].co * v + fv[2].co * w
        if len(fv) == 4:
            u2 = random.random()
            v2 = random.random() * (1 - u2)
            w2 = 1 - u2 - v2
            if random.random() < 0.5:
                pt = fv[0].co * u2 + fv[1].co * v2 + fv[2].co * w2
            else:
                pt = fv[0].co * u2 + fv[2].co * v2 + fv[3].co * w2

        near_boundary = any(v.index in boundary for v in face.verts)
        sigma = edge_sigma if near_boundary else base_sigma
        pt += Vector((
            random.gauss(0, sigma),
            random.gauss(0, sigma),
            random.gauss(0, sigma),
        ))
        pt = obj.matrix_world @ pt
        verts_out.append(pt)

    bpy.ops.object.mode_set(mode='OBJECT')

    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts_out)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for v in verts_out:
            f.write(f"{v.x:.6f} {v.y:.6f} {v.z:.6f}\n")


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir", help="Path to sample directory containing ground_truth.obj")
    parser.add_argument("target_faces", type=int)
    args = parser.parse_args(argv)

    sample_dir = Path(args.sample_dir)
    gt_path = sample_dir / "ground_truth.obj"
    pc_path = sample_dir / "input_pointcloud.ply"
    temp_gt = sample_dir / "ground_truth_new.obj"

    if not gt_path.exists():
        print(f"ERROR: ground_truth.obj not found at {gt_path}")
        sys.exit(1)

    model_name = sample_dir.name
    print(f"MODEL: {model_name} | target={args.target_faces}")
    clean_scene()

    # 1. Import existing ground_truth
    print("[1/3] Import...")
    obj = import_obj(gt_path)
    old_verts = len(obj.data.vertices)
    old_faces = len(obj.data.polygons)
    print(f"  {old_verts} verts, {old_faces} faces")

    # 2. QuadriFlow remesh
    print(f"[2/3] QuadriFlow (target={args.target_faces})...")
    quadriflow_remesh(obj, args.target_faces)
    new_verts = len(obj.data.vertices)
    new_faces = len(obj.data.polygons)
    quads = sum(1 for p in obj.data.polygons if len(p.vertices) == 4)
    print(f"  {new_verts} verts, {new_faces} faces, {100*quads/max(1,new_faces):.1f}% quads")

    # 3. Save GT and point cloud
    print("[3/3] Save...")
    export_obj(obj, temp_gt)
    export_sparse_pointcloud(
        obj, pc_path,
        num_points=DAMAGE_CFG["pc_samples"],
        noise_scale=DAMAGE_CFG["pc_noise"],
        edge_noise_extra=DAMAGE_CFG["edge_noise_extra"],
    )

    # Atomic replace: only overwrite original if export succeeded
    if temp_gt.exists():
        os.replace(str(temp_gt), str(gt_path))

    print(f"DONE: {old_faces}->{new_faces} faces")
    print("RESULT ok")


if __name__ == "__main__":
    main()
