"""
Semi-automated QuadRemesher pipeline for models where QuadriFlow hangs.

Workflow per model:
  1. Click "Load Model 1" (or auto-loads after each confirm)
  2. Click "REMESH IT" in QuadRemesher panel (N > Quad Remesh tab)
  3. Click "CONFIRM & Export" -> exports GT + damaged + pointcloud, loads next
  4. Repeat

Models to process (14 timeout):
"""
import bpy
import os
import random
from pathlib import Path
from mathutils import Vector

DATA_DIR = r"D:\ClothesNetData\ClothesNetM"
OUT_DIR = r"D:\ClothesNetData\supervised_data"

TIMEOUT_MODELS = [
    "Dress/Long_Gallus/DLG_Dress315/DLG_Dress315.obj",
    "Dress/Long_Tube/DLT_Dress096/DLT_Dress096.obj",
    "Dress/Long_Tube/DLT_Dress243/DLT_Dress243.obj",
    "Hat/HA_hat039/HA_hat039.obj",
    "Socks/Long/SOL_Socks027/SOL_Socks027.obj",
]

DAMAGE_CFG = {"pc_samples": 2048, "pc_noise": 0.01, "edge_noise_extra": 0.02}

_qr_state = {}


# ---- Operators ----

class QR_LOAD_OT(bpy.types.Operator):
    """Load first (or specified) model"""
    bl_idname = "wm.qr_load"
    bl_label = "QR: Load Model"

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        _load_model(self.index)
        return {'FINISHED'}


class QR_CONFIRM_OT(bpy.types.Operator):
    """Export GT + damaged + pointcloud, then load next model"""
    bl_idname = "wm.qr_confirm"
    bl_label = "QR: Confirm & Export"

    def execute(self, context):
        state = _qr_state
        out_dir = state.get("out_dir")
        model_name = state.get("model_name", "")
        model_idx = state.get("model_idx", 0)

        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "No active mesh. Click the remeshed model.")
            return {'CANCELLED'}

        gt_path = str(out_dir / "ground_truth.obj")
        dam_path = str(out_dir / "input_damaged.obj")
        pc_path = str(out_dir / "input_pointcloud.ply")

        print(f"\n  [Export] GT: {len(obj.data.vertices)}v, {len(obj.data.polygons)}f")
        bpy.ops.wm.obj_export(filepath=gt_path, export_materials=False,
                               export_uv=False, export_normals=True)

        print(f"  [Damage] edge noise + sparse point cloud...")
        _add_edge_noise(obj, DAMAGE_CFG["edge_noise_extra"])
        print(f"  [Export] Edge-damaged: {len(obj.data.vertices)}v, {len(obj.data.polygons)}f")
        bpy.ops.wm.obj_export(filepath=dam_path, export_materials=False,
                               export_uv=False, export_normals=True)

        print(f"  [Export] Sparse point cloud...")
        _export_sparse_pc(obj, pc_path, DAMAGE_CFG["pc_samples"],
                          DAMAGE_CFG["pc_noise"], DAMAGE_CFG["edge_noise_extra"])
        print(f"  DONE! {model_name}")

        # Load next
        next_idx = model_idx + 1
        if next_idx >= len(TIMEOUT_MODELS):
            print("\n" + "=" * 60)
            print("ALL 14 MODELS PROCESSED!")
            print("=" * 60)
            _qr_state["model_idx"] = next_idx
            self.report({'INFO'}, "ALL 14 DONE!")
            return {'FINISHED'}

        _load_model(next_idx)
        return {'FINISHED'}


class QR_SKIP_OT(bpy.types.Operator):
    """Skip current model, load next"""
    bl_idname = "wm.qr_skip"
    bl_label = "QR: Skip & Next"

    def execute(self, context):
        state = _qr_state
        model_idx = state.get("model_idx", 0)
        model_name = state.get("model_name", "")
        print(f"  SKIPPED {model_name}")

        next_idx = model_idx + 1
        if next_idx >= len(TIMEOUT_MODELS):
            print("\nALL 14 PROCESSED!")
            _qr_state["model_idx"] = next_idx
            return {'FINISHED'}
        _load_model(next_idx)
        return {'FINISHED'}


# ---- Panel ----

class QR_PANEL(bpy.types.Panel):
    """QuadRemesher manual pipeline panel"""
    bl_label = "QR Manual Pipeline"
    bl_idname = "VIEW3D_PT_qr_pipeline"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QR Pipeline"

    def draw(self, context):
        layout = self.layout
        idx = _qr_state.get("model_idx", 0)
        name = _qr_state.get("model_name", "")

        if idx >= len(TIMEOUT_MODELS):
            layout.label(text="ALL DONE!", icon='CHECKMARK')
            return

        layout.label(text=f"Progress: {idx}/{len(TIMEOUT_MODELS)}")
        if name:
            layout.label(text=f"Current: {name}", icon='MESH_DATA')

        layout.separator()

        # Big confirm button
        row = layout.row(align=True)
        row.scale_y = 2.0
        row.operator("wm.qr_confirm", text="CONFIRM & EXPORT", icon='EXPORT')

        layout.separator()

        row = layout.row(align=True)
        row.operator("wm.qr_skip", text="Skip & Next", icon='NEXT_KEYFRAME')
        row.operator("wm.qr_load", text="Reload Current", icon='FILE_REFRESH').index = idx

        layout.separator()
        box = layout.box()
        box.label(text="Workflow:", icon='INFO')
        box.label(text="1. Click REMESH IT")
        box.label(text="   (Quad Remesh tab)")
        box.label(text="2. Click CONFIRM above")


# ---- Internal helpers ----

def _get_bbox_size(obj):
    bbox = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    mins = Vector([min(v[i] for v in bbox) for i in range(3)])
    maxs = Vector([max(v[i] for v in bbox) for i in range(3)])
    return (maxs - mins).length


def _add_edge_noise(obj, noise_scale):
    """Jitter only boundary/edge vertices to simulate bad edge reconstruction."""
    import bmesh as _bmesh
    sigma = _get_bbox_size(obj) * noise_scale

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = _bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    boundary = {e.verts[0].index for e in bm.edges if len(e.link_faces) == 1}
    boundary.update(e.verts[1].index for e in bm.edges if len(e.link_faces) == 1)
    bpy.ops.object.mode_set(mode='OBJECT')

    count = 0
    for i in boundary:
        obj.data.vertices[i].co += Vector((
            random.gauss(0, sigma), random.gauss(0, sigma), random.gauss(0, sigma)))
        count += 1
    obj.data.update()
    print(f"  Edge noise: {count} boundary verts jittered (sigma={sigma:.4f})")


def _export_sparse_pc(obj, filepath, num_points=2048, noise_scale=0.01, edge_noise_extra=0.02):
    """Sample sparse point cloud from mesh surface with noise (edge points get extra)."""
    import bmesh as _bmesh
    size = _get_bbox_size(obj)
    base_sigma = size * noise_scale
    edge_sigma = size * edge_noise_extra

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = _bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    boundary = {e.verts[0].index for e in bm.edges if len(e.link_faces) == 1}
    boundary.update(e.verts[1].index for e in bm.edges if len(e.link_faces) == 1)

    faces = list(bm.faces)
    if not faces:
        bpy.ops.object.mode_set(mode='OBJECT')
        return

    areas = [f.calc_area() for f in faces]
    total_area = sum(areas) or 1.0

    verts_out = []
    for _ in range(num_points):
        r = random.random() * total_area
        cum = 0.0
        face = faces[-1]
        for f, a in zip(faces, areas):
            cum += a
            if cum >= r:
                face = f
                break

        fv = face.verts
        if len(fv) == 3 or (len(fv) == 4 and random.random() < 0.5):
            u = random.random(); v = random.random() * (1 - u); w = 1 - u - v
            pt = fv[0].co * u + fv[1].co * v + fv[2].co * w
        else:
            u = random.random(); v = random.random() * (1 - u); w = 1 - u - v
            pt = fv[0].co * u + fv[2].co * v + fv[3].co * w

        near_edge = any(v.index in boundary for v in face.verts)
        sigma = edge_sigma if near_edge else base_sigma
        pt += Vector((random.gauss(0, sigma), random.gauss(0, sigma), random.gauss(0, sigma)))
        verts_out.append(obj.matrix_world @ pt)

    bpy.ops.object.mode_set(mode='OBJECT')

    with open(filepath, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts_out)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for v in verts_out:
            f.write(f"{v.x:.6f} {v.y:.6f} {v.z:.6f}\n")
    print(f"  Sparse PC: {len(verts_out)} pts (base_sigma={base_sigma:.4f}, edge_sigma={edge_sigma:.4f})")


def _load_model(idx):
    global _qr_state
    rel_path = TIMEOUT_MODELS[idx]
    full_path = str(Path(DATA_DIR) / rel_path)
    model_name = Path(full_path).parent.name
    out_dir = Path(OUT_DIR) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean scene (low-level API)
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)

    bpy.ops.wm.obj_import(filepath=full_path)
    obj = bpy.context.selected_objects[0]
    obj.name = model_name
    bpy.context.view_layer.objects.active = obj

    _qr_state = {"model_idx": idx, "model_path": full_path,
                 "model_name": model_name, "out_dir": out_dir}

    print(f"\n{'='*60}")
    print(f"[{idx+1}/{len(TIMEOUT_MODELS)}] {model_name}")
    print(f"  Verts: {len(obj.data.vertices)}, Faces: {len(obj.data.polygons)}")
    print(f"  1. Click REMESH IT in Quad Remesh panel (N > Quad Remesh)")
    print(f"  2. Click CONFIRM & EXPORT in QR Pipeline panel")
    print(f"{'='*60}")


# ---- Register ----

def _find_existing_model():
    """If scene already has a mesh, adopt it as current state (avoid overwriting remesh)."""
    mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
    if not mesh_objs:
        return False
    obj = mesh_objs[0]
    name = obj.name
    # Try to match to a TIMEOUT_MODELS entry
    for i, rel_path in enumerate(TIMEOUT_MODELS):
        if Path(rel_path).parent.name == name:
            out_dir = Path(OUT_DIR) / name
            out_dir.mkdir(parents=True, exist_ok=True)
            _qr_state["model_idx"] = i
            _qr_state["model_name"] = name
            _qr_state["out_dir"] = out_dir
            _qr_state["model_path"] = str(Path(DATA_DIR) / rel_path)
            bpy.context.view_layer.objects.active = obj
            print(f"\n{'='*60}")
            print(f"Detected existing remeshed model: {name} (index {i})")
            print(f"  Verts: {len(obj.data.vertices)}, Faces: {len(obj.data.polygons)}")
            print(f"  Click CONFIRM & EXPORT to save and continue")
            print(f"{'='*60}")
            return True
    # No match found, just use it generically
    out_dir = Path(OUT_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    _qr_state["model_idx"] = 0
    _qr_state["model_name"] = name
    _qr_state["out_dir"] = out_dir
    _qr_state["model_path"] = ""
    bpy.context.view_layer.objects.active = obj
    print(f"Detected existing mesh: {name} (unmatched, using as index 0)")
    return True


def register():
    global _qr_state
    bpy.utils.register_class(QR_LOAD_OT)
    bpy.utils.register_class(QR_CONFIRM_OT)
    bpy.utils.register_class(QR_SKIP_OT)
    bpy.utils.register_class(QR_PANEL)

    # If there's already a mesh in the scene, adopt it; otherwise load first model
    if not _find_existing_model():
        _load_model(0)


def unregister():
    bpy.utils.unregister_class(QR_PANEL)
    bpy.utils.unregister_class(QR_SKIP_OT)
    bpy.utils.unregister_class(QR_CONFIRM_OT)
    bpy.utils.unregister_class(QR_LOAD_OT)


if __name__ == "__main__":
    register()
