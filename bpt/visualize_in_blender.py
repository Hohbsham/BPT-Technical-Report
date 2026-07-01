"""
Import validation results into Blender and arrange them for visual comparison.
Run this script inside Blender's Python console or via --python flag.
"""

import bpy
import os
from pathlib import Path
from mathutils import Vector

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=True)

base = Path(r"D:\ClothesNetData\bpt\validation_results")
spacing = 3.0
lighting_setup_done = False

def setup_scene():
    """Lights, camera, world settings."""
    global lighting_setup_done
    if lighting_setup_done:
        return

    # World
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.15, 0.15, 0.16, 1.0)
    bg.inputs['Strength'].default_value = 0.5

    # Key light
    bpy.ops.object.light_add(type='SUN', location=(10, -10, 15))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.rotation_euler = (0.8, 0.2, 0.5)

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-5, 5, 8))
    area = bpy.context.active_object
    area.data.energy = 50.0
    area.data.size = 10.0

    # Camera
    bpy.ops.object.camera_add(location=(15, -15, 10))
    cam = bpy.context.active_object
    cam.rotation_euler = (1.1, 0.0, 0.8)
    bpy.context.scene.camera = cam
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

    lighting_setup_done = True


def import_obj(path):
    """Import OBJ file, return imported objects."""
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(path))
    after = set(bpy.data.objects)
    new_objs = list(after - before)
    return new_objs


def import_ply(path):
    """Import PLY point cloud, return imported object."""
    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=str(path))
    after = set(bpy.data.objects)
    new_objs = list(after - before)
    return new_objs


def apply_material(obj, color, name_suffix="", alpha=1.0):
    """Create and assign a simple material."""
    mat_name = f"Mat_{color[0]:.1f}_{color[1]:.1f}_{color[2]:.1f}_{name_suffix}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Base Color'].default_value = (*color, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.6
        if alpha < 1.0:
            mat.blend_method = 'BLEND'
            bsdf.inputs['Alpha'].default_value = alpha

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def point_cloud_to_mesh(obj, radius=0.015):
    """Convert point cloud vertices to small spheres via Geometry Nodes or instancing."""
    # Simple approach: use the object's vertices with a wireframe/thick display
    obj.show_in_front = True
    # Make points visible
    if hasattr(obj.data, 'attributes'):
        pass  # Will handle differently
    # Use display settings for visibility
    return obj


def move_to(obj, position):
    """Move object to absolute world position."""
    if hasattr(obj, 'location'):
        obj.location = Vector(position)


# ============================================
# ROW 1: Quad samples - original vs reconstructed
# ============================================
row1_samples = [
    "DLG_Dress032_0",
    "DLG_Dress198",
    "DLG_Dress394",
]

row1_y = 6.0
x_start = -len(row1_samples) * spacing

print("=" * 60)
print("Importing meshes into Blender...")
print("=" * 60)

setup_scene()

# Row 1: Clean meshes (blue)
for i, name in enumerate(row1_samples):
    obj_path = base / f"{name}_clean.obj"
    if not obj_path.exists():
        obj_path = base / f"{name}_original.obj"
    if obj_path.exists():
        objs = import_obj(obj_path)
        x = x_start + i * spacing * 2
        for obj in objs:
            move_to(obj, (x, row1_y, 0))
            apply_material(obj, (0.3, 0.5, 0.9), "clean")
        print(f"  [{i+1}] {name} (clean) -> x={x:.1f}")

# Row 2: Reconstructed meshes (orange)
row2_y = row1_y - spacing * 1.2
for i, name in enumerate(row1_samples):
    obj_path = base / f"{name}_reconstructed_quad.obj"
    if obj_path.exists():
        objs = import_obj(obj_path)
        x = x_start + i * spacing * 2
        for obj in objs:
            move_to(obj, (x, row2_y, 0))
            apply_material(obj, (0.95, 0.5, 0.2), "recon")
        print(f"  [{i+1}] {name} (reconstructed) -> x={x:.1f}")

# Row 3: Point clouds (white points)
row3_y = row2_y - spacing * 1.2
for i, name in enumerate(row1_samples):
    obj_path = base / f"{name}_pointcloud.ply"
    if obj_path.exists():
        objs = import_ply(obj_path)
        x = x_start + i * spacing * 2
        for obj in objs:
            move_to(obj, (x, row3_y, 0))
            apply_material(obj, (0.9, 0.9, 0.9), "pc", alpha=0.7)
            # Make points visible
            if obj.type == 'MESH':
                # Display vertices
                obj.display_type = 'WIRE'
        print(f"  [{i+1}] {name} (point cloud) -> x={x:.1f}")

# Labels using text objects
def add_text(label, position, size=0.4):
    bpy.ops.object.text_add(location=position)
    text_obj = bpy.context.active_object
    text_obj.data.body = label
    text_obj.data.size = size
    text_obj.data.align_x = 'CENTER'
    # White emissive material
    mat = bpy.data.materials.get("Mat_text")
    if mat is None:
        mat = bpy.data.materials.new(name="Mat_text")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
        bsdf.inputs['Emission Color'].default_value = (1, 1, 1, 1)
        bsdf.inputs['Emission Strength'].default_value = 2.0
    text_obj.data.materials.append(mat)
    return text_obj

# Column headers (sample names)
for i, name in enumerate(row1_samples):
    x = x_start + i * spacing * 2
    label = name.replace("DLG_", "").replace("Dress", "Dress ")
    add_text(label, (x, row1_y + 1.2, 0), size=0.35)

# Row labels
label_x = x_start - spacing * 1.2
add_text("ORIGINAL", (label_x, row1_y, 0), size=0.3)
add_text("RECONSTRUCTED", (label_x, row2_y, 0), size=0.3)
add_text("POINT CLOUD", (label_x, row3_y, 0), size=0.3)

print(f"\nDone! Imported {len(row1_samples)} samples with 3 rows each.")
print(f"  Row 1 (Blue):   Clean original meshes")
print(f"  Row 2 (Orange): BPT reconstructed (quad mode)")
print(f"  Row 3 (White):  Noisy point cloud input")
print(f"\nCamera positioned at top-right. Use middle-mouse to orbit.")
