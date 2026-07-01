import socket
import json

code = """
import bpy
import os
from pathlib import Path
from mathutils import Vector

base = Path(r"D:\\ClothesNetData\\bpt\\validation_results")

# Clear everything
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=True)

# Remove old materials except defaults
for m in list(bpy.data.materials):
    bpy.data.materials.remove(m)
for img in list(bpy.data.images):
    bpy.data.images.remove(img)

# ---- Scene setup ----
scene = bpy.context.scene
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes['Background']
bg.inputs['Color'].default_value = (0.25, 0.25, 0.28, 1.0)
bg.inputs['Strength'].default_value = 0.8

# Better lighting
bpy.ops.object.light_add(type='SUN', location=(5, -10, 12))
sun = bpy.context.active_object
sun.data.energy = 4.0
sun.rotation_euler = (0.6, 0.3, 0.5)

bpy.ops.object.light_add(type='AREA', location=(-5, 8, 8))
fill = bpy.context.active_object
fill.data.energy = 80.0
fill.data.size = 15.0

# Viewport shading
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        space = area.spaces[0]
        space.shading.type = 'MATERIAL'
        space.shading.use_scene_lights = True
        space.shading.use_scene_world = True
        break

# Camera
bpy.ops.object.camera_add(location=(18, -18, 12))
cam = bpy.context.active_object
cam.rotation_euler = (1.05, 0.0, 0.8)
scene.camera = cam

# ---- Helper functions ----
def make_material(name, color, alpha=1.0, rough=0.5):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    bsdf.inputs['Roughness'].default_value = rough
    if alpha < 1.0:
        mat.blend_method = 'BLEND'
        mat.shadow_method = 'CLIP'
        bsdf.inputs['Alpha'].default_value = alpha
    return mat

def import_obj_at(path, x, y):
    before = set(bpy.data.objects)
    if str(path).endswith('.ply'):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.wm.obj_import(filepath=str(path))
    after = set(bpy.data.objects)
    for obj in (after - before):
        obj.location = Vector((x, y, 0))
    return list(after - before)

def assign_material(objs, mat):
    for obj in objs:
        if obj.data and hasattr(obj.data, 'materials'):
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)

def add_text_label(label, x, y, z, size=0.5):
    bpy.ops.object.text_add(location=(x, y, z))
    t = bpy.context.active_object
    t.data.body = label
    t.data.size = size
    t.data.align_x = 'CENTER'
    t.data.extrude = 0.02
    # Emissive white material
    mat = bpy.data.materials.new(name='text_'+label)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1)
    bsdf.inputs['Emission Color'].default_value = (1, 1, 1, 1)
    bsdf.inputs['Emission Strength'].default_value = 1.5
    t.data.materials.append(mat)
    return t

# ---- Materials ----
mat_clean = make_material('Clean', (0.25, 0.55, 0.90), rough=0.35)
mat_recon = make_material('Reconstructed', (0.95, 0.45, 0.15), rough=0.35)
mat_pc = make_material('PointCloud', (0.95, 0.95, 0.95), rough=0.5)

# ---- Arrange models ----
spacing_x = 5.0  # Wider spacing
samples = ["DLG_Dress032_0", "DLG_Dress198", "DLG_Dress394"]
row_y_offsets = [6.0, 0.0, -6.0]

print("="*60)
print("Re-importing with wider spacing...")
print("="*60)

for i, name in enumerate(samples):
    x = -len(samples) * spacing_x / 2 + i * spacing_x + spacing_x / 2

    # Row 1: Clean original
    clean_path = base / f"{name}_clean.obj"
    if not clean_path.exists():
        clean_path = base / f"{name}_original.obj"
    if clean_path.exists():
        objs = import_obj_at(clean_path, x, row_y_offsets[0])
        assign_material(objs, mat_clean)
        print(f"  [{i+1}] {name} clean at x={x:.1f}")

    # Row 2: Reconstructed
    recon_path = base / f"{name}_reconstructed_quad.obj"
    if recon_path.exists():
        objs = import_obj_at(recon_path, x, row_y_offsets[1])
        assign_material(objs, mat_recon)
        print(f"  [{i+1}] {name} recon at x={x:.1f}")

    # Row 3: Point cloud
    pc_path = base / f"{name}_pointcloud.ply"
    if pc_path.exists():
        objs = import_obj_at(pc_path, x, row_y_offsets[2])
        assign_material(objs, mat_pc)
        # Make point cloud visible as vertex points
        for obj in objs:
            if obj.type == 'MESH':
                obj.display_type = 'WIRE'
        print(f"  [{i+1}] {name} point cloud at x={x:.1f}")

# ---- Labels ----
label_x = -len(samples) * spacing_x / 2 - 2.5

add_text_label("Original", label_x, row_y_offsets[0], 0, size=0.45)
add_text_label("Reconstructed", label_x, row_y_offsets[1], 0, size=0.45)
add_text_label("Point Cloud", label_x, row_y_offsets[2], 0, size=0.45)

for i, name in enumerate(samples):
    x = -len(samples) * spacing_x / 2 + i * spacing_x + spacing_x / 2
    short = name.replace("DLG_", "").replace("Dress", "D")
    add_text_label(short, x, row_y_offsets[0] + 2.0, 0, size=0.4)

print(f"\\nDone! Layout:")
print(f"  Columns: {', '.join(samples)}")
print(f"  Row 1 (Blue):   Original mesh")
print(f"  Row 2 (Orange): BPT quad reconstruction")
print(f"  Row 3 (White):  Input point cloud (noisy scan)")
print(f"  Spacing: {spacing_x}m between columns")
"""

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(60)
try:
    sock.connect(('localhost', 9876))
    msg = json.dumps({'type': 'execute_code', 'params': {'code': code}})
    sock.sendall(msg.encode('utf-8'))

    chunks = []
    while True:
        try:
            chunk = sock.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
        except socket.timeout:
            break

    response = b''.join(chunks).decode('utf-8')
    data = json.loads(response)
    if data.get('status') == 'success':
        print('SUCCESS')
        result = data.get('result', {})
        if isinstance(result, dict) and result.get('executed'):
            stdout = result.get('result', '')
            if stdout:
                print(stdout[:3000])
    else:
        print('ERROR: ' + data.get('message', ''))
except Exception as e:
    print('Error: ' + str(e))
finally:
    sock.close()
