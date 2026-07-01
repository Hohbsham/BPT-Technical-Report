"""
Blender 后台渲染脚本 — mesh 多角度截图
用法: blender --background --python qr_render_mesh.py -- <obj_path> <output_dir> <sample_name>
输出: <output_dir>/<sample_name>_view1.png, _view2.png, _view3.png
"""
import bpy
import sys
import os
from mathutils import Vector, Matrix
from math import radians


def setup_scene():
    """Setup lighting, camera, and render settings for EEVEE."""
    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.meshes):
        if m.users == 0:
            bpy.data.meshes.remove(m)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)

    # Render engine
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 768
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.eevee.taa_render_samples = 32

    # World background
    world = bpy.data.worlds.new("World") if not bpy.data.worlds else bpy.data.worlds[0]
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0.15, 0.15, 0.15, 1.0)
        bg.inputs['Strength'].default_value = 0.5

    # Lighting: SUN key + AREA fill
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.data.angle = 0.1

    bpy.ops.object.light_add(type='AREA', location=(-3, 3, 5))
    area = bpy.context.active_object
    area.data.energy = 50.0
    area.data.size = 5.0

    # Material for mesh
    mat = bpy.data.materials.new("MeshMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.3, 0.55, 0.9, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.35
        bsdf.inputs['Specular IOR Level'].default_value = 0.2

    return mat


def import_mesh(obj_path, material):
    """Import OBJ and apply material."""
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    # Apply material
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

    # Flat shading for wireframe clarity
    for f in obj.data.polygons:
        f.use_smooth = False

    # Normalize position
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)

    # Get bounding box for camera framing
    bbox = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    size = max(
        max(v[i] for v in bbox) - min(v[i] for v in bbox)
        for i in range(3)
    )

    return obj, size


def setup_camera(target_obj, size, view_index):
    """Position camera for a specific view angle."""
    # Remove existing cameras
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam)

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam

    distance = size * 3.5
    center = target_obj.location.copy()

    if view_index == 0:
        # Front-top-right (isometric-like)
        cam.location = center + Vector((distance*0.7, -distance*0.7, distance*0.5))
    elif view_index == 1:
        # Side profile (right)
        cam.location = center + Vector((distance, 0, distance*0.3))
    else:
        # Top-down
        cam.location = center + Vector((0, 0, distance))

    # Point at center
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    return cam


def render_view(output_path):
    """Render current scene to file."""
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)


def main():
    # Parse args (skip Blender's own args before '--')
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        print("Usage: blender --background --python qr_render_mesh.py -- <obj_path> <output_dir> <sample_name>")
        sys.exit(1)

    if len(argv) < 3:
        print(f"ERROR: Need 3 args, got {len(argv)}: {argv}")
        sys.exit(1)

    obj_path = argv[0]
    output_dir = argv[1]
    sample_name = argv[2]

    if not os.path.exists(obj_path):
        print(f"ERROR: OBJ not found: {obj_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Setup
    mat = setup_scene()
    obj, size = import_mesh(obj_path, mat)

    # Render 3 views
    view_names = ["view1_front", "view2_side", "view3_top"]
    for i, vname in enumerate(view_names):
        setup_camera(obj, size, i)
        output_path = os.path.join(output_dir, f"{sample_name}_{vname}.png")
        render_view(output_path)
        print(f"Rendered: {output_path}")

    print(f"DONE: {sample_name} (size={size:.2f})")


if __name__ == "__main__":
    main()
