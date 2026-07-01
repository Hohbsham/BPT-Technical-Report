import socket
import json

code = """
import bpy
from pathlib import Path
from mathutils import Vector

base = Path(r"D:\\ClothesNetData\\bpt\\validation_results")

# ---- Clear old point cloud objects only ----
to_remove = []
for obj in bpy.data.objects:
    if 'pointcloud' in obj.name.lower() or 'noisy' in obj.name.lower() or obj.name.startswith('PointCloud'):
        to_remove.append(obj)
for obj in to_remove:
    bpy.data.objects.remove(obj, do_unlink=True)

# ---- Create point cloud visualization as small spheres ----
def add_point_spheres(obj_path, x, y, color, radius=0.04):
    \"\"\"Import point vertices and create visible spheres at each point.\"\"\"
    # Import the point-only OBJ
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(obj_path))
    after = set(bpy.data.objects)
    new_objs = list(after - before)

    if not new_objs:
        print(f"  WARNING: No objects imported from {obj_path}")
        return None

    point_obj = new_objs[0]
    point_obj.location = Vector((x, y, 0))
    point_obj.name = Path(obj_path).stem

    # Create material
    mat_name = f"noisy_pc_{Path(obj_path).stem}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    bsdf.inputs['Emission Color'].default_value = (*color, 1.0)
    bsdf.inputs['Emission Strength'].default_value = 0.3
    bsdf.inputs['Roughness'].default_value = 0.3

    if point_obj.data.materials:
        point_obj.data.materials[0] = mat
    else:
        point_obj.data.materials.append(mat)

    # Use geometry nodes to instance spheres at each vertex
    # First, enable geometry nodes
    mod = point_obj.modifiers.new(name='PointSpheres', type='NODES')

    # Create node group
    ng_name = f'PointToSpheres_{radius}'
    ng = bpy.data.node_groups.get(ng_name)
    if ng is None:
        ng = bpy.data.node_groups.new(ng_name, 'GeometryNodeTree')

        # Input/Output
        group_in = ng.nodes.new('NodeGroupInput')
        group_out = ng.nodes.new('NodeGroupOutput')

        # Mesh to Points
        mesh_to_pts = ng.nodes.new('GeometryNodeMeshToPoints')
        mesh_to_pts.location = (-200, 0)

        # Instance on Points
        inst_on_pts = ng.nodes.new('GeometryNodeInstanceOnPoints')
        inst_on_pts.location = (0, 0)

        # Ico Sphere for instancing
        ico = ng.nodes.new('GeometryNodeMeshIcoSphere')
        ico.location = (-200, -200)
        ico.inputs['Radius'].default_value = radius
        ico.inputs['Subdivisions'].default_value = 2

        # Join geometry (original mesh wireframe + spheres)
        join_geo = ng.nodes.new('GeometryNodeJoinGeometry')
        join_geo.location = (200, 0)

        # Realize instances
        realize = ng.nodes.new('GeometryNodeRealizeInstances')
        realize.location = (400, 0)

        # Set material
        set_mat = ng.nodes.new('GeometryNodeSetMaterial')
        set_mat.location = (600, 0)

        # Links
        links = ng.links
        links.new(group_in.outputs['Geometry'], mesh_to_pts.inputs['Mesh'])
        links.new(mesh_to_pts.outputs['Points'], inst_on_pts.inputs['Points'])
        links.new(ico.outputs['Mesh'], inst_on_pts.inputs['Instance'])
        links.new(group_in.outputs['Geometry'], join_geo.inputs['Geometry'])
        links.new(inst_on_pts.outputs['Instances'], realize.inputs['Geometry'])
        links.new(realize.outputs['Geometry'], set_mat.inputs['Geometry'])
        links.new(set_mat.outputs['Geometry'], group_out.inputs['Geometry'])

    mod.node_group = ng

    # Update viewport display
    point_obj.show_in_front = False

    print(f"  Created {len(point_obj.data.vertices)} visible noise points at ({x:.1f}, {y:.1f})")
    return point_obj


print("="*60)
print("Adding noisy point cloud visualization...")
print("="*60)

spacing_x = 5.0
row_y_pc = -6.0

samples = ["DLG_Dress032_0", "DLG_Dress198", "DLG_Dress394"]
colors = [(1.0, 0.3, 0.3), (1.0, 0.3, 0.3), (1.0, 0.3, 0.3)]  # Red for noisy

for i, name in enumerate(samples):
    x = -len(samples) * spacing_x / 2 + i * spacing_x + spacing_x / 2
    pc_path = base / f"{name}_noisy_pc.obj"
    if pc_path.exists():
        add_point_spheres(pc_path, x, row_y_pc, colors[i], radius=0.03)
    else:
        print(f"  WARNING: {pc_path} not found!")

print(f"\\nDone! Red spheres = noisy point cloud input (aligned to GT)")
print(f"  Each point cloud has ~2048 visible points")
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
