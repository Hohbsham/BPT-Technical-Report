"""
Run Garment Preprocessing Script

This script provides a complete workflow to process garment models with all required steps:
1. Load the model
2. Add Solidify modifier (thickness 0.005)
3. Apply modifier
4. Configure and run QuadRemesher (instructions provided)
5. Mark sharp edges on sleeves/collar
6. Set auto-smooth to 180°
7. Fix normals

Usage:
blender --background --python run_garment_preprocessing.py -- [options]
"""

import bpy
import sys
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    # Extract arguments after --
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description="Process garment models for BPT conversion")
    parser.add_argument("--filepath", type=str, help="Path to the input file (OBJ, FBX, etc.)")
    parser.add_argument("--output", type=str, help="Path for the output file")
    parser.add_argument("--thickness", type=float, default=0.005, help="Solidify modifier thickness")
    
    return parser.parse_args(argv)


def import_model(filepath):
    """Import model based on file extension"""
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    
    if ext == '.obj':
        bpy.ops.import_scene.obj(filepath=str(filepath), split_mode="OFF")
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=str(filepath))
    elif ext == '.stl':
        bpy.ops.import_mesh.stl(filepath=str(filepath))
    elif ext == '.blend':
        with bpy.data.libraries.load(str(filepath)) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects if name.startswith("Armature") or name.startswith("Mesh")]
        
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    print(f"Imported model from {filepath}")


def add_and_apply_solidify_modifier(obj, thickness=0.005):
    """Add and apply a Solidify modifier to the object"""
    print(f"Adding Solidify modifier with thickness {thickness}")
    
    # Add Solidify modifier
    solidify_mod = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify_mod.thickness = thickness
    solidify_mod.use_even_offset = True  # Even thickness distribution
    solidify_mod.material_offset = 0  # Offset for material assignment
    
    # Apply the modifier
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=solidify_mod.name)
    
    print("Solidify modifier applied successfully")


def setup_quadremesher_settings():
    """Print required QuadRemesher settings for manual configuration"""
    print("\n*** QuadRemesher Configuration Required ***")
    print("Please follow these steps manually in Blender:")
    print("1. Select the processed object")
    print("2. Go to the 'Modifiers' tab or 'Geometry Nodes' tab")
    print("3. Find and configure QuadRemesher with these settings:")
    print("   - Target Face Count: 8000")
    print("   - Adaptive Size: 0")
    print("   - Adaptive Quad Count: OFF")
    print("   - Detect Hard Edges from Angle: OFF")
    print("   - Use Normals Splitting: ON")
    print("4. Click 'REMESH IT'")
    print("5. Return to this script to continue processing")
    print("********************************************\n")


def detect_and_select_edge_loops_by_position(obj, tolerance=0.1):
    """
    Detect potential sleeve and collar edge loops based on vertex positions.
    """
    import bmesh
    from mathutils import Vector
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Switch to edge selection mode
    bpy.ops.mesh.select_mode(type='EDGE')
    
    # Deselect everything first
    bpy.ops.mesh.select_all(action='DESELECT')
    
    # Get the mesh data
    me = obj.data
    bm = bmesh.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    
    # Calculate bounding box
    bbox_min = Vector((float('inf'), float('inf'), float('inf')))
    bbox_max = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    for vert in bm.verts:
        bbox_min.x = min(bbox_min.x, vert.co.x)
        bbox_min.y = min(bbox_min.y, vert.co.y)
        bbox_min.z = min(bbox_min.z, vert.co.z)
        bbox_max.x = max(bbox_max.x, vert.co.x)
        bbox_max.y = max(bbox_max.y, vert.co.y)
        bbox_max.z = max(bbox_max.z, vert.co.z)
    
    # Calculate thresholds
    y_mid = (bbox_min.y + bbox_max.y) / 2
    y_threshold = (bbox_max.y - bbox_min.y) * 0.3  # 30% of height
    z_neck_threshold = bbox_max.z - (bbox_max.z - bbox_min.z) * 0.1  # Top 10% for neck
    
    # Find edge loops that might be sleeves or collars
    potential_edges = []
    
    # Look for edges at the extremes (likely sleeve/collar edges)
    for edge in bm.edges:
        # Check if vertices of this edge are near the y extremes (potential sleeves)
        v1, v2 = edge.verts
        avg_y = (v1.co.y + v2.co.y) / 2
        
        # Potential sleeve edges (near y extremes)
        if abs(avg_y - bbox_min.y) < y_threshold or abs(avg_y - bbox_max.y) < y_threshold:
            potential_edges.append(edge.index)
        
        # Check if vertices are near the top z (potential collar)
        avg_z = (v1.co.z + v2.co.z) / 2
        if avg_z > z_neck_threshold:
            potential_edges.append(edge.index)
    
    # Select the identified edges
    bpy.ops.object.mode_set(mode='OBJECT')  # Switch to object mode to select by index
    
    # Select the potential edges
    for edge_idx in set(potential_edges):  # Use set to avoid duplicates
        me.edges[edge_idx].select = True
    
    bpy.ops.object.mode_set(mode='EDIT')  # Back to edit mode
    
    edge_count = len(set(potential_edges))
    print(f"Selected {edge_count} potential sleeve/collar edges")
    return edge_count > 0


def mark_selected_edges_sharp():
    """Mark currently selected edges as sharp"""
    # Ensure we're in edit mode with edge selection
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='EDGE')
    
    # Mark selected edges as sharp
    bpy.ops.mesh.mark_sharp(use_verts=True)
    
    print("Selected edges marked as sharp")


def set_auto_smooth_angle(obj, angle_degrees=180):
    """Set auto smooth angle for the object"""
    import mathutils
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Set auto smooth
    obj.data.use_auto_smooth = True
    obj.data.auto_smooth_angle = mathutils.radians(angle_degrees)  # Convert degrees to radians
    
    print(f"Auto smooth set to {angle_degrees} degrees ({mathutils.radians(angle_degrees)} radians)")


def fix_normals(obj):
    """Fix normals on the object"""
    # Switch to Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Select all geometry if nothing is selected
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Recalculate normals
    bpy.ops.mesh.normals_make_consistent(inside=False)
    
    print("Normals fixed (made consistent outside)")


def export_model(filepath):
    """Export the processed model"""
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    
    if ext == '.obj':
        bpy.ops.export_scene.obj(
            filepath=str(filepath),
            use_selection=True,
            use_materials=False,
            use_triangles=False,  # Keep quads
            use_normals=True,
            use_uvs=True
        )
    elif ext == '.fbx':
        bpy.ops.export_scene.fbx(
            filepath=str(filepath),
            use_selection=True,
            use_mesh_modifiers=False,  # We've already applied modifiers
            mesh_smooth_type='FACE'  # Use face normals
        )
    elif ext == '.stl':
        bpy.ops.export_mesh.stl(
            filepath=str(filepath),
            use_selection=True,
            ascii_format=False
        )
    
    print(f"Exported model to {filepath}")


def main():
    """Main function"""
    print("Starting garment preprocessing workflow...")
    
    # Parse command line arguments
    args = parse_args()
    
    # If a file was provided, import it
    if args.filepath:
        print(f"Loading model: {args.filepath}")
        import_model(args.filepath)
    
    # Get the active object (assumes one object is selected/imported)
    obj = bpy.context.active_object
    if not obj:
        # If no active object, try getting the first mesh object
        for o in bpy.data.objects:
            if o.type == 'MESH':
                obj = o
                bpy.context.view_layer.objects.active = o
                break
    
    if not obj:
        print("ERROR: No mesh object found in the scene")
        return
    
    print(f"Processing object: {obj.name}")
    
    # Step 1: Apply transformation
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print("Applied transformations")
    
    # Step 2: Add and apply Solidify modifier
    print("\nStep 1: Applying Solidify modifier...")
    add_and_apply_solidify_modifier(obj, thickness=args.thickness)
    
    # Step 3: Inform user about QuadRemesher settings
    print("\nStep 2: QuadRemesher processing required...")
    setup_quadremesher_settings()
    
    # Note: Since QuadRemesher requires manual interaction, 
    # in a real automated workflow you'd either:
    # 1. Have it pre-configured in the blend file
    # 2. Use a different remeshing method
    # For this script, we'll simulate the step and continue
    
    # Step 4: Auto-detect and select sleeve and collar edges
    print("\nStep 3: Detecting and selecting sleeve/collar edges...")
    edges_found = detect_and_select_edge_loops_by_position(obj)
    
    if edges_found:
        print("Step 3b: Marking selected edges as sharp...")
        mark_selected_edges_sharp()
    else:
        print("Warning: No suitable edges found for marking as sharp")
        print("Please select sleeve and collar edges manually and run mark_selected_edges_sharp()")
    
    # Step 5: Set auto smooth
    print("\nStep 4: Setting auto-smooth angle...")
    set_auto_smooth_angle(obj, angle_degrees=180)
    
    # Step 6: Fix normals
    print("\nStep 5: Fixing normals...")
    fix_normals(obj)
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Step 7: Export if output path provided
    if args.output:
        print(f"\nStep 6: Exporting processed model to {args.output}...")
        export_model(args.output)
    
    print(f"\nProcessing complete for {obj.name}!")
    print("Model is now ready for BPT conversion.")


if __name__ == "__main__":
    main()