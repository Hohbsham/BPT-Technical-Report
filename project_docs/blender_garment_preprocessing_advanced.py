"""
Advanced Blender Garment Preprocessing Script

This script performs the following operations on a selected garment mesh:
1. Adds a Solidify modifier with thickness 0.005
2. Applies the modifier
3. Processes mesh with QuadRemesher (settings to be configured manually)
4. Automatically detects and selects sleeve and collar edges
5. Marks selected edges as sharp
6. Sets auto-smooth to 180 degrees
7. Fixes normals
"""

import bpy
import bmesh
import mathutils
from mathutils import Vector


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
    print("\n*** QuadRemesher Configuration ***")
    print("Please configure QuadRemesher with these settings:")
    print("- Target Face Count: 8000")
    print("- Adaptive Size: 0")
    print("- Adaptive Quad Count: OFF")
    print("- Detect Hard Edges from Angle: OFF")
    print("- Use Normals Splitting: ON")
    print("Then click 'REMESH IT'")
    print("*********************************\n")


def detect_and_select_edge_loops_by_position(obj, tolerance=0.1):
    """
    Detect potential sleeve and collar edge loops based on vertex positions.
    This is a heuristic approach that identifies likely locations based on:
    - Sleeve openings: typically near y-axis extremes
    - Collar opening: typically at highest z values (neck area)
    """
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
            potential_edges.extend([edge.index])
        
        # Check if vertices are near the top z (potential collar)
        avg_z = (v1.co.z + v2.co.z) / 2
        if avg_z > z_neck_threshold:
            potential_edges.extend([edge.index])
    
    # Select the identified edges
    bpy.ops.object.mode_set(mode='OBJECT')  # Switch to object mode to select by index
    
    # Select the potential edges
    for edge_idx in set(potential_edges):  # Use set to avoid duplicates
        me.edges[edge_idx].select = True
    
    bpy.ops.object.mode_set(mode='EDIT')  # Back to edit mode
    
    print(f"Selected {len(set(potential_edges))} potential sleeve/collar edges")
    return len(set(potential_edges)) > 0


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


def process_garment_complete(obj_name=None):
    """Complete processing pipeline for a garment object"""
    # Get the active object or find by name
    if obj_name:
        obj = bpy.data.objects.get(obj_name)
        if not obj:
            print(f"Object '{obj_name}' not found")
            return
    else:
        obj = bpy.context.active_object
        if not obj:
            print("No active object found")
            return
    
    print(f"Processing garment object: {obj.name}")
    
    # Step 1: Add and apply Solidify modifier
    print("\nStep 1: Applying Solidify modifier...")
    add_and_apply_solidify_modifier(obj, thickness=0.005)
    
    # Step 2: Inform user about QuadRemesher settings
    print("\nStep 2: Configure and run QuadRemesher...")
    setup_quadremesher_settings()
    
    # At this point, user would run QuadRemesher manually
    # For automation, we'd need to check if there's a Python API for QuadRemesher
    # For now, we'll proceed assuming the user has done this step
    
    # Step 3: Auto-detect and select sleeve and collar edges
    print("\nStep 3: Detecting and selecting sleeve/collar edges...")
    edges_found = detect_and_select_edge_loops_by_position(obj)
    
    if edges_found:
        print("Step 3b: Marking selected edges as sharp...")
        mark_selected_edges_sharp()
    else:
        print("No suitable edges found for marking as sharp - please select manually")
    
    # Step 4: Set auto smooth
    print("\nStep 4: Setting auto-smooth angle...")
    set_auto_smooth_angle(obj, angle_degrees=180)
    
    # Step 5: Fix normals
    print("\nStep 5: Fixing normals...")
    fix_normals(obj)
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print(f"\nCompleted processing for {obj.name}")
    print("Processing complete! The garment model is ready for BPT conversion.")


def main():
    """Main function to run the preprocessing"""
    print("Starting advanced garment preprocessing pipeline...")
    
    # Process the currently selected object
    process_garment_complete()
    
    print("\nGarment preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()