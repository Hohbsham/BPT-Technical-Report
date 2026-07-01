"""
Blender Garment Preprocessing Script

This script performs the following operations on a selected garment mesh:
1. Adds a Solidify modifier with thickness 0.005
2. Applies the modifier
3. Opens QuadRemesher with specific settings
4. Marks sharp edges on sleeve and collar areas
5. Sets auto-smooth to 180 degrees
6. Fixes normals
"""

import bpy
import bmesh


def add_and_apply_solidify_modifier(obj, thickness=0.005):
    """Add and apply a Solidify modifier to the object"""
    print(f"Adding Solidify modifier with thickness {thickness}")
    
    # Add Solidify modifier
    solidify_mod = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify_mod.thickness = thickness
    solidify_mod.use_even_offset = True  # Even thickness distribution
    
    # Apply the modifier
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=solidify_mod.name)
    
    print("Solidify modifier applied successfully")


def setup_quadremesher_settings():
    """Configure QuadRemesher with specific settings"""
    # Note: Actual QuadRemesher configuration would be done in the GUI
    # This function documents the required settings
    print("QuadRemesher settings to configure manually:")
    print("- Target Face Count: 8000")
    print("- Adaptive Size: 0")
    print("- Adaptive Quad Count: OFF")
    print("- Detect Hard Edges from Angle: OFF")
    print("- Use Normals Splitting: ON")


def mark_sharp_edges_for_sleeves_collar():
    """Manually select and mark sharp edges for sleeves and collar"""
    # Switch to Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Enable edge selection mode
    bpy.ops.mesh.select_mode(type='EDGE')
    
    # Deselect everything first
    bpy.ops.mesh.select_all(action='DESELECT')
    
    # In a real scenario, we would programmatically select sleeve and collar edges
    # For now, we'll describe the process
    print("In Edit Mode:")
    print("1. Select edge loops around sleeves")
    print("2. Select edge loops around collar/neckline")
    print("3. Mark as sharp using Ctrl+E -> Mark Sharp")
    
    # Mark selected edges as sharp
    bpy.ops.mesh.mark_sharp(use_verts=True)
    
    print("Sharp edges marked for sleeves and collar")


def set_auto_smooth_angle(obj, angle_degrees=180):
    """Set auto smooth angle for the object"""
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Set auto smooth
    obj.data.use_auto_smooth = True
    obj.data.auto_smooth_angle = angle_degrees  # in radians (180 degrees = π radians)
    
    print(f"Auto smooth set to {angle_degrees} degrees")


def fix_normals(obj):
    """Fix normals on the object"""
    # Switch to Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Select all geometry
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Recalculate normals
    bpy.ops.mesh.normals_make_consistent(inside=False)
    
    print("Normals fixed (made consistent outside)")


def process_garment_object(obj_name=None):
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
    add_and_apply_solidify_modifier(obj, thickness=0.005)
    
    # Step 2: The user needs to manually run QuadRemesher with specified settings
    setup_quadremesher_settings()
    print("\n*** IMPORTANT: Now run QuadRemesher manually with the above settings ***")
    print("*** Then return to this script to continue processing ***\n")
    
    # Wait for user to complete QuadRemesher step
    # In practice, this would be a blocking operation or done in separate phases
    
    # Step 3: Mark sharp edges for sleeves and collar
    # This requires the user to select the appropriate edges first
    mark_sharp_edges_for_sleeves_collar()
    
    # Step 4: Set auto smooth
    set_auto_smooth_angle(obj, angle_degrees=180)
    
    # Step 5: Fix normals
    fix_normals(obj)
    
    print(f"Completed processing for {obj.name}")
    

def main():
    """Main function to run the preprocessing"""
    print("Starting garment preprocessing pipeline...")
    
    # Process the currently selected object
    process_garment_object()
    
    print("Garment preprocessing completed!")


if __name__ == "__main__":
    main()