# How to Use Blender Preprocessing Scripts

This document explains how to use the Blender scripts to process your garment models according to your requirements.

## Prerequisites

1. Install Blender 3.3 or later from https://www.blender.org/
2. Install the QuadRemesher plugin from https://exoside.com/quadremesher/
3. Ensure you have Python environment access

## Step-by-Step Instructions

### Option 1: Manual Processing in Blender GUI (Recommended for your specific workflow)

1. **Open Blender**
   - Launch Blender application

2. **Load your garment model**
   - File → Import → Choose your format (OBJ, FBX, etc.)
   - Select your garment model file

3. **Add Solidify modifier**
   - Select your model in Object mode
   - Go to Modifiers tab
   - Click "Add Modifier" → Solidify
   - Set Thickness to 0.005
   - Click "Apply"

4. **Open QuadRemesher plugin**
   - Go to the "Geometry Nodes" tab or search for QuadRemesher
   - Set parameters as follows:
     - Target Face Count: 8000
     - Adaptive Size: 0
     - Adaptive Quad Count: OFF
     - Detect Hard Edges from Angle: OFF
     - Use Normals Splitting: ON
   - Click "REMESH IT"

5. **Select and mark sharp edges**
   - Switch to Edit Mode (Tab key)
   - Change selection mode to Edge (Ctrl+Tab → Edge)
   - Select edge loops around sleeves and collar/neckline:
     - Position cursor near the sleeve/collar area
     - Hold Alt and click on an edge to select the whole loop
     - Or manually select edges with Ctrl+click
   - With edges selected, press Ctrl+E → Mark Sharp

6. **Set auto-smooth**
   - Switch back to Object mode (Tab key)
   - Go to Object Data Properties (triangle icon in Properties panel)
   - Under Normals section, check "Auto Smooth"
   - Set the angle to 180°

7. **Fix normals**
   - Switch to Edit Mode (Tab key)
   - Select all (A key)
   - Press Shift+N to recalculate normals (make them consistent outside)

### Option 2: Using the Provided Scripts

If you prefer to use the Python scripts:

1. **Save your model as a .blend file** after Step 4 (QuadRemesher processing)

2. **Run the preprocessing script:**
   ```
   "C:\Program Files\Blender Foundation\Blender 3.3\blender.exe" --background your_model.blend --python "D:/ClothesNetData/project_docs/blender_garment_preprocessing_advanced.py"
   ```

3. **Alternative: Process from command line**
   ```
   "C:\Program Files\Blender Foundation\Blender 3.3\blender.exe" --background --python "D:/ClothesNetData/project_docs/run_garment_preprocessing.py" -- --filepath "path/to/your/model.obj" --output "path/to/output/model.obj"
   ```

## Key Points

- **Solidify Thickness**: 0.005 units as requested
- **QuadRemesher Settings**: 
  - Target face count: 8000
  - Adaptive Size: 0
  - Adaptive Quad Count: OFF
  - Detect Hard Edges from Angle: OFF
  - Use Normals Splitting: ON
- **Sharp Edges**: Applied to sleeve and collar edge loops
- **Auto-Smooth**: 180° angle
- **Normals**: Fixed to face outward consistently

## Troubleshooting

### QuadRemesher Not Appearing
- Make sure the plugin is properly installed and enabled in Edit → Preferences → Add-ons
- Restart Blender after installation

### Script Fails to Execute
- Verify the file paths in the script are correct
- Ensure Blender is installed in the default location or update the script accordingly

### Edge Selection Issues
- The automatic edge detection in the script is heuristic-based
- Manual selection often yields better results for specific garment features

## Next Steps

Once your model is processed, it will be ready for:
1. BPT (Blocked and Patchified Tokenization) conversion
2. Neural rendering applications
3. Further processing in your pipeline

Your garment model will now have proper thickness, quad-based topology, correctly marked sharp edges, and properly oriented normals - exactly as you requested!