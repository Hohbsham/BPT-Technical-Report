# Blender Garment Preprocessing Guide

This guide explains how to use the preprocessing scripts to prepare garment models for BPT (Blocked and Patchified Tokenization) conversion.

## Overview

The preprocessing pipeline includes several steps to prepare garment models:

1. Adding thickness with Solidify modifier
2. Remeshing with QuadRemesher
3. Edge handling (sharpening sleeve and collar edges)
4. Normal corrections
5. Smoothing adjustments

## Required Software

- Blender 3.3 or later
- QuadRemesher plugin (https://exoside.com/quadremesher/)
- Python 3.7+ (for supporting scripts)

## Scripts Included

### 1. `blender_garment_preprocessing.py`
Basic preprocessing script that guides through the steps.

### 2. `blender_garment_preprocessing_advanced.py`
Advanced script that attempts to automatically detect sleeve and collar edges.

### 3. Existing Scripts
- `blender_quadremesher_bpt_converter.py` - QuadRemesher integration
- `bpt_converter.py` - BPT format conversion
- `garment_bpt_pipeline.py` - Full pipeline orchestrator

## Usage Instructions

### Method 1: Manual Processing in Blender

1. Open Blender
2. Load your garment model
3. Run the preprocessing script:
   ```bash
   blender --background your_model.blend --python "D:/ClothesNetData/project_docs/blender_garment_preprocessing_advanced.py"
   ```

### Method 2: Automated Pipeline

1. Prepare your garment models in a source directory
2. Run the full pipeline:
   ```bash
   cd "D:/ClothesNetData/project_docs/"
   python garment_bpt_pipeline.py --input-dir "D:/Your/Input/Directory" --output-dir "D:/Your/Output/Directory"
   ```

## Detailed Steps Performed

### 1. Solidify Modifier
- Thickness: 0.005 units
- Even offset: Enabled
- Material offset: 0

### 2. QuadRemesher Settings
- Target Face Count: 8000
- Adaptive Size: 0
- Adaptive Quad Count: OFF
- Detect Hard Edges from Angle: OFF
- Use Normals Splitting: ON
- Click "REMESH IT"

### 3. Edge Processing
- Automatic detection of sleeve and collar edge loops
- Mark detected edges as sharp
- If automatic detection fails, manual selection required

### 4. Smoothing
- Auto-smooth enabled
- Angle: 180 degrees (π radians)

### 5. Normal Correction
- Recalculate normals to face outward
- Ensures consistency across the mesh

## Troubleshooting

### QuadRemesher Not Found
- Verify QuadRemesher plugin is installed in Blender
- Check that the plugin is enabled in Preferences > Add-ons

### Automatic Edge Detection Fails
- Manually select sleeve and collar edges in Edit mode
- Use Alt+RMB to select edge loops
- Run the marking step separately

### Script Execution Errors
- Verify Blender Python API compatibility
- Check file paths are accessible
- Ensure sufficient memory for large meshes

## Integration with BPT Pipeline

The processed meshes are ready for BPT (Blocked and Patchified Tokenization) conversion which is the next step in the pipeline for neural rendering applications.

## Notes

- For best results, ensure your garment models are properly oriented (Y-forward, Z-up)
- Models should be at approximately human scale (meters)
- Clean topology will yield better results with QuadRemesher