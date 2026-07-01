"""
Garment BPT Pipeline - Complete workflow from mesh generation to BPT format
This script orchestrates the entire process: 
1. Uses QuadRemesher to convert irregular meshes to quad meshes
2. Optimizes meshes for BPT format
3. Converts to BPT (Blocked and Patchified Tokenization) format
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse


def run_blender_script(script_path: str, blend_file: str = None):
    """Run a Blender script using the installed Blender executable"""
    blender_exe = r"C:\Program Files\Blender Foundation\Blender 3.3\blender.exe"
    
    if not os.path.exists(blender_exe):
        print(f"Blender executable not found at {blender_exe}")
        print("Please install Blender 3.3 or update the path in this script")
        return False
    
    cmd = [blender_exe, "--background"]
    
    if blend_file and os.path.exists(blend_file):
        cmd.append(blend_file)
    
    cmd.extend(["--python", script_path])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Blender script timed out after 1 hour")
        return False
    except Exception as e:
        print(f"Error running Blender script: {e}")
        return False


def run_python_script(script_path: str):
    """Run a Python script using the current Python interpreter"""
    cmd = [sys.executable, script_path]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running Python script: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Garment BPT Pipeline")
    parser.add_argument("--input-dir", default=r"D:\ClothesNetData", help="Input directory containing garment meshes")
    parser.add_argument("--intermediate-dir", default=r"D:\ClothesNetData_Processed", help="Intermediate directory for quad meshes")
    parser.add_argument("--output-dir", default=r"D:\ClothesNetData_BPT_Converted", help="Output directory for BPT files")
    parser.add_argument("--block-size", type=int, default=16, help="BPT block size")
    parser.add_argument("--patch-size", type=int, default=4, help="BPT patch size (should be 4 for quads)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GARMENT BPT PIPELINE")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Intermediate directory: {args.intermediate_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"BPT block size: {args.block_size}")
    print(f"BPT patch size: {args.patch_size}")
    print("=" * 60)
    
    # Step 1: Process meshes with QuadRemesher to create quad meshes
    print("\nStep 1: Processing meshes with QuadRemesher to create quad meshes...")
    quad_script = r"D:\ClothesNetData\project_docs\blender_quadremesher_bpt_converter.py"
    
    if not os.path.exists(quad_script):
        print(f"QuadRemesher script not found: {quad_script}")
        return False
    
    # Update the script to use the correct directories
    with open(quad_script, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # Replace hardcoded paths with arguments
    script_content = script_content.replace(
        'input_directory = "D:\\\\ClothesNetData"', 
        f'input_directory = "{args.input_dir}"'
    )
    script_content = script_content.replace(
        'output_directory = "D:\\\\ClothesNetData_BPT_Ready"', 
        f'output_directory = "{args.intermediate_dir}"'
    )
    
    # Write updated script
    updated_script = r"D:\ClothesNetData\project_docs\blender_quadremesher_bpt_converter_updated.py"
    with open(updated_script, 'w') as f:
        f.write(script_content)
    
    success = run_blender_script(updated_script)
    if not success:
        print("Step 1 failed: QuadRemesher processing failed")
        return False
    else:
        print("Step 1 completed successfully: Quad meshes created")
    
    # Step 2: Convert quad meshes to BPT format
    print("\nStep 2: Converting quad meshes to BPT format...")
    bpt_script = r"D:\ClothesNetData\project_docs\bpt_converter.py"
    
    if not os.path.exists(bpt_script):
        print(f"BPT converter script not found: {bpt_script}")
        return False
    
    # Create a temporary script to run BPT conversion with specific parameters
    bpt_runner = r"D:\ClothesNetData\project_docs\bpt_runner.py"
    with open(bpt_runner, 'w') as f:
        f.write(f'''
import sys
sys.path.append(r"D:\\ClothesNetData\\project_docs")

from bpt_converter import batch_convert_to_bpt

# Run the BPT conversion with specified parameters
batch_convert_to_bpt(
    input_dir=r"{args.intermediate_dir}",
    output_dir=r"{args.output_dir}",
    block_size={args.block_size},
    patch_size={args.patch_size}
)
''')
    
    success = run_python_script(bpt_runner)
    if not success:
        print("Step 2 failed: BPT conversion failed")
        return False
    else:
        print("Step 2 completed successfully: BPT files created")
    
    # Step 3: Validation and summary
    print("\nStep 3: Validating and summarizing results...")
    
    # Count input files
    input_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.obj', '.fbx', '.stl')):
                input_files.append(os.path.join(root, file))
    
    # Count intermediate files
    intermediate_files = []
    for root, dirs, files in os.walk(args.intermediate_dir):
        for file in files:
            if file.lower().endswith(('_quad.obj', '_quad.ply')):
                intermediate_files.append(os.path.join(root, file))
    
    # Count output files
    output_files = []
    for root, dirs, files in os.walk(args.output_dir):
        for file in files:
            if file.lower().endswith('.bpt'):
                output_files.append(os.path.join(root, file))
    
    print(f"\nSUMMARY:")
    print(f"- Input mesh files: {len(input_files)}")
    print(f"- Intermediate quad files: {len(intermediate_files)}")
    print(f"- Output BPT files: {len(output_files)}")
    
    if len(output_files) > 0:
        print(f"- Success rate: {len(output_files)}/{len(input_files)} ({len(output_files)/len(input_files)*100:.1f}%)")
        print("\nPipeline completed successfully! BPT-ready garment meshes are in:", args.output_dir)
    else:
        print("\nNo BPT files were generated. Check the logs above for errors.")
        return False
    
    return True


def show_help():
    """Show help information"""
    print("""
Garment BPT Pipeline - Help Information

This pipeline converts garment meshes to BPT (Blocked and Patchified Tokenization) format:
1. Uses QuadRemesher plugin in Blender to convert irregular meshes to quad meshes
2. Optimizes meshes for BPT format compatibility
3. Converts to BPT format for neural rendering applications

Requirements:
- Blender 3.3 installed at default location
- QuadRemesher plugin installed in Blender
- Input meshes in OBJ, FBX, or STL format

Usage:
python garment_bpt_pipeline.py [OPTIONS]

Options:
  --input-dir PATH      Input directory with garment meshes (default: D:\ClothesNetData)
  --intermediate-dir PATH  Directory for intermediate quad meshes (default: D:\ClothesNetData_Processed)
  --output-dir PATH     Output directory for BPT files (default: D:\ClothesNetData_BPT_Converted)
  --block-size N        BPT block size (default: 16)
  --patch-size N        BPT patch size (default: 4)

Example:
python garment_bpt_pipeline.py --input-dir "D:\MyGarments" --block-size 32

The pipeline will:
1. Process all meshes in the input directory with QuadRemesher
2. Convert the resulting quad meshes to BPT format
3. Save BPT files in the output directory
""")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
    else:
        success = main()
        if success:
            print("\nPipeline completed successfully!")
            sys.exit(0)
        else:
            print("\nPipeline failed!")
            sys.exit(1)