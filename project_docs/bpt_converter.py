"""
BPT (Blocked and Patchified Tokenization) Converter for 3D Garment Meshes
This script converts quad-dominant meshes to BPT format suitable for neural rendering
"""

import os
import numpy as np
from typing import List, Tuple, Dict
import struct
from pathlib import Path


class BPTConverter:
    """
    Converts quad meshes to BPT (Blocked and Patchified Tokenization) format
    BPT represents 3D meshes as a collection of patches organized in blocks
    """
    
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.uv_coords = []
        self.normals = []
    
    def load_obj(self, obj_path: str):
        """Load OBJ file and extract vertices, faces, UVs, and normals"""
        vertices = []
        faces = []
        uvs = []
        normals = []
        
        with open(obj_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                if parts[0] == 'v':
                    # Vertex coordinates
                    vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'vt':
                    # Texture coordinates
                    uvs.append([float(x) for x in parts[1:3]])
                elif parts[0] == 'vn':
                    # Normal coordinates
                    normals.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'f':
                    # Face definition
                    face = []
                    for vertex_def in parts[1:]:
                        # Handle formats like "v/vt/vn", "v//vn", or "v/vt"
                        indices = vertex_def.split('/')
                        vertex_idx = int(indices[0]) - 1  # OBJ uses 1-based indexing
                        face.append(vertex_idx)
                    
                    # Only process quads (4 vertices) and triangles (3 vertices)
                    if len(face) in [3, 4]:
                        faces.append(face)
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = faces
        self.uv_coords = np.array(uvs, dtype=np.float32) if uvs else np.zeros((len(vertices), 2), dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32) if normals else np.zeros((len(vertices), 3), dtype=np.float32)
        
        print(f"Loaded mesh: {len(self.vertices)} vertices, {len(self.faces)} faces")
        
        # Validate that we have mostly quads
        quad_count = sum(1 for face in self.faces if len(face) == 4)
        tri_count = sum(1 for face in self.faces if len(face) == 3)
        total = len(self.faces)
        
        quad_ratio = quad_count / total if total > 0 else 0
        print(f"Face composition: {quad_count} quads ({quad_ratio:.2%}), {tri_count} tris")
        
        if quad_ratio < 0.8:
            print("Warning: Less than 80% quads detected. BPT works best with quad-dominant meshes.")
        
        return True
    
    def patchify_mesh(self, patch_size: int = 4) -> List[np.ndarray]:
        """
        Convert mesh faces into patches of specified size
        For quads, each face is already a patch of size 4
        For triangles, we duplicate vertices to make them quads
        """
        patches = []
        
        for face in self.faces:
            if len(face) == 4:  # Already a quad
                patches.append(np.array(face, dtype=np.int32))
            elif len(face) == 3:  # Triangle - convert to quad by duplicating vertex
                # For triangle (v0, v1, v2), create quad (v0, v1, v2, v2)
                quad = [face[0], face[1], face[2], face[2]]
                patches.append(np.array(quad, dtype=np.int32))
        
        print(f"Created {len(patches)} patches from {len(self.faces)} faces")
        return patches
    
    def blockify_patches(self, patches: List[np.ndarray], block_size: int = 16) -> List[List[np.ndarray]]:
        """
        Group patches into blocks of specified size
        Each block contains up to block_size patches
        """
        blocks = []
        
        for i in range(0, len(patches), block_size):
            block = patches[i:i+block_size]
            blocks.append(block)
        
        print(f"Created {len(blocks)} blocks with up to {block_size} patches each")
        return blocks
    
    def encode_bpt(self, block_size: int = 16, patch_size: int = 4) -> bytes:
        """
        Encode the mesh into BPT format
        Format: [header][vertex_data][block_data]
        """
        # Patchify the mesh
        patches = self.patchify_mesh(patch_size)
        
        # Blockify the patches
        blocks = self.blockify_patches(patches, block_size)
        
        # Encode to binary format
        encoded_data = bytearray()
        
        # Write header
        # Magic number for BPT format
        encoded_data.extend(b'BPT\x00')
        
        # Version (1 byte)
        encoded_data.extend(struct.pack('<B', 1))
        
        # Number of vertices (4 bytes)
        encoded_data.extend(struct.pack('<I', len(self.vertices)))
        
        # Number of blocks (4 bytes)
        encoded_data.extend(struct.pack('<I', len(blocks)))
        
        # Block size (4 bytes)
        encoded_data.extend(struct.pack('<I', block_size))
        
        # Patch size (4 bytes)
        encoded_data.extend(struct.pack('<I', patch_size))
        
        # Write vertex data
        for vertex in self.vertices:
            encoded_data.extend(struct.pack('<fff', vertex[0], vertex[1], vertex[2]))
        
        # Write block data
        for block in blocks:
            # Number of patches in this block (4 bytes)
            encoded_data.extend(struct.pack('<I', len(block)))
            
            # Write patch data
            for patch in block:
                # Write patch vertex indices
                for vertex_idx in patch:
                    encoded_data.extend(struct.pack('<I', int(vertex_idx)))
                
                # Pad to patch_size if needed (for triangles converted to quads)
                for _ in range(patch_size - len(patch)):
                    encoded_data.extend(struct.pack('<I', 0))  # Padding with 0
        
        return bytes(encoded_data)
    
    def save_bpt(self, output_path: str, block_size: int = 16, patch_size: int = 4):
        """Save the mesh in BPT format"""
        encoded_data = self.encode_bpt(block_size, patch_size)
        
        with open(output_path, 'wb') as f:
            f.write(encoded_data)
        
        print(f"BPT file saved: {output_path}, size: {len(encoded_data)} bytes")
    
    def validate_bpt_compatibility(self) -> bool:
        """Validate that the mesh is compatible with BPT format"""
        # Check if we have valid geometry
        if len(self.vertices) == 0 or len(self.faces) == 0:
            print("Error: No vertices or faces in mesh")
            return False
        
        # Check if all faces are either triangles or quads
        for face in self.faces:
            if len(face) not in [3, 4]:
                print(f"Error: Non-triangle/quadrilateral face found: {len(face)} vertices")
                return False
        
        return True


def convert_single_mesh_to_bpt(input_path: str, output_path: str, block_size: int = 16, patch_size: int = 4):
    """Convert a single mesh file to BPT format"""
    converter = BPTConverter()
    
    # Load the mesh
    if not converter.load_obj(input_path):
        print(f"Failed to load {input_path}")
        return False
    
    # Validate compatibility
    if not converter.validate_bpt_compatibility():
        print(f"Mesh {input_path} is not compatible with BPT format")
        return False
    
    # Save in BPT format
    converter.save_bpt(output_path, block_size, patch_size)
    return True


def batch_convert_to_bpt(input_dir: str, output_dir: str, block_size: int = 16, patch_size: int = 4):
    """Batch convert mesh files to BPT format"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all OBJ files
    obj_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, file))
    
    print(f"Found {len(obj_files)} OBJ files to convert")
    
    successful = 0
    failed = 0
    
    for i, obj_file in enumerate(obj_files):
        print(f"Converting {i+1}/{len(obj_files)}: {obj_file}")
        
        # Generate output path
        rel_path = os.path.relpath(obj_file, input_dir)
        output_path = os.path.join(output_dir, rel_path.replace('.obj', '_quad.bpt'))
        
        # Create output subdirectory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert the file
        try:
            if convert_single_mesh_to_bpt(obj_file, output_path, block_size, patch_size):
                successful += 1
                print(f"  -> Saved: {output_path}")
            else:
                failed += 1
                print(f"  -> FAILED: {obj_file}")
        except Exception as e:
            failed += 1
            print(f"  -> ERROR processing {obj_file}: {str(e)}")
    
    print(f"BPT conversion completed!")
    print(f"Successful: {successful}, Failed: {failed}")


def main():
    """Main function to demonstrate BPT conversion"""
    input_directory = "D:\\ClothesNetData_Processed"
    output_directory = "D:\\ClothesNetData_BPT_Converted"
    
    print("Starting BPT conversion for garment meshes...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    
    # Perform batch conversion
    batch_convert_to_bpt(input_directory, output_directory)
    
    print("BPT conversion completed!")


if __name__ == "__main__":
    main()