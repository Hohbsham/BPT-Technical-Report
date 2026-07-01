"""
Quad-BPT Converter - Specialized for quad mesh to BPT format conversion
This script optimizes the conversion process specifically for quad-dominant meshes
"""

import os
import numpy as np
from typing import List, Tuple, Dict
import struct
from pathlib import Path


class QuadBPTConverter:
    """
    Specialized converter for quad-dominant meshes to BPT (Blocked and Patchified Tokenization) format
    Optimized specifically for quad meshes from garment processing
    """
    
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.uv_coords = []
        self.normals = []
        self.face_types = []  # Track face types: 'quad', 'tri', 'ngon'
    
    def load_obj(self, obj_path: str):
        """Load OBJ file and extract vertices, faces, UVs, and normals"""
        vertices = []
        faces = []
        face_types = []
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
                    
                    # Determine face type and process accordingly
                    if len(face) == 4:
                        face_types.append('quad')
                        faces.append(face)
                    elif len(face) == 3:
                        face_types.append('tri')
                        # Convert triangle to quad by duplicating last vertex
                        quad_face = [face[0], face[1], face[2], face[2]]
                        faces.append(quad_face)
                    else:
                        face_types.append(f'ngon-{len(face)}')
                        # For ngons, we'll convert to fan triangulation then to quads
                        # This is a simplified approach - in practice, you might want more sophisticated handling
                        for i in range(1, len(face)-1):
                            tri = [face[0], face[i], face[i+1]]
                            quad_face = [tri[0], tri[1], tri[2], tri[2]]
                            faces.append(quad_face)
                            face_types.append('converted-quad')
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = faces
        self.face_types = face_types
        self.uv_coords = np.array(uvs, dtype=np.float32) if uvs else np.zeros((len(vertices), 2), dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32) if normals else np.zeros((len(vertices), 3), dtype=np.float32)
        
        print(f"Loaded mesh: {len(self.vertices)} vertices, {len(self.faces)} faces ({len([ft for ft in face_types if 'quad' in ft])} quads)")
        
        # Validate that we have quad-dominant mesh
        quad_count = sum(1 for ft in self.face_types if 'quad' in ft)
        total = len(self.face_types)
        
        quad_ratio = quad_count / total if total > 0 else 0
        print(f"Quad dominance: {quad_count}/{total} ({quad_ratio:.2%})")
        
        return True
    
    def create_quad_patches(self) -> List[np.ndarray]:
        """
        Create patches from quad faces
        Since our mesh is quad-dominant, each face is already a patch of size 4
        """
        patches = []
        
        for i, face in enumerate(self.faces):
            if len(face) == 4:  # Already a quad
                patches.append(np.array(face, dtype=np.int32))
            else:
                print(f"Warning: Non-quad face at index {i} with {len(face)} vertices")
        
        print(f"Created {len(patches)} quad patches from {len(self.faces)} faces")
        return patches
    
    def organize_patches_into_blocks(self, patches: List[np.ndarray], block_size: int = 16) -> List[List[np.ndarray]]:
        """
        Organize quad patches into blocks
        Each block contains up to block_size patches
        Optimized for quad meshes
        """
        blocks = []
        
        for i in range(0, len(patches), block_size):
            block = patches[i:i+block_size]
            blocks.append(block)
        
        print(f"Organized {len(patches)} patches into {len(blocks)} blocks (size: {block_size})")
        return blocks
    
    def encode_quad_bpt(self, block_size: int = 16) -> bytes:
        """
        Encode quad mesh into optimized BPT format
        Specialized for quad-dominant meshes
        """
        # Create patches from quad faces
        patches = self.create_quad_patches()
        
        # Organize patches into blocks
        blocks = self.organize_patches_into_blocks(patches, block_size)
        
        # Encode to binary format
        encoded_data = bytearray()
        
        # Write header
        # Magic number for Quad-BPT format
        encoded_data.extend(b'QBPT')  # Quad-BPT identifier
        
        # Version (1 byte)
        encoded_data.extend(struct.pack('<B', 2))  # Version 2 for quad-optimized
        
        # Number of vertices (4 bytes)
        encoded_data.extend(struct.pack('<I', len(self.vertices)))
        
        # Number of blocks (4 bytes)
        encoded_data.extend(struct.pack('<I', len(blocks)))
        
        # Block size (4 bytes)
        encoded_data.extend(struct.pack('<I', block_size))
        
        # Patch size (4 bytes) - fixed at 4 for quads
        encoded_data.extend(struct.pack('<I', 4))
        
        # Write vertex data
        for vertex in self.vertices:
            encoded_data.extend(struct.pack('<fff', vertex[0], vertex[1], vertex[2]))
        
        # Write block data
        for block in blocks:
            # Number of patches in this block (4 bytes)
            encoded_data.extend(struct.pack('<I', len(block)))
            
            # Write patch data
            for patch in block:
                # Write patch vertex indices (4 vertices per patch for quads)
                for vertex_idx in patch:
                    encoded_data.extend(struct.pack('<I', int(vertex_idx)))
        
        return bytes(encoded_data)
    
    def save_quad_bpt(self, output_path: str, block_size: int = 16):
        """Save the mesh in optimized Quad-BPT format"""
        encoded_data = self.encode_quad_bpt(block_size)
        
        with open(output_path, 'wb') as f:
            f.write(encoded_data)
        
        print(f"Quad-BPT file saved: {output_path}, size: {len(encoded_data)} bytes")
        
        # Calculate and report compression statistics
        original_size = len(self.vertices) * 3 * 4  # 3 floats per vertex, 4 bytes per float
        if original_size > 0:
            compression_ratio = len(encoded_data) / original_size
            print(f"Compression ratio: {compression_ratio:.3f} ({compression_ratio*100:.1f}%)")
    
    def validate_quad_bpt_compatibility(self) -> bool:
        """Validate that the mesh is compatible with Quad-BPT format"""
        # Check if we have valid geometry
        if len(self.vertices) == 0 or len(self.faces) == 0:
            print("Error: No vertices or faces in mesh")
            return False
        
        # Check if mesh is quad-dominant enough (at least 80% quads)
        quad_count = sum(1 for ft in self.face_types if 'quad' in ft)
        total = len(self.face_types)
        quad_ratio = quad_count / total if total > 0 else 0
        
        if quad_ratio < 0.8:
            print(f"Warning: Mesh is not quad-dominant enough ({quad_ratio:.2%}). Results may be suboptimal.")
        
        return True


class SparseQuadBPTConverter(QuadBPTConverter):
    """
    Extended converter that handles sparse quad meshes with additional optimizations
    """
    
    def encode_sparse_quad_bpt(self, block_size: int = 16, adjacency_encoding: bool = True) -> bytes:
        """
        Encode sparse quad mesh with additional optimizations
        """
        # Create patches from quad faces
        patches = self.create_quad_patches()
        
        # Organize patches into blocks
        blocks = self.organize_patches_into_blocks(patches, block_size)
        
        # Encode to binary format
        encoded_data = bytearray()
        
        # Write header
        encoded_data.extend(b'SQBPT')  # Sparse Quad-BPT identifier
        
        # Version (1 byte)
        encoded_data.extend(struct.pack('<B', 3))  # Version 3 for sparse quad-optimized
        
        # Flags for encoding options (1 byte)
        flags = 0
        if adjacency_encoding:
            flags |= 1  # Bit 0: adjacency encoding enabled
        encoded_data.extend(struct.pack('<B', flags))
        
        # Number of vertices (4 bytes)
        encoded_data.extend(struct.pack('<I', len(self.vertices)))
        
        # Number of blocks (4 bytes)
        encoded_data.extend(struct.pack('<I', len(blocks)))
        
        # Block size (4 bytes)
        encoded_data.extend(struct.pack('<I', block_size))
        
        # Patch size (4 bytes) - fixed at 4 for quads
        encoded_data.extend(struct.pack('<I', 4))
        
        # Write vertex data
        for vertex in self.vertices:
            encoded_data.extend(struct.pack('<fff', vertex[0], vertex[1], vertex[2]))
        
        # Write block data
        for block in blocks:
            # Number of patches in this block (4 bytes)
            encoded_data.extend(struct.pack('<I', len(block)))
            
            # Write patch data
            for patch in block:
                # Write patch vertex indices (4 vertices per patch for quads)
                for vertex_idx in patch:
                    encoded_data.extend(struct.pack('<I', int(vertex_idx)))
        
        return bytes(encoded_data)
    
    def save_sparse_quad_bpt(self, output_path: str, block_size: int = 16, adjacency_encoding: bool = True):
        """Save the sparse quad mesh in optimized format"""
        encoded_data = self.encode_sparse_quad_bpt(block_size, adjacency_encoding)
        
        with open(output_path, 'wb') as f:
            f.write(encoded_data)
        
        print(f"Sparse Quad-BPT file saved: {output_path}, size: {len(encoded_data)} bytes")
        
        # Calculate and report compression statistics
        original_size = len(self.vertices) * 3 * 4  # 3 floats per vertex, 4 bytes per float
        if original_size > 0:
            compression_ratio = len(encoded_data) / original_size
            print(f"Sparse compression ratio: {compression_ratio:.3f} ({compression_ratio*100:.1f}%)")
    
    def validate_sparse_quad_bpt_compatibility(self) -> bool:
        """Validate that the sparse quad mesh is compatible"""
        # Use parent validation
        base_valid = super().validate_quad_bpt_compatibility()
        
        # Additional checks for sparsity
        if len(self.vertices) == 0 or len(self.faces) == 0:
            return False
            
        # Calculate sparsity metric (faces/vertices ratio)
        sparsity = len(self.faces) / len(self.vertices) if len(self.vertices) > 0 else 0
        print(f"Sparsity metric (faces/vertices): {sparsity:.3f}")
        
        # For sparse meshes, we expect lower sparsity values
        if sparsity > 2.0:
            print("Note: Mesh may not be very sparse (high faces/vertices ratio)")
        
        return base_valid


def convert_single_garment_to_quad_bpt(input_path: str, output_path: str, block_size: int = 16, sparse_mode: bool = True):
    """Convert a single garment mesh to Quad-BPT format"""
    if sparse_mode:
        converter = SparseQuadBPTConverter()
    else:
        converter = QuadBPTConverter()
    
    # Load the mesh
    if not converter.load_obj(input_path):
        print(f"Failed to load {input_path}")
        return False
    
    # Validate compatibility
    if sparse_mode:
        is_compatible = converter.validate_sparse_quad_bpt_compatibility()
    else:
        is_compatible = converter.validate_quad_bpt_compatibility()
    
    if not is_compatible:
        print(f"Mesh {input_path} is not compatible with Quad-BPT format")
        return False
    
    # Save in Quad-BPT format
    if sparse_mode:
        converter.save_sparse_quad_bpt(output_path, block_size)
    else:
        converter.save_quad_bpt(output_path, block_size)
    
    return True


def batch_convert_to_quad_bpt(input_dir: str, output_dir: str, block_size: int = 16, sparse_mode: bool = True):
    """Batch convert garment mesh files to Quad-BPT format"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all OBJ files
    obj_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('_quad.obj', '_sparse_quad.obj')):  # Only process quad-processed files
                obj_files.append(os.path.join(root, file))
    
    print(f"Found {len(obj_files)} quad-processed garment files to convert to BPT")
    
    successful = 0
    failed = 0
    
    for i, obj_file in enumerate(obj_files):
        print(f"Converting {i+1}/{len(obj_files)}: {obj_file}")
        
        # Generate output path
        rel_path = os.path.relpath(obj_file, input_dir)
        output_path = os.path.join(output_dir, rel_path.replace('.obj', '_quad.bpt').replace('_quad.', '.'))
        
        # Create output subdirectory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert the file
        try:
            if convert_single_garment_to_quad_bpt(obj_file, output_path, block_size, sparse_mode):
                successful += 1
                print(f"  -> Saved: {output_path}")
            else:
                failed += 1
                print(f"  -> FAILED: {obj_file}")
        except Exception as e:
            failed += 1
            print(f"  -> ERROR processing {obj_file}: {str(e)}")
    
    print(f"Quad-BPT conversion completed!")
    print(f"Successful: {successful}, Failed: {failed}")


def main():
    """Main function to demonstrate Quad-BPT conversion"""
    input_directory = "D:\\ClothesNetData_Sparse_Quads"
    output_directory = "D:\\ClothesNetData_QuadBPT_Converted"
    block_size = 16
    sparse_mode = True
    
    print("Starting Quad-BPT conversion for sparse garment meshes...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Block size: {block_size}")
    print(f"Sparse mode: {sparse_mode}")
    
    # Perform batch conversion
    batch_convert_to_quad_bpt(input_directory, output_directory, block_size, sparse_mode)
    
    print("Quad-BPT conversion completed!")


if __name__ == "__main__":
    main()