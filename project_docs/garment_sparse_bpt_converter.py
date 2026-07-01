"""
Garment Sparse BPT Converter - Specialized for sparse quad mesh processing
This script focuses on creating sparse representations of garment meshes optimized for BPT format
"""

import bpy
import os
import glob
import mathutils
from pathlib import Path
import sys


def clear_scene():
    """清空当前场景"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_obj(filepath):
    """导入OBJ文件"""
    bpy.ops.import_scene.obj(filepath=filepath, split_mode="OFF")


def optimize_garment_sparse_quads():
    """
    专门针对服装的稀疏四边形网格优化
    使用QuadRemesher并调整参数以生成更稀疏的网格
    """
    # 确保QuadRemesher插件可用
    if "quad_remesher_1_4" in bpy.context.preferences.addons:
        print("Optimizing garment mesh with QuadRemesher for sparse quad representation...")
        
        # 切换到编辑模式
        bpy.ops.object.mode_set(mode='EDIT')
        
        # 选择所有几何体
        bpy.ops.mesh.select_all(action='SELECT')
        
        # 使用QuadRemesher进行稀疏化处理
        # 这里我们假设QuadRemesher有参数控制稀疏程度
        bpy.ops.qr.remesh()
        
        # 退出编辑模式
        bpy.ops.object.mode_set(mode='OBJECT')
        
        print("QuadRemesher optimization completed")
    else:
        print("QuadRemesher plugin not found, using built-in methods...")
        optimize_garment_sparse_quads_builtin()


def optimize_garment_sparse_quads_builtin():
    """
    使用Blender内置功能进行稀疏四边形优化
    针对服装mesh的特点进行参数调整
    """
    print("Optimizing garment mesh with built-in tools for sparse quad representation...")
    
    # 切换到编辑模式
    bpy.ops.object.mode_set(mode='EDIT')
    
    # 选择所有顶点
    bpy.ops.mesh.select_all(action='SELECT')
    
    # 合并相近顶点（使用较小阈值保持精度）
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    
    # 使用Limited Dissolve进行稀疏化
    # 对于服装，使用较大的角度限制以合并更多面
    bpy.ops.mesh.limited_dissolve(angle_limit=mathutils.radians(30.0))
    
    # 转换为四边形
    bpy.ops.mesh.tris_convert_to_quads(
        face_threshold=mathutils.radians(30.0), 
        shape_threshold=mathutils.radians(30.0)
    )
    
    # 优化网格布局
    bpy.ops.mesh.vert_connect_nonplanar(angle_limit=mathutils.radians(30.0))
    
    # 再次进行limited dissolve以进一步稀疏化
    bpy.ops.mesh.limited_dissolve(angle_limit=mathutils.radians(10.0))
    
    # 返回对象模式
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print("Built-in sparse quad optimization completed")


def calculate_garment_mesh_stats(obj):
    """计算服装网格的统计信息，特别关注稀疏化指标"""
    mesh = obj.data
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.polygons)
    
    # 统计不同类型的面
    num_tris = sum(1 for face in mesh.polygons if len(face.vertices) == 3)
    num_quads = sum(1 for face in mesh.polygons if len(face.vertices) == 4)
    num_ngons = sum(1 for face in mesh.polygons if len(face.vertices) > 4)
    
    # 计算稀疏化指标
    quad_ratio = num_quads / num_faces if num_faces > 0 else 0
    density = num_faces / (num_vertices + 1)  # 简单的密度指标
    
    print(f"Garment Mesh Statistics:")
    print(f"  Vertices: {num_vertices}")
    print(f"  Faces: {num_faces}")
    print(f"  Triangles: {num_tris} ({num_tris/num_faces*100:.1f}%)")
    print(f"  Quads: {num_quads} ({num_quads/num_faces*100:.1f}%)")
    print(f"  N-gons: {num_ngons} ({num_ngons/num_faces*100:.1f}%)")
    print(f"  Quad Ratio: {quad_ratio:.3f}")
    print(f"  Density: {density:.3f} faces/vertex")
    
    return {
        'vertices': num_vertices,
        'faces': num_faces,
        'tris': num_tris,
        'quads': num_quads,
        'ngons': num_ngons,
        'quad_ratio': quad_ratio,
        'density': density
    }


def process_garment_mesh_sparse(input_path, output_path):
    """处理单个服装网格文件，专注于稀疏化"""
    print(f"Processing garment: {input_path}")
    
    # 清空场景
    clear_scene()
    
    # 根据文件扩展名选择导入方法
    ext = Path(input_path).suffix.lower()
    if ext == '.obj':
        import_obj(input_path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=input_path)
    elif ext == '.stl':
        bpy.ops.import_mesh.stl(filepath=input_path)
    else:
        print(f"Unsupported file format: {ext}")
        return False
    
    # 获取导入的对象
    if bpy.context.selected_objects:
        obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj
        
        # 应用变换
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        # 计算原始网格统计
        print("Original garment mesh statistics:")
        orig_stats = calculate_garment_mesh_stats(obj)
        
        # 进行稀疏化处理
        optimize_garment_sparse_quads()
        
        # 计算处理后网格统计
        print("Processed sparse garment mesh statistics:")
        proc_stats = calculate_garment_mesh_stats(obj)
        
        # 导出处理后的文件
        output_obj = output_path.replace('.bpt', '_sparse_quad.obj')
        bpy.ops.export_scene.obj(
            filepath=output_obj, 
            use_selection=True, 
            use_materials=False,
            use_triangles=False,  # 保持四边形
            use_normals=True
        )
        
        print(f"Saved sparse quad mesh: {output_obj}")
        
        # 记录稀疏化改进
        vertex_reduction = (orig_stats['vertices'] - proc_stats['vertices']) / orig_stats['vertices'] * 100
        face_reduction = (orig_stats['faces'] - proc_stats['faces']) / orig_stats['faces'] * 100
        quad_improvement = proc_stats['quad_ratio'] - orig_stats['quad_ratio']
        
        print(f"Sparse optimization results:")
        print(f"  Vertex reduction: {vertex_reduction:.1f}%")
        print(f"  Face reduction: {face_reduction:.1f}%")
        print(f"  Quad ratio improvement: {quad_improvement:+.3f}")
        
        return True
    else:
        print(f"No objects imported from {input_path}")
        return False


def batch_process_garment_sparse(input_dir, output_dir):
    """批量处理服装数据，专注于稀疏化"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有支持的3D文件
    extensions = ['*.obj', '*.fbx', '*.stl']
    obj_files = []
    for ext in extensions:
        obj_files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    
    print(f"Found {len(obj_files)} garment files to process for sparse representation")
    
    successful = 0
    failed = 0
    
    for i, obj_file in enumerate(obj_files):
        print(f"Processing {i+1}/{len(obj_files)}: {obj_file}")
        
        # 获取文件名（不含扩展名）
        filename = os.path.splitext(os.path.basename(obj_file))[0]
        
        # 构造输出路径
        rel_path = os.path.relpath(obj_file, input_dir)
        output_subdir = os.path.dirname(rel_path)
        output_path = os.path.join(output_dir, output_subdir, f"{filename}")
        
        # 创建输出子目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 处理单个文件
        try:
            if process_garment_mesh_sparse(obj_file, output_path):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {obj_file}: {str(e)}")
            failed += 1
            # 即使出现错误也要继续处理下一个
            continue
    
    print(f"Sparse garment processing completed!")
    print(f"Successful: {successful}, Failed: {failed}")


def main():
    # 设置输入和输出目录
    input_directory = "D:\\ClothesNetData"
    output_directory = "D:\\ClothesNetData_Sparse_Quads"
    
    print("Starting sparse quad optimization for garment meshes...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    
    # 执行批量处理
    batch_process_garment_sparse(input_directory, output_directory)
    
    print("Sparse quad optimization completed! Ready for BPT conversion.")


if __name__ == "__main__":
    main()