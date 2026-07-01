"""
Garment Sparse BPT Converter - Fixed version with improved error handling
This script focuses on creating sparse representations of garment meshes optimized for BPT format
"""

import bpy
import os
import glob
import mathutils
from pathlib import Path
import sys
import traceback


def clear_scene():
    """清空当前场景，增加错误处理"""
    try:
        # 检查场景中是否有对象
        if bpy.context.scene.objects:
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            print("Scene cleared successfully")
        else:
            print("Scene was already empty")
    except Exception as e:
        print(f"Warning: Could not clear scene: {str(e)}")


def import_obj(filepath):
    """导入OBJ文件，增加错误处理和验证"""
    try:
        # 检查文件是否存在
        if not os.path.exists(filepath):
            print(f"Error: File does not exist: {filepath}")
            return False
            
        # 记录导入前的对象数量
        obj_count_before = len(bpy.context.scene.objects)
        
        bpy.ops.import_scene.obj(filepath=filepath, split_mode="OFF")
        
        # 验证导入是否成功
        obj_count_after = len(bpy.context.scene.objects)
        if obj_count_after <= obj_count_before:
            print(f"Warning: No objects were imported from {filepath}")
            return False
            
        print(f"Successfully imported {obj_count_after - obj_count_before} objects from {filepath}")
        return True
    except Exception as e:
        print(f"Error importing {filepath}: {str(e)}")
        traceback.print_exc()
        return False


def ensure_edit_mode():
    """确保进入编辑模式，如果当前不在编辑模式则切换"""
    try:
        # 检查是否有活动对象
        if bpy.context.active_object is None:
            if bpy.context.selected_objects:
                bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
            else:
                print("Error: No active or selected object to enter edit mode")
                return False

        # 如果已经在编辑模式，无需切换
        if bpy.context.mode != 'EDIT_MESH':
            bpy.ops.object.mode_set(mode='EDIT')
        return True
    except Exception as e:
        print(f"Error entering edit mode: {str(e)}")
        return False


def ensure_object_mode():
    """确保进入对象模式"""
    try:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        return True
    except Exception as e:
        print(f"Error entering object mode: {str(e)}")
        return False


def optimize_garment_sparse_quads():
    """
    专门针对服装的稀疏四边形网格优化
    使用QuadRemesher并调整参数以生成更稀疏的网格
    """
    try:
        # 检查是否有活动对象
        if bpy.context.active_object is None:
            print("Error: No active object to optimize")
            return False

        # 确保在编辑模式下
        if not ensure_edit_mode():
            return False

        # 检查是否有选中的几何体
        bpy.ops.mesh.select_all(action='SELECT')
        selected_verts = bpy.context.tool_settings.mesh_select_mode[0]  # vertex selection mode
        # Actually check if geometry is selected by checking the selection state
        selected_count = sum(1 for v in bpy.context.active_object.data.vertices if v.select)
        if selected_count == 0:
            print("Warning: No geometry selected for optimization")
            return False

        # 确保QuadRemesher插件可用
        if "quad_remesher_1_4" in bpy.context.preferences.addons:
            print("Optimizing garment mesh with QuadRemesher for sparse quad representation...")
            
            try:
                bpy.ops.qr.remesh()
                print("QuadRemesher optimization completed")
            except Exception as e:
                print(f"QuadRemesher failed: {str(e)}, falling back to built-in methods")
                # Fallback to built-in methods
                optimize_garment_sparse_quads_builtin()
        else:
            print("QuadRemesher plugin not found, using built-in methods...")
            optimize_garment_sparse_quads_builtin()

        # 返回对象模式
        ensure_object_mode()
        return True
    except Exception as e:
        print(f"Error in optimize_garment_sparse_quads: {str(e)}")
        traceback.print_exc()
        # Try to return to object mode anyway
        try:
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        except:
            pass  # Ignore errors when trying to exit mode
        return False


def optimize_garment_sparse_quads_builtin():
    """
    使用Blender内置功能进行稀疏四边形优化
    针对服装mesh的特点进行参数调?    """
    try:
        print("Optimizing garment mesh with built-in tools for sparse quad representation...")
        
        # 确保在编辑模式下
        if not ensure_edit_mode():
            return False

        # 检查是否有几何体可选择
        bpy.ops.mesh.select_all(action='SELECT')
        selected_count = sum(1 for v in bpy.context.active_object.data.vertices if v.select)
        if selected_count == 0:
            print("Warning: No geometry to optimize in built-in method")
            return False

        # 合并相近顶点（使用较小阈值保持精度）
        try:
            bpy.ops.mesh.remove_doubles(threshold=0.0001)
        except Exception as e:
            print(f"Warning: remove_doubles failed: {str(e)}")

        # 使用Limited Dissolve进行稀疏化
        # 对于服装，使用较大的角度限制以合并更多面
        try:
            bpy.ops.mesh.limited_dissolve(angle_limit=mathutils.radians(30.0))
        except Exception as e:
            print(f"Warning: limited_dissolve failed: {str(e)}")

        # 转换为四边形
        try:
            bpy.ops.mesh.tris_convert_to_quads(
                face_threshold=mathutils.radians(30.0), 
                shape_threshold=mathutils.radians(30.0)
            )
        except Exception as e:
            print(f"Warning: tris_convert_to_quads failed: {str(e)}")

        # 优化网格布局
        try:
            bpy.ops.mesh.vert_connect_nonplanar(angle_limit=mathutils.radians(30.0))
        except Exception as e:
            print(f"Warning: vert_connect_nonplanar failed: {str(e)}")

        # 再次进行limited dissolve以进一步稀疏化
        try:
            bpy.ops.mesh.limited_dissolve(angle_limit=mathutils.radians(10.0))
        except Exception as e:
            print(f"Warning: final limited_dissolve failed: {str(e)}")

        # 返回对象模式
        ensure_object_mode()
        print("Built-in sparse quad optimization completed")
        return True
    except Exception as e:
        print(f"Error in optimize_garment_sparse_quads_builtin: {str(e)}")
        traceback.print_exc()
        # Try to return to object mode anyway
        try:
            bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        except:
            pass  # Ignore errors when trying to exit mode
        return False


def calculate_garment_mesh_stats(obj):
    """计算服装网格的统计信息，特别关注稀疏化指标"""
    try:
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
    except Exception as e:
        print(f"Error calculating mesh stats: {str(e)}")
        traceback.print_exc()
        return {
            'vertices': 0,
            'faces': 0,
            'tris': 0,
            'quads': 0,
            'ngons': 0,
            'quad_ratio': 0.0,
            'density': 0.0
        }


def process_garment_mesh_sparse(input_path, output_path):
    """处理单个服装网格文件，专注于稀疏化"""
    print(f"Processing garment: {input_path}")
    
    try:
        # 清空场景
        clear_scene()
        
        # 根据文件扩展名选择导入方法
        ext = Path(input_path).suffix.lower()
        import_success = False
        if ext == '.obj':
            import_success = import_obj(input_path)
        elif ext == '.fbx':
            try:
                bpy.ops.import_scene.fbx(filepath=input_path)
                import_success = len(bpy.context.scene.objects) > 0
            except Exception as e:
                print(f"Error importing FBX {input_path}: {str(e)}")
        elif ext == '.stl':
            try:
                bpy.ops.import_mesh.stl(filepath=input_path)
                import_success = len(bpy.context.scene.objects) > 0
            except Exception as e:
                print(f"Error importing STL {input_path}: {str(e)}")
        else:
            print(f"Unsupported file format: {ext}")
            return False
        
        if not import_success:
            print(f"Failed to import {input_path}")
            return False
        
        # 获取导入的对象
        imported_objects = bpy.context.selected_objects
        if not imported_objects:
            print(f"No objects imported from {input_path}")
            return False

        # 设置第一个导入的对象为活动对象
        obj = imported_objects[0]
        bpy.context.view_layer.objects.active = obj

        # 应用变换
        try:
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        except Exception as e:
            print(f"Warning: Could not apply transforms: {str(e)}")
        
        # 计算原始网格统计
        print("Original garment mesh statistics:")
        orig_stats = calculate_garment_mesh_stats(obj)
        
        # 进行稀疏化处理
        if not optimize_garment_sparse_quads():
            print(f"Failed to optimize {input_path}")
            return False
        
        # 计算处理后网格统计
        print("Processed sparse garment mesh statistics:")
        proc_stats = calculate_garment_mesh_stats(obj)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出处理后的文件
        output_obj = output_path.replace('.bpt', '_sparse_quad.obj')
        try:
            bpy.ops.export_scene.obj(
                filepath=output_obj, 
                use_selection=True, 
                use_materials=False,
                use_triangles=False,  # 保持四边形
                use_normals=True
            )
            print(f"Saved sparse quad mesh: {output_obj}")
        except Exception as e:
            print(f"Error exporting {output_obj}: {str(e)}")
            return False
        
        # 记录稀疏化改进
        if orig_stats['vertices'] > 0 and orig_stats['faces'] > 0:
            vertex_reduction = (orig_stats['vertices'] - proc_stats['vertices']) / orig_stats['vertices'] * 100
            face_reduction = (orig_stats['faces'] - proc_stats['faces']) / orig_stats['faces'] * 100
            quad_improvement = proc_stats['quad_ratio'] - orig_stats['quad_ratio']
            
            print(f"Sparse optimization results:")
            print(f"  Vertex reduction: {vertex_reduction:.1f}%")
            print(f"  Face reduction: {face_reduction:.1f}%")
            print(f"  Quad ratio improvement: {quad_improvement:+.3f}")
        else:
            print("Could not calculate optimization results due to missing original stats")
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        traceback.print_exc()
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
    failed_files = []  # Track failed files for debugging
    
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
                failed_files.append(obj_file)
        except Exception as e:
            print(f"Exception processing {obj_file}: {str(e)}")
            traceback.print_exc()
            failed += 1
            failed_files.append(obj_file)
            # 即使出现异常也要继续处理下一个
            continue
    
    # 记录失败的文件
    if failed_files:
        failed_log_path = os.path.join(output_dir, "failed_files.log")
        with open(failed_log_path, 'w') as f:
            f.write("# Failed Files Log\n")
            f.write(f"Total failed: {len(failed_files)}\n\n")
            for failed_file in failed_files:
                f.write(f"{failed_file}\n")
        print(f"List of failed files saved to: {failed_log_path}")
    
    print(f"Sparse garment processing completed!")
    print(f"Successful: {successful}, Failed: {failed}")
    return successful, failed


def main():
    # 设置输入和输出目录
    input_directory = "D:\\ClothesNetData"
    output_directory = "D:\\ClothesNetData_Sparse_Quads_Fixed"
    
    print("Starting sparse quad optimization for garment meshes...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    
    # 执行批量处理
    successful, failed = batch_process_garment_sparse(input_directory, output_directory)
    
    print(f"Processing completed! Success: {successful}, Failed: {failed}")
    if failed == 0:
        print("All files processed successfully! Ready for BPT conversion.")
    else:
        print(f"Some files failed. Check the output directory for failed_files.log")


if __name__ == "__main__":
    main()