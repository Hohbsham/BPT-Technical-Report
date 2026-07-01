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


def import_fbx(filepath):
    """导入FBX文件"""
    bpy.ops.import_scene.fbx(filepath=filepath)


def import_stl(filepath):
    """导入STL文件"""
    bpy.ops.import_mesh.stl(filepath=filepath)


def remesh_with_quadremesher():
    """使用QuadRemesher插件处理网格"""
    # 确保QuadRemesher插件已启用
    if "quad_remesher_1_4" in bpy.context.preferences.addons:
        print("Using QuadRemesher for mesh conversion...")
        
        # 切换到编辑模式
        bpy.ops.object.mode_set(mode='EDIT')
        
        # 选择所有几何体
        bpy.ops.mesh.select_all(action='SELECT')
        
        # 使用QuadRemesher
        bpy.ops.qr.remesh()
        
        # 返回对象模式
        bpy.ops.object.mode_set(mode='OBJECT')
    else:
        print("QuadRemesher plugin not found, falling back to built-in methods...")
        remesh_to_quads_builtin()


def remesh_to_quads_builtin():
    """使用Blender内置功能将网格重网格化为四边形"""
    # 切换到编辑模式
    bpy.ops.object.mode_set(mode='EDIT')
    
    # 选择所有顶点
    bpy.ops.mesh.select_all(action='SELECT')
    
    # 合并相近顶点
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    
    # 尝试使用Limited Dissolve简化网格，保持四边形
    bpy.ops.mesh.limited_dissolve(angle_limit=mathutils.radians(180))
    
    # 转换为四边形
    bpy.ops.mesh.tris_convert_to_quads(face_threshold=mathutils.radians(30), 
                                     shape_threshold=mathutils.radians(30))
    
    # 优化网格布局
    bpy.ops.mesh.vert_connect_nonplanar(angle_limit=mathutils.radians(30))
    
    # 返回对象模式
    bpy.ops.object.mode_set(mode='OBJECT')


def optimize_for_bpt():
    """优化网格以适配BPT（Blocked and Patchified Tokenization）格式"""
    # 切换到编辑模式
    bpy.ops.object.mode_set(mode='EDIT')
    
    # 选择所有
    bpy.ops.mesh.select_all(action='SELECT')
    
    # 确保所有面都是四边形（BPT需要四边形网格）
    bpy.ops.mesh.tris_convert_to_quads()
    
    # 检查是否有多边形面（超过4个顶点的面）
    bpy.ops.mesh.select_face_by_sides(number=4, type='NOTEQUAL')
    selected_faces = bpy.context.selected_objects
    if selected_faces:
        print("Warning: Found non-quad faces that may need manual correction")
    
    # 返回对象模式
    bpy.ops.object.mode_set(mode='OBJECT')


def calculate_mesh_statistics(obj):
    """计算网格统计信息"""
    mesh = obj.data
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.polygons)
    num_tris = sum(len(face.vertices) for face in mesh.polygons if len(face.vertices) == 3)
    num_quads = sum(len(face.vertices) for face in mesh.polygons if len(face.vertices) == 4)
    num_ngons = sum(len(face.vertices) for face in mesh.polygons if len(face.vertices) > 4)
    
    print(f"Mesh Statistics:")
    print(f"  Vertices: {num_vertices}")
    print(f"  Faces: {num_faces}")
    print(f"  Triangles: {num_tris}")
    print(f"  Quads: {num_quads}")
    print(f"  N-gons: {num_ngons}")
    
    return {
        'vertices': num_vertices,
        'faces': num_faces,
        'tris': num_tris,
        'quads': num_quads,
        'ngons': num_ngons
    }


def process_single_mesh(input_path, output_path):
    """处理单个网格文件"""
    print(f"Processing: {input_path}")
    
    # 清空场景
    clear_scene()
    
    # 根据文件扩展名选择导入方法
    ext = Path(input_path).suffix.lower()
    if ext == '.obj':
        import_obj(input_path)
    elif ext == '.fbx':
        import_fbx(input_path)
    elif ext == '.stl':
        import_stl(input_path)
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
        print("Original mesh statistics:")
        orig_stats = calculate_mesh_statistics(obj)
        
        # 使用QuadRemesher处理（如果可用）
        remesh_with_quadremesher()
        
        # 额外优化以适配BPT格式
        optimize_for_bpt()
        
        # 计算处理后网格统计
        print("Processed mesh statistics:")
        proc_stats = calculate_mesh_statistics(obj)
        
        # 导出处理后的文件 - 同时导出OBJ和PLY格式
        output_obj = output_path.replace('.ply', '_quad.obj').replace('.fbx', '_quad.obj')
        output_ply = output_path.replace('.obj', '_quad.ply').replace('.fbx', '_quad.ply')
        
        # 导出OBJ
        bpy.ops.export_scene.obj(
            filepath=output_obj, 
            use_selection=True, 
            use_materials=False,
            use_triangles=False,  # 保持四边形
            use_normals=True
        )
        
        # 导出PLY（如果需要）
        if proc_stats['tris'] == 0 and proc_stats['ngons'] == 0:  # 只有四边形时导出PLY
            bpy.ops.export_mesh.ply(filepath=output_ply, use_ascii=True)
        
        print(f"Saved OBJ: {output_obj}")
        if os.path.exists(output_ply):
            print(f"Saved PLY: {output_ply}")
        
        return True
    else:
        print(f"No objects imported from {input_path}")
        return False


def batch_process_clothes_data(input_dir, output_dir):
    """批量处理服装数据"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有支持的3D文件
    extensions = ['*.obj', '*.fbx', '*.stl']
    obj_files = []
    for ext in extensions:
        obj_files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    
    print(f"Found {len(obj_files)} 3D files to process")
    
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
            if process_single_mesh(obj_file, output_path):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {obj_file}: {str(e)}")
            failed += 1
            # 即使出现错误也要继续处理下一个
            continue
    
    print(f"Batch processing completed!")
    print(f"Successful: {successful}, Failed: {failed}")


def setup_scene_for_processing():
    """设置场景以便更好地处理网格"""
    # 设置单位
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.0
    
    # 设置网格显示选项
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.overlay.show_wireframes = True
                    break


def main():
    # 设置输入和输出目录
    input_directory = "D:\\ClothesNetData"
    output_directory = "D:\\ClothesNetData_BPT_Ready"
    
    print("Starting batch processing for BPT-ready quad meshes...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    
    # 设置场景
    setup_scene_for_processing()
    
    # 执行批量处理
    batch_process_clothes_data(input_directory, output_directory)
    
    print("All processing completed! Ready for BPT conversion.")


if __name__ == "__main__":
    main()