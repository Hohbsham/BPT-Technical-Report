import bpy
import os
import glob
import mathutils
from pathlib import Path


def clear_scene():
    """清空当前场景"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_obj(filepath):
    """导入OBJ文件"""
    bpy.ops.import_scene.obj(filepath=filepath, split_mode="OFF")


def remesh_to_quads():
    """将网格重网格化为四边形"""
    # 切换到编辑模式
    bpy.ops.object.mode_set(mode='EDIT')
    
    # 选择所有顶点
    bpy.ops.mesh.select_all(action='SELECT')
    
    # 合并相近顶点
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    
    # 尝试使用Limited Dissolve简化网格，保持四边形
    bpy.ops.mesh.limited_dissolve(angle_limit=mathutils.radians(180))
    
    # 转换为四边形
    bpy.ops.mesh.tris_convert_to_quads(face_threshold=mathutils.radians(30), shape_threshold=mathutils.radians(30))
    
    # 优化网格布局
    bpy.ops.mesh.vert_connect_nonplanar(angle_limit=mathutils.radians(30))
    
    # 返回对象模式
    bpy.ops.object.mode_set(mode='OBJECT')


def apply_decimate_modifier(obj, ratio=0.5):
    """应用简化修改器"""
    # 添加Decimate修改器
    decimate_mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
    decimate_mod.ratio = ratio
    decimate_mod.decimate_type = 'COLLAPSE'
    
    # 应用修改器
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=decimate_mod.name)


def process_single_mesh(input_path, output_path):
    """处理单个网格文件"""
    print(f"Processing: {input_path}")
    
    # 清空场景
    clear_scene()
    
    # 导入OBJ文件
    import_obj(input_path)
    
    # 获取导入的对象
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    
    # 应用变换
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # 重网格化为四边形
    remesh_to_quads()
    
    # 可选：应用简化
    # apply_decimate_modifier(obj, ratio=0.7)  # 保留70%的面
    
    # 导出处理后的OBJ文件
    bpy.ops.export_scene.obj(filepath=output_path, use_selection=True, use_materials=False)
    print(f"Saved: {output_path}")


def batch_process_clothes_data(input_dir, output_dir):
    """批量处理服装数据"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有OBJ文件
    obj_files = glob.glob(os.path.join(input_dir, "**", "*.obj"), recursive=True)
    
    print(f"Found {len(obj_files)} OBJ files to process")
    
    for i, obj_file in enumerate(obj_files):
        print(f"Processing {i+1}/{len(obj_files)}: {obj_file}")
        
        # 获取文件名（不含扩展名）
        filename = os.path.splitext(os.path.basename(obj_file))[0]
        
        # 构造输出路径
        rel_path = os.path.relpath(obj_file, input_dir)
        output_subdir = os.path.dirname(rel_path)
        output_path = os.path.join(output_dir, output_subdir, f"{filename}_quad.obj")
        
        # 创建输出子目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 处理单个文件
        try:
            process_single_mesh(obj_file, output_path)
        except Exception as e:
            print(f"Error processing {obj_file}: {str(e)}")
            
            # 即使出现错误也要继续处理下一个
            continue


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
    output_directory = "D:\\ClothesNetData_Processed"
    
    print("Starting batch processing of clothing data...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    
    # 设置场景
    setup_scene_for_processing()
    
    # 执行批量处理
    batch_process_clothes_data(input_directory, output_directory)
    
    print("Batch processing completed!")


if __name__ == "__main__":
    main()