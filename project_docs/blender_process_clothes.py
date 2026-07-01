import bpy
import os
import glob

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
    
    # 尝试使用Limited Dissolve简化网格，保持四边形
    bpy.ops.mesh.limited_dissolve(angle_limit=3.14159)
    
    # 转换为四边形
    bpy.ops.mesh.tris_convert_to_quads()
    
    # 返回对象模式
    bpy.ops.object.mode_set(mode='OBJECT')


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
    
    # 重网格化为四边形
    remesh_to_quads()
    
    # 导出处理后的OBJ文件
    bpy.ops.export_scene.obj(filepath=output_path, use_selection=True)
    print(f"Saved: {output_path}")


def batch_process_clothes_data(input_dir, output_dir):
    """批量处理服装数据"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有OBJ文件
    obj_files = glob.glob(os.path.join(input_dir, "**", "*.obj"), recursive=True)
    
    print(f"Found {len(obj_files)} OBJ files to process")
    
    for obj_file in obj_files:
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


def main():
    # 设置输入和输出目录
    input_directory = "D:\\ClothesNetData"
    output_directory = "D:\\ClothesNetData_Processed"
    
    print("Starting batch processing of clothing data...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    
    # 执行批量处理
    batch_process_clothes_data(input_directory, output_directory)
    
    print("Batch processing completed!")


if __name__ == "__main__":
    main()