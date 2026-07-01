# visual_verify.py
import os
import sys
import subprocess
from pathlib import Path

class VisualVerifier:
    def __init__(self):
        pass
    
    def verify_conversion_success(self, original_path, processed_path):
        """验证转换成功性 - 通过比较文件属性"""
        print(f"🔍 验证转换结果:")
        print(f"   原始文件: {original_path}")
        print(f"   处理文件: {processed_path}")
        
        if not os.path.exists(original_path):
            print(f"❌ 原始文件不存在: {original_path}")
            return False
        
        if not os.path.exists(processed_path):
            print(f"❌ 处理文件不存在: {processed_path}")
            return False
        
        # 比较文件大小
        orig_size = os.path.getsize(original_path)
        proc_size = os.path.getsize(processed_path)
        
        print(f"   原始大小: {orig_size:,} bytes")
        print(f"   处理大小: {proc_size:,} bytes")
        
        # 对于OBJ文件，处理后应该有合理的大小变化
        if original_path.endswith('.obj'):
            # 检查是否为有效的OBJ文件（包含基本的几何信息）
            return self._validate_obj_content(processed_path)
        elif processed_path.endswith('.bpt'):
            # 对于BPT文件，检查是否有合理的最小大小
            return proc_size > 100  # BPT文件至少应该有100字节
        
        return True
    
    def _validate_obj_content(self, obj_path):
        """验证OBJ文件内容的有效性"""
        try:
            with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 检查是否包含基本的几何元素
            has_vertices = any(line.startswith('v ') for line in lines)
            has_faces = any(line.startswith('f ') for line in lines)
            has_normals = any(line.startswith('vn ') for line in lines)
            has_texcoords = any(line.startswith('vt ') for line in lines)
            
            element_counts = {
                'vertices': sum(1 for line in lines if line.startswith('v ') and not line.startswith('vt ') and not line.startswith('vn ')),
                'texcoords': sum(1 for line in lines if line.startswith('vt ')),
                'normals': sum(1 for line in lines if line.startswith('vn ')),
                'faces': sum(1 for line in lines if line.startswith('f '))
            }
            
            print(f"   几何元素统计: {element_counts}")
            
            # 至少需要顶点和面
            success = has_vertices and has_faces and element_counts['faces'] > 0
            if success:
                print("   ✅ OBJ文件内容有效")
            else:
                print("   ❌ OBJ文件内容不完整")
            
            return success
            
        except Exception as e:
            print(f"   ❌ OBJ文件验证失败: {str(e)}")
            return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python visual_verify.py <原始文件> <处理文件>")
        sys.exit(1)
    
    verifier = VisualVerifier()
    result = verifier.verify_conversion_success(sys.argv[1], sys.argv[2])
    
    print(f"验证结果: {'✅ 通过' if result else '❌ 失败'}")
    sys.exit(0 if result else 1)