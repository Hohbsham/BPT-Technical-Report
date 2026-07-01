# mesh_sanity_check.py
import os
import sys
import argparse
import numpy as np

class MeshSanityChecker:
    def __init__(self, mesh_path, output_dir):
        self.mesh_path = mesh_path
        self.output_dir = output_dir
        self.results = []
    
    def check_file_exists(self):
        """检查文件是否存在"""
        exists = os.path.exists(self.mesh_path)
        self.results.append({
            "name": "文件存在性检查",
            "passed": exists,
            "message": "文件存在" if exists else f"文件不存在: {self.mesh_path}"
        })
        return exists
    
    def check_file_size(self, min_size_kb=10):
        """检查文件大小是否合理 (>10KB)"""
        if not os.path.exists(self.mesh_path):
            return False
        size_kb = os.path.getsize(self.mesh_path) / 1024
        passed = size_kb > min_size_kb
        self.results.append({
            "name": "文件大小检查",
            "passed": passed,
            "message": f"文件大小: {size_kb:.2f} KB"
        })
        return passed
    
    def check_quad_ratio(self, min_ratio=0.8):
        """检查四边形比率 (适用于OBJ格式)"""
        if not os.path.exists(self.mesh_path):
            return False
        try:
            with open(self.mesh_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 统计不同类型的面
            faces = [l for l in lines if l.startswith('f ')]
            quad_faces = []
            tri_faces = []
            ngon_faces = []
            
            for f_line in faces:
                vertices = f_line.split()[1:]  # 跳过'f'
                vertex_count = 0
                for v in vertices:
                    # 处理 v/vt/vn 格式
                    vertex_idx = v.split('/')[0]
                    if vertex_idx:  # 确保不是空字符串
                        vertex_count += 1
                
                if vertex_count == 4:
                    quad_faces.append(f_line)
                elif vertex_count == 3:
                    tri_faces.append(f_line)
                else:
                    ngon_faces.append(f_line)
            
            if not faces:
                self.results.append({
                    "name": "四边形比率检查",
                    "passed": False,
                    "message": "未找到任何面数据"
                })
                return False
            
            quad_ratio = len(quad_faces) / len(faces)
            passed = quad_ratio >= min_ratio
            
            self.results.append({
                "name": "四边形比率检查",
                "passed": passed,
                "message": f"四边形比率: {quad_ratio*100:.1f}% (要求≥{min_ratio*100:.0f}%), "
                          f"四边形: {len(quad_faces)}, 三角形: {len(tri_faces)}, N边形: {len(ngon_faces)}"
            })
            return passed
        except Exception as e:
            self.results.append({
                "name": "四边形比率检查",
                "passed": False,
                "message": f"检查失败: {str(e)}"
            })
            return False
    
    def check_topology(self):
        """检查网格拓扑质量"""
        if not os.path.exists(self.mesh_path):
            return False
        
        try:
            with open(self.mesh_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 提取顶点和面信息
            vertices = []
            faces = []
            
            for line in lines:
                if line.startswith('v '):
                    coords = line.split()[1:4]
                    if len(coords) == 3:
                        try:
                            vertices.append([float(c) for c in coords])
                        except ValueError:
                            continue
                elif line.startswith('f '):
                    face_vertices = []
                    for v in line.split()[1:]:
                        vertex_idx = v.split('/')[0]
                        if vertex_idx:
                            try:
                                idx = int(vertex_idx)
                                if idx > 0 and idx <= len(vertices):
                                    face_vertices.append(idx)
                            except ValueError:
                                continue
                    if len(face_vertices) >= 3:
                        faces.append(face_vertices)
            
            if not vertices or not faces:
                self.results.append({
                    "name": "拓扑完整性检查",
                    "passed": False,
                    "message": "顶点或面数据缺失"
                })
                return False
            
            # 检查顶点数量和面数量的合理性
            vertex_count = len(vertices)
            face_count = len(faces)
            ratio = face_count / vertex_count if vertex_count > 0 else 0
            
            # 对于服装网格，合理的面/顶点比通常在1-3之间
            reasonable_ratio = 0.5 <= ratio <= 5.0
            
            self.results.append({
                "name": "拓扑完整性检查",
                "passed": reasonable_ratio,
                "message": f"顶点: {vertex_count}, 面: {face_count}, 比率: {ratio:.2f} "
                          f"({'合理' if reasonable_ratio else '不合理'})"
            })
            
            return reasonable_ratio
            
        except Exception as e:
            self.results.append({
                "name": "拓扑完整性检查",
                "passed": False,
                "message": f"检查失败: {str(e)}"
            })
            return False
    
    def run_all(self):
        """执行所有检查"""
        print(f"🔍 正在检查: {self.mesh_path}")
        self.check_file_exists()
        
        if os.path.exists(self.mesh_path):
            self.check_file_size()
            if self.mesh_path.endswith('.obj'):
                self.check_quad_ratio()
                self.check_topology()
        
        # 输出检查报告
        passed_count = sum(1 for r in self.results if r['passed'])
        total_count = len(self.results)
        
        print(f"\n📊 检查报告 ({passed_count}/{total_count} 项通过):")
        for r in self.results:
            status = "✅" if r['passed'] else "❌"
            print(f" {status} {r['name']}: {r['message']}")
        
        return passed_count == total_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Mesh文件路径")
    parser.add_argument("--output", default="./output", help="输出目录")
    args = parser.parse_args()
    
    checker = MeshSanityChecker(args.mesh, args.output)
    success = checker.run_all()
    sys.exit(0 if success else 1)