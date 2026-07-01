# 3D服装建模项目

## 项目概述
本项目旨在利用大模型生成服装mesh，然后通过QuadRemesher进行网格修复和规范化处理，最终转换为四边形面片的BPT格式。

## 目录结构
```
D:\ClothesNetData\                    # 项目主目录
├── original_data\                    # 原始数据（如果存在）
├── processed_data\                   # 处理后的数据
├── project_docs\                     # 项目文档和脚本
│   ├── 3D_Clothing_Modeling_README.md  # 项目详细说明
│   ├── blender_process_clothes.py      # 基础Blender处理脚本
│   └── blender_process_clothes_advanced.py # 高级Blender处理脚本
└── logs\                             # 处理日志
```

## 项目文件
- **项目文档**: `D:\ClothesNetData\project_docs\3D_Clothing_Modeling_README.md`
- **基础处理脚本**: `D:\ClothesNetData\project_docs\blender_process_clothes.py`
- **高级处理脚本**: `D:\ClothesNetData\project_docs\blender_process_clothes_advanced.py`

## 核心工作流程
1. 大模型生成初始mesh
2. QuadRemesher网格修复
3. Blender自动化脚本处理
4. BPT格式转换

## 运行脚本
在命令行中执行：
```bash
"C:\Program Files\Blender Foundation\Blender 3.3\blender.exe" --background --python "D:\ClothesNetData\project_docs\blender_process_clothes_advanced.py"
```

## 需要处理的文件
- C:\Users\YANGZ\Desktop\remesh.pdf - 相关文档
- C:\Users\YANGZ\Desktop\Garment预测.docx - 项目文档
- D:\ClothesNetData - 元数据目录