# 3D Garment Point Cloud Regularizer — 项目追踪

> 最后更新: 2026-05-07

---

## 核心目标

训练一个 **human cloth point cloud regularizer**，输入为破损/噪声的部分点云（模拟从3D mesh上"抠下来"），输出为规整的四边形mesh。使用BPT架构。

## 技术路线

```
Image → Hunyuan3D/SAM3D → 重建3D mesh → 分割出单层服装mesh
                                              ↓
                               Quad Remeshing → 监督数据 GT (clean quad mesh)
                                              ↓
clean mesh → 人为破坏(加噪/抠洞/去边) → partial point cloud (INPUT)
quad remeshed mesh                    → clean quad mesh (GROUND TRUTH)
                                              ↓
                                   训练 BPT Regularizer
                                              ↓
                              Demo: 单图重建 / 点云重建 / 物理模拟
```

## 数据Pipeline

### Step 1: 数据来源
- **ClothesNetM** (`d:/ClothesNetData/ClothesNetM/`): 11类, 每类几十种款式
  - Dress, Glove, Hat, Mask, One-piece, Scarf_Tie, Skirt, Socks, Tops, Trousers, UnderPants
- **其他数据集**: VRoid, GarmentNet (待收集)

### Step 2: Quad Remeshing → 监督数据GT

#### 方案A: Exoside QuadRemesher (高质量)
- **插件**: `quad_remesher_1_4` (已安装)
- **参数**: Target Face Count=8000, Adaptive Size=0, Adaptive Quad Count=OFF, Detect Hard Edges=OFF, Normals Splitting=ON
- **操作**: 在Blender GUI中选择物体 → 右侧面板BlenderMCP旁找到QuadRemesher → 点击"REMESH IT"
- **局限**: 必须在Blender GUI中操作（Modal operator），无法完全后台自动化
- **底层API**: `bpy.ops.qremesher.remesh('INVOKE_DEFAULT')` → 启动modal
  - 内部函数: `qr_operators.export_selected_mesh_fbx()` → `qr_exe.exe_launchRemeshing()` → `qr_operators.import_mesh_fbx()`
  - 手动调用需要构造完整的operator context

#### 方案B: QuadriFlow (全自动，内置)
- **API**: `bpy.ops.object.quadriflow_remesh(target_faces=8000)`
- **已测试**: DLG_Dress032_1 → 8869 verts, 8652 faces, **100% quads**
- **优点**: 完全可脚本化，适合批量处理
- **建议**: 先用QuadriFlow快速生成数据跑通流程，最终用QuadRemesher提升质量

### Step 3: 制造破损输入
- clean mesh → 顶点噪声 + 随机孔洞 + 边缘修剪 → damaged mesh → point cloud
- 模拟从3D mesh上"抠下来"的感觉

### Step 4: BPT Regularizer 训练
- 输入: damaged point cloud
- 输出: regularized quad mesh
- 架构: BPT (Blocked and Patchified Tokenization)

### Step 5: Demo
- 单图重建 / 点云重建 / 物理模拟

---

## 环境配置

| 组件 | 状态 | 位置 |
|------|------|------|
| Blender 4.2.1 LTS | ✅ | `C:\Users\YANGZ\blender\blender-4.2.1-windows-x64\blender.exe` |
| QuadRemesher 1.4 | ✅ | 插件 `quad_remesher_1_4`, 操作符 `bpy.ops.qremesher.remesh` |
| QuadriFlow (内置) | ✅ | `bpy.ops.object.quadriflow_remesh()` — 已验证可用, 100% quads |
| BlenderMCP | ✅ | 端口 9876, `claude mcp add blender uvx blender-mcp` |
| Python | ✅ | Python 3.11 + 3.13, uv 0.11.8 |
| 测试模型 | ✅ | DLG_Dress032_1 → 已在Blender中处理 (Garment_Quad_GT, 8652 quads) |

---

## 关键脚本

| 脚本 | 用途 |
|------|------|
| `project_docs/blender_garment_preprocessing_advanced.py` | Solidify + 锐边 + 法线 |
| `project_docs/run_garment_preprocessing.py` | 命令行批量预处理 |
| `project_docs/bpt_quad_converter.py` | BPT格式转换 |
| `project_docs/garment_bpt_pipeline.py` | 完整pipeline |
| `supervised_data_pipeline.py` | **(新建)** 批量生成监督数据 (QuadRemesh + 破损) |

---

## QuadRemesher 使用指南

### GUI操作 (每个模型需要手动操作)
1. Blender里选择服装mesh
2. 在右侧属性面板找到 **QuadRemesher** 标签（一般在修改器面板附近）
3. 设置参数:
   - Target Face Count: **8000**
   - Adaptive Size: **0**
   - Adaptive Quad Count: **OFF**
   - Detect Hard Edges: **OFF**
   - Use Normals Splitting: **ON**
4. 点击 **"REMESH IT"**
5. 等待完成（几秒到几十秒）
6. 导出处理后的mesh为OBJ

### 为什么不能全自动?
QuadRemesher使用 `RUNNING_MODAL` 操作符模式，需要Blender GUI事件循环。底层引擎是独立的 `.exe`，可以通过FBX中转调用，但需要构造完整的operator context。

---

## 快速生成监督数据流程

### 方案: QuadriFlow批量处理 (全自动)

```bash
# 使用Blender后台批量处理
blender --background --python supervised_data_pipeline.py -- \
    --data_dir D:/ClothesNetData/ClothesNetM \
    --output_dir D:/ClothesNetData/supervised_data \
    --target_faces 8000 \
    --category Dress
```

产出结构:
```
supervised_data/
├── Dress/
│   ├── DLG_Dress032_1/
│   │   ├── ground_truth.obj      # QuadriFlow处理后的clean quad mesh
│   │   ├── input_damaged.obj     # 破损版本 (加噪+孔洞)
│   │   └── input_pointcloud.ply  # 输入点云
│   └── ...
```

### 破损参数
- **噪声**: 高斯噪声 σ=0.5%~2% of bounding box
- **孔洞**: 随机删除5%~20% 的面
- **边界**: 随机侵蚀5%~15% 的边界边
- 每种破损提供多档强度 (light/medium/heavy)

---

## 进度

- [x] 环境搭建 (Blender + QuadRemesher + BlenderMCP)
- [x] 单模型流程验证 (DLG_Dress032_1 — Import → Solidify → QuadriFlow → 100% quads)
- [x] QuadriFlow验证 (8652 quads, 可脚本化批量)
- [ ] 批量生成监督数据 (QuadriFlow + 破损)
- [ ] QuadRemesher批量处理 (GUI手动或半自动)
- [ ] BPT Regularizer 训练
- [ ] Demo

## 参考论文

| Paper | Link |
|-------|------|
| AIpparel | https://arxiv.org/pdf/2412.03937 |
| GarmentCode | https://arxiv.org/abs/2306.03642 |
| SewFormer | https://sewformer.github.io/ |
| Panelformer | https://openaccess.thecvf.com/content/WACV2024/papers/Chen_Panelformer_Sewing_Pattern_Reconstruction_From_2D_Garment_Images_WACV_2024_paper.pdf |
| ChatGarment | https://chatgarment.github.io/ |
| DressCode | https://ihe-kaii.github.io/DressCode/ |
| ClothesNet | Dataset |
