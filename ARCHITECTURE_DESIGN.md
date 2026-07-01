# Garment-Adaptive Mesh Generation Architecture

> 不只是 BPT 换数据——针对服装领域的特有几何和语义特征，重新设计生成框架。

## 1. 服装 Mesh 的独有特征（对照通用 3D 物体）

| 特征 | 通用 3D | 服装 Mesh |
|------|---------|----------|
| 拓扑类型 | 闭合 2-manifold | **开放 2-manifold with boundary** |
| 厚度 | 实心体积 | **零厚度薄壳** |
| 对称性 | 无 | **强双侧对称**（左袖=右袖） |
| 语义部件 | 无 | **领/袖/扣/缝**（清晰语义） |
| 变形模式 | 刚性 | **等距变形**（布料不可拉伸） |
| 拓扑约束 | 任意 | **袖口/领口必须开口** |
| 四边形 | 可选 | **必须**（用于物理模拟） |
| 面数范围 | 任意 | **受控**（~2500 大件/~1000 小件） |

## 2. 架构改进（在 BPT 基础上）

### 2.1 输入增强

```
当前: 点云(2048×6) → Michelangelo → 特征(257×1024)

改进:
  点云(2048×6) → Michelangelo → 特征(257×1024)
  类别 Token    → Embedding    → cls_emb(1×1024)    ← 新增
  关键点(1365×3)→ KP Encoder   → kp_feat(32×256)    ← 新增  
  边界信息      → Boundary Emb → bdr_feat(1×256)     ← 新增
                         ↓
                    Cross-Attention Fusion
                         ↓
              增强条件特征(257×1024)
```

**数据来源**：Other_clothes 自带 `kp_*.pcd`（关键点）和 `border.obj`（缝线边界），当前训练完全没用到。

### 2.2 生成约束

| 约束 | 实现方式 |
|------|---------|
| **对称性** | 生成时强制左右 token 序列镜像一致；或者在 loss 中加 symmetry loss |
| **开口约束** | 袖口/领口的边界顶点不能被面片闭合；加 boundary preservation loss |
| **四边形** | 修改 BPT_deserialize 的 quad 路径（同事已加，需调试） |
| **面数控制** | 生成时 max_seq_len 精确控制生成 token 数 → 面数 |
| **薄壳先验** | Position embedding 加入 geodesic distance bias（参考 Neural Garment Dynamics） |

### 2.3 序列结构改进

BPT 原始 token 序列：`[BOS] [face_tokens...] [EOS]`

服装改进版：
```
[BOS] [CATEGORY] [KEYPOINT_TOKENS] [mesh_tokens...] [BOUNDARY_TOKENS] [EOS]
   ↑       ↑            ↑               ↑                 ↑
  起始   服装类别     关键点位置     网格面片序列        缝线边界
         (Tops/       (领口/袖口/      (BPT 原有)        (border.obj
         Dress/       肩部/下摆)                        编码)
         Skirt...)
```

**优势**：
- 类别在前：控制整体拓扑（如 Dress 必须有裙摆区域，Top 必须有领口）
- 关键点在中：给面片生成提供空间锚点
- 边界在后：确保最后的面片恰好闭合到缝线处

### 2.4 损失函数增强

```
L_total = L_CE (cross-entropy on tokens, BPT 原有)
        + λ1 * L_sym (symmetry loss: 左半 token 序列 ≈ 右半 token 序列)
        + λ2 * L_bdr (boundary preservation: 开口边不能被填充)
        + λ3 * L_quad (quad encouragement: 非三角形 token pattern)
        + λ4 * L_geo (geodesic consistency: 相邻 token 的 3D 距离约束)
```

## 3. 数据扩充计划

### 3.1 现有
- ClothesNetM: 3,051 样本（主要是 Clo3D 生成的基础款）
- Other_clothes: 1,365 样本（含丰富语义标注，但面数不可控）

### 3.2 可获取的公开数据集

| 数据集 | 规模 | 特点 | 获取 |
|--------|------|------|------|
| **GarmageSet** (Style3D) | ~10K | 2D-3D 配对，专业设计 | HuggingFace |
| **Cloth3D** | ~8K | 物理模拟，动态序列 | 官网下载 |
| **Deep Fashion 3D** | ~2K | 真实服装扫描 | GitHub |
| **GarmentCodeData** | ~1K | 带缝纫图案 | GitHub |
| **SIZER** | ~500 | 多尺码人体 | 官网 |
| **BCNet** | ~500 | 边界标注 | GitHub |

### 3.3 合成数据生成
- 用 Marvelous Designer / Clo3D 批量导出（已有 ClothesNetM 的经验）
- 混元 3D 生成 → 重拓扑 → 加入训练
- Taobao 3D 服装模型（需购买，单价 ¥5-50/个）

## 4. 实施路线

### Phase 1: 验证（本周）
- [ ] 实现 Category Token 条件输入
- [ ] 用现有数据跑 Triangle 基线
- [ ] H200 或 4×A100，max_len=18000

### Phase 2: 约束（下周）
- [ ] Symmetry loss 实现
- [ ] 调试 quad deserialization

### Phase 3: 语义（第 3 周）
- [ ] Keypoint conditioning
- [ ] Boundary-aware generation

### Phase 4: 数据扩充
- [ ] 下载 GarmageSet + Cloth3D
- [ ] 重新预处理 pipeline

## 5. 与 BPT 原始框架的差异总结

| 维度 | BPT 原始 | 本方案 |
|------|---------|--------|
| 输入 | 仅点云特征 | 点云 + 类别 + 关键点 + 边界 |
| Token 结构 | 纯网格 Token | 层级结构（类别→KP→面片→边界） |
| Loss | Cross-entropy only | CE + 对称 + 边界 + 四边形 + 测地 |
| 输出 | 三角/四边面 mesh | 强制四边面 + 开口保留 |
| 数据 | 通用 3D 形状 | 服装专用 + 语义标注 |
| 领域先验 | 无 | 对称/薄壳/开口/等距变形 |
