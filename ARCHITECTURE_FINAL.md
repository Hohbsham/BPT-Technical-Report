# BPT 服装适配架构 — 完整技术方案

## 设计原则

> **能变成内容的 → Token；应该是约束的 → Loss**

| 信息类型 | 接入方式 | 理由 |
|---------|---------|------|
| Category（分类） | Token (Learned Embedding) | 离散类别，适合 Embedding |
| Keypoints（关键点） | Token (PointNet 编码) | 连续坐标需编码，小型 PointNet |
| Boundary（开口边界） | Loss + Attention Bias | 拓扑约束，不应变成 token |
| Symmetry（对称） | Loss 约束 | 几何先验，不是数据 |

---

## 1. Category Token

**状态**: ✅ v8 已实现并验证

**原理**: 11 类服装标签（Dress/Tops/Skirt/...）编码为 learned embedding，拼到 Michelangelo 特征前方。

```python
cat_emb = nn.Embedding(12, 1024)
category_token = cat_emb(cat_id)  # [B, 1024]
cond_embeds = torch.cat([category_token.unsqueeze(1), michelangelo_features], dim=1)
# → [B, 258, 1024] 代替原来的 [B, 257, 1024]
```

**实验效果**: H20(v6,无Category): epoch1 val=7.79 → A100(v8,有Category): epoch1 val=2.26。起步 loss 降低 71%。

---

## 2. Keypoints Token（📋 下周 P0）

**数据来源**: Other_clothes 自带的 `kp_*.pcd`（每个样本 1365 个 3D 关键点）

**设计方案**:
```python
# 关键点编码器：轻量 PointNet
class KPEncoder(nn.Module):
    def __init__(self, out_dim=256, num_kp=32):
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        self.pool = nn.AdaptiveMaxPool1d(num_kp)  # 下采样到固定数量
    
    def forward(self, kp_coords):
        # kp_coords: [B, N, 3]
        feats = self.mlp(kp_coords)  # [B, N, out_dim]
        feats = self.pool(feats.transpose(1,2))  # [B, out_dim, 32]
        return feats.transpose(1,2)  # [B, 32, out_dim] → 32个token
```

**接入序列**: `[CAT][KP₀][KP₁]...[KP₃₁][BOS][mesh tokens...]`

**依赖**: `kp_*.pcd` 文件（Other_clothes 已有 1365 个样本自带）

---

## 3. Boundary Loss（📋 下周 P1）

**数据来源**: Other_clothes 自带的 `border.obj`（缝线/边界 mesh）

| Loss | 复杂度 | 效果 |
|------|--------|------|
| **L_boundary** | ⭐ 简单 | 直接改善领口/袖口——防止开口被面片填上 |
| L_thin | ⭐⭐ 中等 | 防止厚度伪影（两层衣服） |
| L_geo | ⭐⭐ 中等 | 减少拓扑异常，BPT token 顺序约束 |
| TopoPE | ⭐⭐⭐ 复杂 | 长期提升——拓扑感知位置编码 |

**L_boundary 实现**:
```python
def boundary_preservation_loss(mesh, boundary_vertex_ids):
    """
    如果生成的面片三个顶点都在边界上 → 开口被填上了 → 高惩罚。
    """
    faces = mesh.faces
    boundary_set = set(boundary_vertex_ids)
    loss = 0
    for f in faces:
        b_count = sum(1 for v in f if v in boundary_set)
        loss += b_count / len(f)  # 0=正常, 1=全边界(异常)
    return loss / len(faces)

# L_total = L_CE + 0.1 * L_boundary
```

**依赖**: `border.obj` 文件（Other_clothes 已有 1365 个样本自带）

---

## 4. Thin-Shell Loss

```python
def thin_shell_loss(mesh):
    """相邻面的法向量夹角不能剧烈变化——防止双层面伪影"""
    adj = mesh.face_adjacency
    normals = mesh.face_normals
    loss = sum((1 - torch.dot(normals[f1], normals[f2])) for f1,f2 in adj)
    return loss / len(adj)

# L_total = L_CE + 0.1 * L_boundary + 0.1 * L_thin
```

**依赖**: 面片邻接关系（trimesh 自带计算）

---

## 5. Geodesic Coherence Loss

```python
def geodesic_coherence_loss(tokens_3d_coords, max_dist=2.0):
    """BPT序列中相邻token在3D空间中也应该相邻"""
    loss = 0
    for i in range(len(tokens_3d_coords) - 1):
        dist = torch.norm(tokens_3d_coords[i] - tokens_3d_coords[i+1])
        loss += max(0, dist - max_dist)
    return loss
```

---

## 6. Symmetry Loss（📋 下下周）

```python
def symmetry_loss(mesh):
    """服装左右对称：左半token序列 ≈ 右半token序列"""
    # 沿XZ平面翻转mesh，计算Chamfer距离
    flipped_verts = mesh.vertices.clone()
    flipped_verts[:, 0] *= -1  # 翻转X轴
    # Chamfer distance between original and flipped
    return chamfer_distance(mesh.vertices, flipped_verts)
```

---

## 7. TopoPE: Topology-Aware Position Encoding

```python
class TopologyPositionEncoding(nn.Module):
    """
    除了序列位置，还编码边界距离和边界标记。
    让模型"知道"领口一圈的顶点是连在一起的。
    """
    def __init__(self, dim):
        self.boundary_emb = nn.Embedding(2, dim)     # 是/否边界顶点
        self.dist_emb = nn.Linear(1, dim)             # 到最近边界的测地距离
    
    def forward(self, pos_emb, boundary_dist, is_boundary):
        return pos_emb + self.boundary_emb(is_boundary) + self.dist_emb(boundary_dist)
```

---

## 实施路线图

| 优先级 | 任务 | 依赖 | 复杂度 | 周期 | 收益 |
|--------|------|------|--------|------|------|
| ✅ 已完成 | Category Token | 类别映射文件 | ⭐ | 完成 | v8验证：loss -71% |
| P0 | Keypoints Token | `kp_*.pcd` | ⭐⭐ | 1周 | 空间锚点，提升局部精度 |
| P0 | L_boundary | `border.obj` | ⭐ | 1周 | 领口/袖口开口质量 |
| P1 | L_thin | trimesh 邻接 | ⭐⭐ | 1周 | 防止厚度伪影 |
| P1 | L_geo | token→3D映射 | ⭐⭐ | 1周 | 减少拓扑异常 |
| P2 | Symmetry Loss | 无额外依赖 | ⭐ | 1周 | 全局结构一致性 |
| P3 | TopoPE | 需要改model init | ⭐⭐⭐ | 2周 | 长期架构提升 |

---

## 总 Loss 公式

```
L_total = L_CE (BPT 原有 token 预测)
        + 0.1 × L_boundary (开口约束)
        + 0.1 × L_thin    (薄壳约束)
        + 0.05 × L_geo    (测地约束)
        + 0.05 × L_sym    (对称约束)
```

---

## Token 序列结构（完整版）

```
[BOS]
[CAT]            ← 服装类别 (11类+unknown)
[KP₀]...[KP₃₁]   ← 关键点 (32个PointNet编码token)
[mesh tokens...] ← BPT 面片序列（原有）
[EOS]

Cross-Attention 条件:
├── Michelangelo 点云特征 (257×1024) ← 原有
├── Category Embedding (1×1024)     ← ✅ v8
├── Keypoint Features (32×256)      ← 📋 P0
└── Boundary Attention Bias         ← 📋 P0
```
