# BPT 下周实施计划

## P0: Keypoints Token + Boundary Loss（下周一二）

### 1. Keypoints Token（kp_*.pcd 编码）

**文件**: `bpt/keypoint_encoder.py`
```python
class KPEncoder(nn.Module):
    """轻量 PointNet 编码关键点到 32 个 token"""
    def __init__(self, out_dim=256, num_kp=32):
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, out_dim))
        self.pool = nn.AdaptiveMaxPool1d(num_kp)
```

**数据集改动**: `SupervisedDatasetV8` 增加 `kp_path` 字段，从 `Other_clothes/{sample}/kp_*.pcd` 加载

**训练改动**: `train_cloud_v9.py` 把 kp_tokens 拼到 cond_embeds 后面

### 2. Boundary Preservation Loss

**文件**: `bpt/losses.py`
```python
def boundary_loss(mesh, boundary_verts):
    """开口边界不能被面片填上"""
    ...
def thin_shell_loss(mesh):
    """相邻面法向量一致"""
    ...
```

**训练改动**: `L_total = L_CE + 0.1 * L_boundary + 0.1 * L_thin`

---

## P1: GarmageSet 下载 + 融合（下周三）

1. 注册 HuggingFace → 获取 token
2. `huggingface_hub.snapshot_download('Style3D/GarmageSet')`
3. 提取 caption.csv → 映射到我们的 11 类
4. 预处理：OBJ → QuadriFlow → supervised_data 格式
5. 合并到现有训练集

---

## P2: Symmetry Loss（下周四）

```python
def symmetry_loss(mesh):
    flipped = mesh.vertices.copy()
    flipped[:, 0] *= -1
    return chamfer_distance(mesh.vertices, flipped)

L_total += 0.05 * L_sym
```

---

## P3: TopoPE（下周后期）

在 model init 时加 `boundary_emb` 和 `dist_emb`，需要改 model.py。

---

## 分工

| 任务 | Agent | 状态 |
|------|-------|------|
| 数据质量检查 | BPT_Data_check | ✅ 脚本已写好 |
| Keypoints 编码 | BPTtrain | 📋 下周 |
| Boundary Loss | BPTtrain | 📋 下周 |
| GarmageSet 下载 | BPT_Data_check | 📋 下周 |
| Symmetry Loss | BPTtrain | 📋 下下周 |
| TopoPE | BPTtrain | 📋 下下周 |
