# BPT 服装重建 — 下阶段规划

## 优先级排序

### P0: Triangle Mesh 基线（本周）

**目标**：用三角面训练 BPT，跑通完整 pipeline，建立性能基线。

- 数据：现有 supervised_data，mesh_type=triangle（QuadriFlow 已转三角面）
- 代码：v6 直接可用，改 `--mesh_type triangle`
- 平台：H200（141GB）或 4×A100（DeepSpeed 多卡）
- 关键参数：`--max_code_len 18000` + position embedding 插值
- 成功标准：Chamfer < 10%，生成完整 mesh（> 500 顶点）

### P1: 数据修复（本周并行）

- 修复 ~340 个袖口/领口闭合异常的样本
- 工具：Blender + QuadriFlow 重处理
- 或暂时排除，用 2807-340 ≈ 2467 个干净样本训练

### P2: 引入语义信息（下周）

Other_clothes 每个样本自带：

| 数据 | 怎么用 |
|------|--------|
| `border.obj` | 缝线/边界 → 加到 loss 约束，强化边缘生成 |
| `kp_*.pcd` | 关键点 → Cross Attention 条件输入 |
| 目录分类（Collar/Sleeve/...） | 前置 token 区分服装类型 |

实现方式：
```python
# 在 token 序列最前面加一个类别 token
category_token = Embedding(num_categories, dim)
input = [category_token, ...mesh_tokens]
```

### P3: 3D 服装分割（下下周）

- 评估开源方案（当前"功能符合但质量一般"）
- 或：用混元生成穿衣人体 → 渲染 → 训练 2D 分割 → 映射到 3D
- 最终交付：从穿衣人体 mesh 自动分离出单层服装

## 分工

| 人 | 任务 |
|----|------|
| 杨子旭 | P0: Triangle BPT 训练 + 语义实验 |
| 杨子旭 | P1: 数据修复 |
| 林铖 | P3: 分割方案调研 + 数据 |
| 卢思远 | P3: 分割自动化方案 |

## 时间线

```
第1周: P0 Triangle 基线 + P1 数据修复
第2周: P2 语义信息引入（keypoint/boundary/category）
第3周: P3 分割方案评估 + 自建数据初步
第4周: 全流程集成 + 组会演示
```

## 需要的资源

- GPU: H200 141GB 或 4×A100 40GB（约 ¥15-50/h）
- 时间: 每轮训练 8-15 小时
- 数据: 现有 2807 样本即可开始，后续修复 340 个问题样本
