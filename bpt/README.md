# BPT Fine-Tuning for Clothing Mesh Reconstruction

基于 BPT (Blocked and Patchified Tokenization) 的服装网格重建微调项目。

## 项目目标

从带噪声的深度扫描点云（2048点）重建干净的服装四边面网格。

```
噪声点云(2048×6) → Michelangelo编码器[冻结] → 条件特征(257×1024)
                                                       ↓ 交叉注意力
BOS → MeshTransformer(24层/1024维/711M参数) → 逐token预测 → BPT解码 → 干净mesh
```

## 数据

### 数据集构成

| 来源 | 原始样本 | 处理后四边面 | 排除(三角面) | 说明 |
|------|---------|-------------|-------------|------|
| ClothesNetM | 3051 | 2831 | 220 | 原始服装数据集，经 QuadriFlow 重网格化 |
| Other_clothes | 1365 | 356 | 1009 | 补充服装数据集 |
| **合计** | **4416** | **3187** | **1229** | — |

### 面数标准化

所有样本按类型分为两类，通过 QuadriFlow 统一面数：

| 类型 | 目标面数 | 范围 | 包含类别 |
|------|---------|------|---------|
| 大件 | 2500 四边面 | 1800–3200 | Dress, Tops, Jackets, Pants, Skirts, One-piece, Underwear |
| 小件 | 1000 四边面 | 700–1300 | Gloves, Hats, Masks, Socks, Scarves/Ties |

> 排除的 1229 个样本为 QuadriFlow 无法生成四边面的顽固三角面网格（三轮重试后仍为三角面），位于 `supervised_data_excluded/` 和 `supervised_data_other_excluded/`。

### 数据目录

| 目录 | 说明 | 大小 |
|------|------|------|
| `supervised_data/` | 3187个样本，每个包含 `ground_truth.obj`(干净四边面mesh) + `input_pointcloud.ply`(噪声点云2048点) + `input_damaged.obj` | ~1.8GB |
| `cached_embeddings/` | 预计算的Michelangelo编码器输出，3186个 `.pt` 文件 | ~1.5GB |
| `weights/bpt-8-16-500m.pt` | BPT预训练权重(500M参数，MeshGPT格式) | 1.5GB |
| `ClothesNetM/` | 原始源数据 | 13.3GB |
| `Other_clothes/` | 补充原始源数据 | 7.4GB |

### 每样本格式

```
supervised_data/{sample_name}/
├── ground_truth.obj      # QuadriFlow 四边面 mesh（训练目标）
├── input_pointcloud.ply  # 噪声点云，2048 XYZ 点（模型输入）
└── input_damaged.obj     # 边界噪声 mesh（中间产物）
```

## 数据预处理流程

```
原始 mesh (ClothesNetM / Other_clothes)
    │
    ├── [1] QuadriFlow remesh (target=2500 大件 / 1000 小件)
    │       → 生成规整四边面
    ├── [2] 面数标准化 (standardize_faces.py)
    │       → 两轮 QuadriFlow 调整至目标范围
    ├── [3] 排除三角面 mesh
    │       → 三轮重试后仍为三角面的移入 excluded 目录
    ├── [4] 点云生成 (process_one_model.py)
    │       → 面积加权采样 2048 点 + 高斯噪声
    └── [5] Michelangelo 嵌入缓存 (precompute_embeddings.py)
            → 预计算编码器输出，加速训练 ~10x
```

### 相关脚本

| 脚本 | 用途 |
|------|------|
| `batch_supervised_pipeline.py` | 批量处理原始 mesh → supervised_data |
| `process_one_model.py` | Blender 后台单样本处理（QuadriFlow + 点云） |
| `remesh_to_target.py` | Blender 后台单样本重网格化 |
| `standardize_faces.py` | 批量面数标准化 |
| `precompute_embeddings.py` | 预计算 Michelangelo 嵌入缓存 |

## 环境要求

### 硬件

| GPU | 显存 | batch_size | max_code_len |
|-----|------|------------|-------------|
| RTX 4090 | 24GB | 2 | 4000 |
| A40 | 48GB | 4 | 6000 |
| A100 40GB | 40GB | 4 | 6000-8000 |
| A100 80GB | 80GB | 8 | 10000 |

**最低要求**: 24GB 显存 GPU，推荐 RTX 4090 或以上。

### 软件

- **Python**: 3.10+
- **PyTorch**: >= 2.0.0 (推荐 2.1+)
- **CUDA**: 11.8 或 12.1
- **Blender**: 4.2+ (仅数据预处理需要，含 QuadriFlow)
- **依赖**: 见 `requirements_cloud.txt`

## 快速开始

### 1. 安装依赖

```bash
# 先安装 PyTorch（根据 CUDA 版本选择）
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118

# 再安装其他依赖
pip install -r requirements_cloud.txt
```

### 2. Overfit Test（训练前必做）

验证pipeline正确性，用少量样本确认loss能降到<0.1：

```bash
python overfit_test.py \
    --n_samples 3 \
    --epochs 50 \
    --max_code_len 500 \
    --lr 5e-5 \
    --cache_dir cached_embeddings \
    --data_dir ../supervised_data
```

**判断标准**：

| 最终loss | 结论 |
|----------|------|
| < 0.3 | ✅ 完美 |
| < 0.5 | ✅ 通过，可以正式训练 |
| 0.5~1.0 | ⚠️ 降低 lr 或多跑几个 epoch |
| > 1.0 或 NaN | ❌ 检查数据/嵌入是否有问题 |

> **注意**：多样本 overfit 可能需要调低学习率（5e-5）以避免 NaN。1 样本测试用 1e-4 通常没问题。overfit_test.py 使用随机采样（seed=42）。

### 3. 正式训练

```bash
python train_cloud.py \
    --data_dir ../supervised_data \
    --cache_dir cached_embeddings \
    --model_path weights/bpt-8-16-500m.pt \
    --config config/BPT-cloud-4k-8-16.yaml \
    --output_dir checkpoints \
    --epochs 50 \
    --batch_size 2 \
    --max_code_len 4000 \
    --grad_accum_steps 4 \
    --lr 1e-4
```

**训练参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 总训练轮数 |
| `--batch_size` | 2 | 每GPU batch大小 |
| `--max_code_len` | 4000 | token序列最大长度（超出截断） |
| `--grad_accum_steps` | 4 | 梯度累积步数（有效batch = batch_size × grad_accum） |
| `--lr` | 1e-4 | 学习率（AdamW），如遇 NaN 降至 5e-5 |
| `--weight_decay` | 0.01 | 权重衰减 |
| `--grad_clip` | 1.0 | 梯度裁剪阈值 |
| `--save_every` | 5 | 每N epoch保存checkpoint |
| `--no_cache` | false | 禁用缓存嵌入（编码器在训练循环中运行，慢10倍） |
| `--augment` | true | 数据增强（点云Z轴旋转） |
| `--freeze_encoder` | true | 冻结Michelangelo编码器 |
| `--mesh_type` | auto | 输出面类型：triangle / quad / auto |

### 4. 推理

```bash
python main.py \
    --config config/BPT-cloud-4k-8-16.yaml \
    --model_path checkpoints/best_model.pt \
    --input_pointcloud input.ply \
    --mesh_type quad \
    --output output_mesh.obj
```

### 5. 监控

```bash
# GPU使用
watch -n 2 nvidia-smi

# 训练日志（如果用nohup后台运行）
tail -f train.log
```

### 6. 输出

- `checkpoints/best_model.pt` — 验证集loss最低的模型
- `checkpoints/checkpoint_epoch{N}.pt` — 周期检查点
- `checkpoints/final_model.pt` — 最终模型

## 训练预估（RTX 4090 24GB）

| 指标 | 值 |
|------|-----|
| 速度 | ~1.2-1.5 s/it |
| 每epoch | ~0.5小时 (batch_size=2, 3187样本) |
| 50 epoch | ~25小时 |
| 峰值显存 | ~22GB |

## Token 格式

BPT使用 `block_size=8, offset_size=16, quant_bit=16` 进行离散化：

```
mesh → patchify(哈密顿环游走) → discretize(分块/偏移编码) → token序列
```

训练使用 **triangle模式**（trimesh自动将四边面三角化），推理可用 `--mesh_type quad` 输出四边面。

## 验证结果

**Mesh往返验证（2026-05-20）**：

| 模式 | 面覆盖率 | Chamfer(%bbox) | Hausdorff(%bbox) | 质量 |
|------|---------|----------------|-----------------|------|
| Triangle | 100% | 0.77% | 1.36% | EXCELLENT |
| Quad | 81-83% | 0.79-1.02% | 1.56-2.81% | EXCELLENT |

**Overfit Test（2026-05-29 最新）**：

| 设置 | 结果 | 详情 |
|------|------|------|
| 1样本 / lr=1e-4 / max_len=200 | ✅ PASS | loss 18.5→0.04, 56 epoch |
| 1样本 / lr=1e-4 / max_len=500 | ✅ PASS | loss 18.5→0.05, 39 epoch |
| 3样本 / lr=5e-5 / max_len=500 | ⚠️ 慢 | 50 epoch loss 降至 1.05 |

## 同事改动说明

同事在 `main.py` 和 `model/serializaiton.py` 中的改动（已在代码中）：
- `main.py`：新增 `--mesh_type` 参数（triangle/quad/auto），推理时选择输出面类型
- `serializaiton.py`：改进 quad 反序列化，调整度残差警告阈值（仅 ≥100 时报警），新增自动面类型检测
- `train.log`：确认全量训练 pipeline 可在同事机器上正常运行

## 目录结构

```
bpt/
├── train_cloud.py              # 云端训练主脚本
├── overfit_test.py             # 过拟合测试（训练前必跑，随机采样）
├── precompute_embeddings.py    # 预计算Michelangelo嵌入
├── validate_roundtrip.py       # mesh→tokens→mesh往返验证
├── main.py                     # 推理脚本
├── model/
│   ├── model.py                # MeshTransformer(711M参数)
│   ├── serializaiton.py        # BPT序列化/反序列化（含quad支持）
│   ├── data_utils.py           # 网格数据处理
│   └── miche_conditioner.py    # Michelangelo点云编码器
├── miche/                      # Michelangelo编码器权重
├── config/
│   └── BPT-cloud-4k-8-16.yaml  # 训练配置
├── weights/
│   └── bpt-8-16-500m.pt        # 预训练权重
├── cached_embeddings/          # 预计算嵌入缓存（3186个.pt文件）
├── requirements_cloud.txt      # 依赖清单
└── README.md                   # 本文件

../
├── supervised_data/            # 训练数据（3187样本）
├── supervised_data_excluded/   # 排除的三角面（220样本）
├── supervised_data_other_excluded/ # 排除的三角面（1009样本）
├── ClothesNetM/                # 原始源数据
├── Other_clothes/              # 补充原始源数据
├── batch_supervised_pipeline.py # 批量处理脚本
├── standardize_faces.py        # 面数标准化脚本
├── remesh_to_target.py         # 单样本重网格化脚本
├── process_one_model.py        # Blender 单样本处理
└── project_docs/               # 工程文档和旧版脚本
```

## 常见问题

**Q: OOM（显存不足）怎么办？**
A: 减小 `max_code_len`（4000→3000→2000）或减小 `batch_size`（2→1），同时增大 `grad_accum_steps`（4→8）保持有效batch不变。

**Q: loss不下降 / NaN？**
A: 先跑 `overfit_test.py` 确认pipeline。降低学习率到 5e-5，或减少 batch 内样本数。检查 `cached_embeddings/` 是否与 `supervised_data/` 匹配。

**Q: 加载权重报错 "No module named deepspeed"？**
A: 权重文件包含deepspeed引用。Linux上安装：`pip install deepspeed>=0.14.0`。Windows上需用patch（已嵌入 `overfit_test.py` 和 `precompute_embeddings.py`）。

**Q: AutoDL上怎么部署？**
A: 
1. 租RTX 4090，选PyTorch 2.1+镜像
2. 上传 `bpt/` 和 `supervised_data/` 到 `/root/autodl-tmp/`
3. `pip install -r requirements_cloud.txt`
4. 先跑overfit_test，通过后跑正式训练

**Q: 训练中断了怎么续？**
A: 从checkpoint加载继续训（需修改 `train_cloud.py` 加载checkpoint的逻辑）。

**Q: 哪些样本被排除了？**
A: 共排除 1229 个三角面样本：
- `supervised_data_excluded/`：220个（来自 ClothesNetM）
- `supervised_data_other_excluded/`：1009个（来自 Other_clothes）
这些样本 QuadriFlow 三轮都无法转为四边面，无法用于四边面 BPT 训练。

## 原始论文

BPT: Scaling Mesh Generation via Compressive Tokenization
- [Paper](https://arxiv.org/abs/2411.07025) | [Code](https://github.com/whaohan/bpt) | [Weight](https://huggingface.co/whaohan/bpt)

```bibtex
@article{weng2024scaling,
  title={Scaling Mesh Generation via Compressive Tokenization}, 
  author={Haohan Weng and Zibo Zhao and Biwen Lei and Xianghui Yang and Jian Liu and Zeqiang Lai and Zhuo Chen and Yuhong Liu and Jie Jiang and Chunchao Guo and Tong Zhang and Shenghua Gao and C. L. Philip Chen},
  journal={arXiv preprint arXiv:2411.07025},
  year={2024}
}
```
