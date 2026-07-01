# BPT 服装网格训练 — 周进展报告 (2026.06.03–06.09)

## 一、项目目标

训练一个 BPT (Blocked and Patchified Tokenization) 模型，从 2048 点噪声点云重建规整的四边形服装网格。

```
噪声点云(2048×6) → Michelangelo编码器[冻结] → 条件特征(257×1024)
                                                      ↓ 交叉注意力
BOS → MeshTransformer(24层/1024维/711M参数) → 逐token预测 → BPT解码 → 干净mesh
```

---

## 二、数据工程

### 2.1 数据源
| 数据集 | 原始样本 | 处理 |
|--------|---------|------|
| ClothesNetM | 3,051 | QuadriFlow 四边面化 |
| Other_clothes | 1,365 | QuadriFlow 四边面化 |
| **合计入训** | **2,807** | 面数标准化(大件2500/小件1000) |
| **独立测试集** | **311** | 完全不动,保持独立 |
| **排除** | 1,299 | 面数>5000 或 QuadriFlow 失败 |

### 2.2 数据处理链路

```
原始 OBJ → QuadriFlow remesh → 面数标准化(3轮) → 排除顽固三角面
                                                           ↓
  supervised_data/{sample}/
    ├── ground_truth.obj      # 干净四边面 GT
    ├── input_damaged.obj     # 破损版 (加噪+删面)
    └── input_pointcloud.ply  # 2048点噪声点云
                                                           ↓
  Michelangelo 编码器 → cached_embeddings/{sample}.pt  # 预计算 fp16→fp32
```

### 2.3 数据清理
- 移除 70 个脏样本：面数>5000 (QuadriFlow 失败)、顶点退化 (全零 mesh)
- 分离 311 个独立测试样本 (10%)，保证泛化评估公正

---

## 三、训练环境演进

### 3.1 平台历程

| 阶段 | 平台 | GPU | 显存 | max_len | 结果 |
|------|------|-----|------|---------|------|
| 1 | MUST 学校服务器 | V100 | 32GB | 4,000 | 50 epoch 完成，推理碎片(0% quad, Chamfer 22-51%) |
| 2 | AutoDL | RTX 4090 | 24GB | 2,000 | 训不动，NaN 崩溃 |
| 3 | **AutoDL** | **H20** | **96GB** | **10,000** | **fp32, batch=2, 稳定训练中** |

### 3.2 为什么换平台

V100 32GB 只能跑 max_code_len=4,000，但实际 mesh 的 token 中位数是 14,407。4,000 只覆盖 0% 的样本——模型从头到尾都在学碎片。H20 96GB 能跑 10,000，覆盖约 50% 样本，显著改善。

---

## 四、代码修复历程 (共约 20 次崩溃)

### 4.1 致命 Bug 及修复

| # | Bug | 症状 | 根因 | 修复 |
|---|-----|------|------|------|
| 1 | dtype 冲突 | `Half != float` crash | 嵌入 fp16，验证无 autocast | 嵌入全转 fp32 + 纯 fp32 训练 |
| 2 | UnboundLocalError | `loss` 未定义 | NaN 批次 `continue` 后 loss 未赋值 | `loss = None` 初始化在 if/else 前 |
| 3 | NaN 污染权重 | 推理全崩 | `loss.backward()` 在 NaN 检查前执行 → NaN 梯度更新参数 | `torch.isfinite(loss)` 拦截，跳过 backward |
| 4 | 梯度爆炸 | 首批 NaN | 初始 loss >18，fp16 溢出，无 warmup | lr warmup：半 epoch 从 0.01× 线性升至 1× |
| 5 | .pyc 覆盖新代码 | 修了 5 次都不生效 | Python 字节码缓存优先于源文件 | `python -B` + 每次部署前 `rm -rf __pycache__` |
| 6 | fp32 OOM | 94.6GB/96GB | batch=4 时 fp32 翻倍显存 | batch_size 4→2, grad_accum 2→4 |
| 7 | try/except 缩进错误 | `loss` 只在 else 分支定义 | 补丁把 try 块放进了 else 内 | 移到 if/else 外侧 |
| 8 | 双 Python 路径 | `null bytes` crash | 命令中误写 `{PY} {PY}` | 改为单个 `{PY}` |
| 9 | position embedding resize | 18000 → crash | 插值代码不完整 (sed 截断) | 放弃 resize，使用预训练原生的 10000 |
| 10 | scheduler `total_steps` | NameError | warmup 代码在 total_steps 定义前引用 | 移动 total_steps 计算到 optimizer 前 |

### 4.2 教训

> ⚠️ **绝对不要用 sed 在服务器上改代码。** 始终在本地写 → 编译验证 → 一次 SCP 上传 → `python -B` 启动。

---

## 五、AutoResearch Pipeline

### 5.1 架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                      AutoResearch Loop                                │
│                                                                      │
│   program.md (研究目标)                                               │
│        │                                                             │
│        ▼                                                             │
│   Claude 编辑 train_cloud.py (调超参)                                 │
│        │                                                             │
│        ▼                                                             │
│   SCP 上传 → AutoDL H20 → python -B train_cloud.py                   │
│        │                                                             │
│        ▼                                                             │
│   monitor.py 每30分钟 SSH 巡检 (VRAM/进程/NaN/错误)                   │
│        │                                                             │
│        ├── 崩溃 → 读日志 → 自动重启                                   │
│        ├── NaN>10 → 降lr重启                                         │
│        └── 完成 → 触发推理评估                                        │
│        │                                                             │
│        ▼                                                             │
│   infer_and_eval.py 对 311 测试集推理                                  │
│        │                                                             │
│        ▼                                                             │
│   Chamfer/Hausdorff/FaceCoverage 指标                                 │
│        │                                                             │
│        ├── Chamfer<2% + Coverage>80% → EXCELLENT → git tag + 精调    │
│        ├── Chamfer<5% → GOOD → 提高 max_len                          │
│        ├── Chamfer<10% → FAIR → 加 epoch                             │
│        └── Chamfer>10% → POOR → 人工介入                             │
│        │                                                             │
│        ▼                                                             │
│   Git commit + push → 下一轮循环                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 文件清单

| 文件 | 角色 | 位置 |
|------|------|------|
| `program.md` | 研究目标，人类维护 | `d:\ClothesNetData\bpt\` |
| `train_cloud.py` | 训练代码，Claude 可编辑 | `d:\ClothesNetData\bpt\` (v6 final) |
| `monitor.py` | 30分钟巡检+自动修复 | `d:\ClothesNetData\monitor.py` |
| `infer_and_eval.py` | 推理+评估 | `d:\ClothesNetData\bpt\` |
| `autoresearch_pipeline.py` | 流程编排器 | `d:\ClothesNetData\bpt\` |
| `AUTORESEARCH_PIPELINE.md` | 完整流程文档 | `d:\ClothesNetData\` |
| `crash_monitor.py` | 服务器端崩溃监控 | `d:\ClothesNetData\bpt\` |
| `deploy_all.sh` | 一键部署脚本 | `d:\ClothesNetData\` |

### 5.3 部署流程 (6步)

```bash
# Step 1: 数据验证
python -c "import torch; assert torch.cuda.is_available()" && \
  echo "Train: $(ls supervised_data/*/ | wc -l) | Test: $(ls test_set/*/ | wc -l)"

# Step 2: Overfit test
python overfit_test.py --n_samples 3 --epochs 50 --max_code_len 4000 --lr 2e-5
# 必须 loss < 0.5 才能继续

# Step 3: 打包上传
tar czf deploy.tar.gz bpt/ supervised_data/ --exclude="*.zip"
scp -P PORT deploy.tar.gz root@HOST:/root/autodl-tmp/

# Step 4: 云端部署
ssh -p PORT root@HOST << 'EOF'
cd /root/autodl-tmp && tar xzf deploy.tar.gz && cd bpt
python -B train_cloud.py --config config/BPT-open-8k-8-16.yaml \
  --model_path weights/bpt-8-16-500m.pt --data_dir ../supervised_data \
  --cache_dir ./cached_embeddings --output_dir ./checkpoints \
  --epochs 30 --batch_size 2 --max_code_len 10000 --grad_accum_steps 4 \
  --lr 1e-5 --save_every 5 --augment > train_full.log 2>&1 &
EOF

# Step 5: 启动监控 (本地)
python monitor.py  # cron: 每30分钟 SSH 巡检

# Step 6: 训练完成 → 推理评估 → 反馈调参 → 下一轮
```

### 5.4 GPU 选型速查

| GPU | VRAM | fp精度 | batch | max_len | 覆盖 | 约价 |
|-----|------|--------|-------|---------|------|------|
| RTX 4090 | 24GB | fp16 | 1 | 2,000 | ~5% | ¥2/h |
| V100 | 32GB | fp16 | 1 | 4,000 | ~10% | 免费(学校) |
| A100 40G | 40GB | fp16 | 2 | 6,000 | ~30% | ¥3.5/h |
| A800 80G | 80GB | fp32 | 4 | 10,000 | ~50% | ¥6.3/h |
| **H20 96GB** | **96GB** | **fp32** | **2** | **10,000** | **~50%** | **¥7.4/h** |
| H200 141G | 141GB | fp32 | 4 | 18,000 | ~80% | ¥15/h |

---

## 六、QuadRemesher 工具包

### 6.1 位置
`d:\ClothesNetData\quadremesher_tools.zip` (40KB, 25个文件)

### 6.2 包含脚本

| 类别 | 脚本 | 功能 |
|------|------|------|
| 半自动处理 | `quadremesher_manual_pipeline.py` | Blender GUI 中逐个 mesh 点 REMESH IT |
| | `supervised_data_pipeline.py` | 批量生成监督数据 (remesh+破损+点云) |
| 全自动处理 | `blender_quadremesher_bpt_converter.py` | QuadriFlow 全自动重网格化 |
| | `batch_supervised_pipeline.py` | 批量处理全部 OBJ→supervised_data |
| | `process_one_model.py` | Blender 后台单样本处理 |
| | `remesh_to_target.py` | QuadriFlow target face count |
| 面数标准化 | `standardize_faces.py` (+pass2/3) | 三轮标准化，排除顽固三角面 |
| 预处理 | `blender_process_clothes.py` | 基础: Limited Dissolve + Tris to Quads |
| | `blender_process_clothes_advanced.py` | 高级: remove_doubles + decimate |
| | `blender_garment_preprocessing.py` | Solidify + 锐边 + 法线 |
| BPT 转换 | `bpt_quad_converter.py` | Quad mesh → BPT 格式 |
| | `garment_bpt_pipeline.py` | 完整流水线: remesh→BPT |
| 质量检查 | `mesh_sanity_check.py` | 网格验证 |
| | `visual_verify.py` | 可视化验证 |

### 6.3 QuadRemesher 使用要点

- 插件地址: https://exoside.com/
- Blender 内操作: 选 mesh → 右侧面板 → QuadRemesher → 设置参数 → REMESH IT
- 推荐参数: Target Face Count=8000, Adaptive Size=0, Normals Splitting=ON
- 局限: `bpy.ops.qremesher.remesh()` 是 modal operator，必须在 Blender GUI 中执行
- 替代方案: `bpy.ops.object.quadriflow_remesh()` — Blender 内置，可全脚本化，已验证输出 100% quads

---

## 七、GitHub 仓库计划

### 7.1 仓库结构

```
github.com/<user>/bpt-garment-regularizer/
├── README.md                       # 项目说明 + 快速开始
├── AUTORESEARCH_PIPELINE.md        # AutoResearch 完整流程
├── WEEKLY_PROGRESS.md              # 本文件
├── requirements.txt                # Python 依赖
├── program.md                      # AutoResearch 研究目标
│
├── bpt/                            # BPT 模型 + 训练
│   ├── train_cloud.py              # 训练脚本 (v6 final)
│   ├── overfit_test.py             # 过拟合测试
│   ├── infer_and_eval.py           # 推理评估
│   ├── monitor.py                  # 监控脚本
│   ├── autoresearch_pipeline.py    # AutoResearch 编排
│   ├── model/                      # 模型架构
│   │   ├── model.py                # MeshTransformer (711M)
│   │   ├── serializaiton.py        # BPT 序列化 (quad支持)
│   │   ├── data_utils.py           # 网格工具
│   │   └── miche_conditioner.py    # Michelangelo 编码器
│   ├── miche/                      # Michelangelo 编码器权重
│   ├── config/                     # 训练配置
│   └── weights/                    # 预训练权重 (需下载)
│
├── preprocessing/                  # 数据预处理
│   ├── quadremesher_manual_pipeline.py
│   ├── batch_supervised_pipeline.py
│   ├── standardize_faces.py
│   └── README.md                   # 预处理说明
│
├── results/                        # 实验结果
│   ├── v100_baseline/              # V100 max_len=4000 碎片
│   └── h20_v6/                     # H20 fp32 max_len=10000
│
└── memory/                         # Claude 记忆文件
    └── project_bpt_autoresearch_skill.md
```

### 7.2 维护计划

| 频率 | 操作 |
|------|------|
| 每天 | `monitor.py` 自动巡检 (cron) |
| 每次训练完成 | 推理评估 → metrics.json → git commit |
| 每周 | 更新 WEEKLY_PROGRESS.md |
| 每个 milestone | git tag + release notes |
| 发现好配置 | 更新 program.md 最佳实践 |

### 7.3 Skill 化

完成后的模型 + 推理脚本打包为 Claude Code Skill：
```json
{
  "name": "bpt-garment-regularizer",
  "description": "从点云重建服装四边形网格",
  "inputs": ["pointcloud.ply (2048点)"],
  "outputs": ["garment_quad.obj"],
  "requirements": ["H20 96GB+", "bpt-8-16-500m.pt权重"]
}
```

---

## 八、当前状态 (截至 2026.06.09)

| 项目 | 状态 |
|------|------|
| 数据 | 2,807 train / 311 test, 嵌入 fp32 ✅ |
| 代码 | train_cloud.py v6: fp32 + warmup + NaN guard + `-B` ✅ |
| 训练 | H20 96GB, batch=2, max_len=10000, Epoch 1 进行中 🔄 |
| 监控 | cron 30分钟巡检, 崩溃自动重启 ✅ |
| Pipeline | 文档完整, 6步流程就绪 ✅ |
| GitHub | 待推送 |
| 上次训练 | v5 (fp16, 30epoch完成, 权重被NaN污染)→ 丢弃 |
| 推理验证 | 待v6训练完成后执行 |

---

## 九、下一步

1. **v6 训练完成** — 预计 20-30 小时
2. **推理验证** — 311 独立测试集, Chamfer/Hausdorff/FaceCoverage
3. **对比 V100** — 量化 H20 的提升幅度
4. **GitHub 推送** — 代码 + 结果 + pipeline 全上传
5. **Skill 注册** — Claude Code 可调用
6. **max_len 提升** — 通过 position embedding 插值安全扩展到 18000
