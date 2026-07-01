# BPT Garment Regularizer — Agent Brief

> 最后更新: 2026-06-09 | v6 训练中 | H20 96GB

## 一句话

训练 BPT 模型从 2048 点噪声点云重建规整的四边形服装 mesh。

## 当前状态

| 项目 | 状态 |
|------|------|
| 训练 | **H20 96GB, fp32, batch=2, max_len=10000, Epoch 1 进行中** |
| 数据 | 2,807 train / 311 test (独立) / 3,117 embeddings (fp32) |
| 代码 | train_cloud.py v6 — fp32 + lr warmup + NaN gradient guard + `-B` |
| 监控 | cron 30min 巡检, 崩溃自动重启 |
| 推理 | 待训练完成, 311 测试集评估 |
| 上次完成 | v5 (fp16, 30epoch, NaN污染权重→丢弃) |

## 关键指标

- 模型: 711M 参数 MeshTransformer (BPT)
- 数据源: ClothesNetM + Other_clothes
- 测试集: 311 样本, 完全独立, 一次都没动过
- 评估: Chamfer (%bbox), Hausdorff, FaceCoverage, QuadRatio

## 文件地图

```
d:\ClothesNetData\
├── WEEKLY_PROGRESS.md           ← 周报 (人类读)
├── AUTORESEARCH_PIPELINE.md     ← AutoResearch 流程 (Agent 读)
├── monitor.py                   ← 30min 巡检脚本
├── bpt\                         ← 核心代码
│   ├── train_cloud.py           ← 训练脚本 (v6 final)
│   ├── overfit_test.py          ← 训前验证
│   ├── infer_and_eval.py        ← 推理+评估
│   ├── autoresearch_pipeline.py ← AutoResearch 编排
│   ├── program.md               ← 研究目标
│   ├── config\                  ← 训练配置
│   ├── model\                   ← 模型架构
│   ├── weights\                 ← bpt-8-16-500m.pt (需下载)
│   └── cached_embeddings\       ← 3117×.pt (预计算, 已转fp32)
├── supervised_data\             ← 训练数据 (2807样本)
├── test_set\                    ← 独立测试集 (311样本, 不动!)
├── project_docs\                ← 旧文档/脚本
├── quadremesher_tools.zip       ← QuadRemesher 工具包
└── memory\                      ← Agent 记忆文件
```

## 训练命令 (v6)

```bash
python -B train_cloud.py \
  --config config/BPT-open-8k-8-16.yaml \
  --model_path weights/bpt-8-16-500m.pt \
  --data_dir ../supervised_data \
  --cache_dir ./cached_embeddings \
  --output_dir ./checkpoints \
  --epochs 30 --batch_size 2 --max_code_len 10000 \
  --grad_accum_steps 4 --lr 1e-5 --save_every 5 --augment
```

## 已知陷阱 (Agent 必读)

| 陷阱 | 修复 |
|------|------|
| **不要 sed 改代码** | 本地写 → 编译 → SCP 一次上传 |
| **必须 `python -B`** | 否则 .pyc 覆盖新代码 |
| **嵌入了 fp16?** | 必须全转 fp32 (`emb.float()`) |
| **OOM?** | fp32 用 batch=2, fp16 可用 batch=4 |
| **NaN epoch?** | 检查 `torch.isfinite()` 是否在 backward 前 |
| **推理碎片?** | max_len 不够, 换大 GPU 或插值扩展 pos_emb |
| **SSH 连不上?** | AutoDL 实例可能关机, 去控制台开机 |

## 当前部署

- 平台: AutoDL (autodl.com)
- 实例: H20-NVLink 96GB, region-42
- SSH: `ssh -p 37805 root@region-42.seetacloud.com`
- 密码: `2gPzTHOKQqfZ`
- 数据盘: `/root/autodl-tmp/` (50GB)

## Agent 工作流

```python
# 1. 检查状态
from monitor import check_status
status = ssh_exec("grep 'Epoch' train_full.log | tail -3")

# 2. 如果崩溃
if not alive and errors > 0:
    error = ssh_exec("grep Error train_full.log | tail -5")
    fix_and_restart(error)

# 3. 如果完成
if "Training complete" in log:
    ssh_exec("python infer_and_eval.py --model_path checkpoints/final_model.pt \
              --test_dir ../test_set --mesh_type quad")
    results = json.load(open("eval_results/metrics.json"))
    decide_next(results)  # 见 AUTORESEARCH_PIPELINE.md

# 4. 监控循环
cron("*/30 * * * *", monitor.check)
```

## 下一次训练计划

1. v6 完成后 → 推理验证 → 如果 Chamfer<5%: 尝试 max_len=18000 (position embedding 插值)
2. 如果 Chamfer>10%: 检查数据质量, 排查极端 mesh
3. 目标: Chamfer<2%, QuadRatio>90%
