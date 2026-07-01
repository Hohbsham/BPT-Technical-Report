# BPT Garment Regularizer — AutoResearch Pipeline

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     AutoResearch Loop                           │
│                                                                 │
│  program.md ──► Claude ──► train_cloud.py ──► H20 GPU ──► val  │
│       ▲              │              │               │           │
│       │              │              │               ▼           │
│       │         edit code      SSH deploy      checkpoint       │
│       │              │              │               │           │
│       │              ▼              ▼               ▼           │
│       └──────── feedback ◄── infer_and_eval ◄──── model.pt     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 文件清单

| 文件 | 角色 | 谁改 |
|------|------|------|
| `program.md` | 研究目标 | 人类写 |
| `train_cloud.py` | 训练代码 | Claude 改 |
| `monitor.py` | 30分钟检查 | 自动 |
| `infer_and_eval.py` | 推理+评估 | 固定 |
| `autoresearch_history.json` | 实验记录 | 自动 |

## Step 1: 数据准备

```bash
# 1. 清理数据
python clean_data.py  # 移除面数>5000的脏样本

# 2. 分离测试集 (10%)
python split_test.py --data_dir supervised_data --test_dir test_set --ratio 0.1

# 3. 预计算嵌入 (一次性)
python precompute_embeddings.py --data_dir supervised_data --output_dir cached_embeddings

# 4. 转 fp32 (避免 dtype crash)
python -c "
import torch, os
for f in os.listdir('cached_embeddings'):
    if f.endswith('.pt'):
        emb = torch.load(f'cached_embeddings/{f}', map_location='cpu')
        torch.save(emb.float(), f'cached_embeddings/{f}')
"
```

## Step 2: 本地验证 → 云端部署

```bash
# 2a. 本地 Overfit Test (必须)
python overfit_test.py --n_samples 3 --epochs 100 --max_code_len 4000 --lr 2e-5

# 2b. 通过后才打包上传
tar czf deploy.tar.gz bpt/ supervised_data/ --exclude="*.zip"

# 2c. SCP 上传到云端
scp -P PORT deploy.tar.gz root@HOST:/root/autodl-tmp/

# 2d. SSH 部署
ssh -p PORT root@HOST << 'EOF'
cd /root/autodl-tmp && tar xzf deploy.tar.gz
cd bpt && pip install -r requirements.txt

# 启动训练 (fp32, batch=2, 30 epochs, lr warmup)
python -B train_cloud.py \
    --config config/BPT-open-8k-8-16.yaml \
    --model_path weights/bpt-8-16-500m.pt \
    --data_dir ../supervised_data \
    --cache_dir ./cached_embeddings \
    --output_dir ./checkpoints \
    --epochs 30 --batch_size 2 --max_code_len 10000 \
    --grad_accum_steps 4 --lr 1e-5 --save_every 5 --augment \
    > train_full.log 2>&1 &
EOF
```

## Step 3: 训练监控 (Auto)

```python
# monitor.py — cron 每 30 分钟执行
1. SSH 连接 → 检查 VRAM/进程/错误
2. 如果崩溃 → 读错误日志 → 自动重启
3. 如果 NaN > 阈值 → 降 lr 重启
4. 如果完成 → 触发 Step 4
5. 写 monitor_log.json
```

## Step 4: 推理验证

```bash
# 训练完成后自动执行
python infer_and_eval.py \
    --model_path checkpoints/best_model.pt \
    --test_dir ../test_set \
    --output_dir eval_results \
    --mesh_type quad

# 输出 metrics.json:
# {
#   "chamfer_pct": 2.5,    // < 5% = GOOD, < 2% = EXCELLENT
#   "hausdorff_pct": 4.1,
#   "face_coverage": 78,   // > 80% = EXCELLENT
#   "quad_ratio": 92,      // > 90% = EXCELLENT
#   "quality": "GOOD"
# }
```

## Step 5: AutoResearch 决策

```python
# autoresearch_pipeline.py — 根据评估结果自动调参
def decide_next(results, config):
    chamfer = results["chamfer_pct"]

    if chamfer < 2 and results["face_coverage"] > 80:
        action = "EXCELLENT"
        # 精调: 降 lr, 加 epochs
        config["lr"] *= 0.5
        config["epochs"] += 10

    elif chamfer < 5:
        action = "GOOD"
        # 提高 max_code_len
        config["max_code_len"] = min(config["max_code_len"] + 2000, 20000)
        config["lr"] *= 0.7

    elif chamfer < 10:
        action = "FAIR"
        # 增加训练量
        config["epochs"] += 10
        config["batch_size"] = min(config["batch_size"] * 2, 4)

    else:
        action = "POOR"
        # 根本性问题: 检查数据/架构/初始化
        raise NeedHumanReview("Chamfer > 10% — check data quality")

    return config, action
```

## 完整流程 (一次性部署)

```bash
# deploy_all.sh — 从零到 AutoResearch
#!/bin/bash
set -e

echo "[1/6] 环境检查"
python -c "import torch; assert torch.cuda.is_available()"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "[2/6] 数据验证"
echo "  Train: $(ls -d supervised_data/*/ | wc -l)"
echo "  Test:  $(ls -d test_set/*/ | wc -l)"
echo "  Emb:   $(ls cached_embeddings/*.pt | wc -l)"

echo "[3/6] Overfit test"
python overfit_test.py --n_samples 3 --epochs 50 --max_code_len 4000 --lr 2e-5
if [ $? -ne 0 ]; then echo "OVERFIT FAILED"; exit 1; fi

echo "[4/6] 启动训练"
nohup python -B train_cloud.py \
    --config config/BPT-open-8k-8-16.yaml \
    --model_path weights/bpt-8-16-500m.pt \
    --data_dir ../supervised_data \
    --cache_dir ./cached_embeddings \
    --output_dir ./checkpoints \
    --epochs 30 --batch_size 2 --max_code_len 10000 \
    --grad_accum_steps 4 --lr 1e-5 --save_every 5 --augment \
    > train_full.log 2>&1 &
echo "PID: $!"

echo "[5/6] 启动监控"
nohup python -B monitor.py > monitor.log 2>&1 &
echo "Monitor PID: $!"

echo "[6/6] AutoResearch 就绪"
echo "  monitor.py: 每30分钟检查/自动修复"
echo "  infer_and_eval.py: 训练完成后自动推理"
echo "  autoresearch_pipeline.py: 结果反馈 → 自动调参 → 下一轮"
```

## GPU 适配速查

| GPU | VRAM | 推荐配置 | 覆盖 |
|-----|------|---------|------|
| RTX 4090 | 24GB | fp16, batch=1, max_len=2000 | ~5% |
| V100 | 32GB | fp16, batch=1, max_len=4000 | ~10% |
| A100 40GB | 40GB | fp16, batch=2, max_len=6000 | ~30% |
| A100 80GB | 80GB | fp32, batch=4, max_len=10000 | ~50% |
| **H20 96GB** | **96GB** | **fp32, batch=2, max_len=10000** | **~50%** |
| H200 | 141GB | fp32, batch=4, max_len=18000 | ~80% |

## 常见问题速查

| 症状 | 原因 | 修复 |
|------|------|------|
| `Half != float` crash | fp16嵌入+fp32模型 | 嵌入转fp32 |
| `UnboundLocalError: loss` | NaN批continue后loss未定义 | `loss=None` 初始化 |
| epoch average NaN | 单批NaN污染 | `torch.isfinite()` 拦截 |
| 训练崩但权重完好 | 梯度爆炸(首批) | lr warmup |
| 推理碎片 | max_len太短 | 换大显存GPU |
| .pyc覆盖新代码 | Python缓存 | `python -B` |
