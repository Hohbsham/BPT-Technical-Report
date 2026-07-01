# BPT 云端训练部署指南

## 文件清单

需要上传到云端的内容：

| 文件/目录 | 大小 | 说明 |
|-----------|------|------|
| `bpt/` (整个项目) | ~200MB | 模型代码、配置、miche 编码器 |
| `supervised_data/` | 1.8GB | 3051 个样本 (pointcloud.ply + ground_truth.obj) |
| `cached_embeddings/` | 1.6GB | 预计算的 Michelangelo 编码器输出 |
| `weights/bpt-8-16-500m.pt` | 1.6GB | BPT 预训练权重 (含编码器) |
| **合计** | **~5.2GB** | |

## 云端环境配置

### 1. 上传数据
```bash
# 从本地上传 (rsync)
rsync -avz --progress bpt/ user@cloud-ip:/workspace/bpt/
rsync -avz --progress supervised_data/ user@cloud-ip:/workspace/supervised_data/
rsync -avz --progress cached_embeddings/ user@cloud-ip:/workspace/cached_embeddings/

# 或使用 scp
scp -r bpt supervised_data cached_embeddings user@cloud-ip:/workspace/
```

### 2. 安装依赖
```bash
cd /workspace/bpt
pip install -r requirements_cloud.txt
```

### 3. 启动训练
```bash
# 基础配置 (epochs=50, max_code_len=4000, batch_size=2, grad_accum=4)
bash run_cloud.sh

# 自定义参数
bash run_cloud.sh --epochs 100 --max_code_len 6000 --batch_size 4

# 更大 batch (A100 40GB)
MAX_CODE_LEN=6000 BATCH_SIZE=4 GRAD_ACCUM=2 bash run_cloud.sh

# 不用缓存嵌入 (编码器在训练循环中运行)
bash run_cloud.sh --no_cache

# 多 GPU (DeepSpeed ZeRO-2)
deepspeed train_cloud.py \
    --model_path weights/bpt-8-16-500m.pt \
    --data_dir ./supervised_data \
    --cache_dir ./cached_embeddings \
    --epochs 50 \
    --batch_size 4 \
    --max_code_len 6000 \
    --deepspeed \
    --deepspeed_config ds_config.json
```

## 云端 GPU 配置建议

| GPU | VRAM | 推荐 batch_size | 推荐 max_code_len |
|-----|------|----------------|------------------|
| T4 | 16GB | 1-2 | 3000-4000 |
| A10 | 24GB | 2-4 | 4000-6000 |
| A100 | 40GB | 4-8 | 6000-10000 |
| A100 | 80GB | 8-16 | 10000 |

## 预计训练时间 (A100 40GB)

| max_code_len | batch_size | s/it | 小时/epoch | 50 epochs |
|-------------|-----------|------|-----------|-----------|
| 4000 | 2 | ~1.5 | ~0.6 | ~30h |
| 6000 | 4 | ~2.5 | ~0.5 | ~25h |
| 10000 | 4 | ~4.0 | ~0.8 | ~40h |

## 监控

```bash
# 查看训练日志
tail -f checkpoints/train.log

# GPU 使用情况
watch -n 1 nvidia-smi
```

## 输出

- `checkpoints/best_model.pt` — 验证集 loss 最低的模型
- `checkpoints/checkpoint_epoch{N}.pt` — 每 5 epoch 的检查点
- `checkpoints/final_model.pt` — 最终模型
