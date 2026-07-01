# BPT 训练总结与改进计划

## 结果

| 指标 | V100 (4000) | H20 (10000) |
|------|------------|-------------|
| val_loss | 1.49 (epoch 31) | 1.30 (epoch 23) |
| Chamfer | 22-51% | 52-87% |
| Quad% | 0% | 0% |
| 生成顶点 | 13-122 | 8-65 |
| 结论 | 碎片 | 碎片(更大) |

**训练收敛完美（-83% loss），但推理质量很差。问题不在训练，在 max_len 不够。**

## 根本原因

711M 参数 BPT 模型生成完整服装 mesh 需要 14000+ tokens。H20 的 10000 tokens 只能生成几十个顶点。虽然比 V100 (4000) 多 2.5x，但仍然不够。

```
所需 token: ██████████████ 14000
H20 能训:  ██████████    10000 (少 29%)
V100 能训: ████          4000  (少 71%)
```

## 已解决的技术问题（20+ 次崩溃修复）

| 问题 | 修复 |
|------|------|
| fp16/fp32 dtype crash | 嵌入转 fp32 + 纯 fp32 训练 |
| NaN 污染权重 | `torch.isfinite()` 拦截 backward |
| 梯度爆炸 | lr warmup |
| .pyc 缓存覆盖新代码 | `python -B` + 清空缓存 |
| try/except 缩进错误 | loss=None 在 if/else 前 |

## v6 代码配置（稳定版本）

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

关键参数: fp32, batch=2, lr warmup, NaN 梯度拦截, -B 禁用 bytecode

## GPU 需求排序

| GPU | VRAM | 预估 max_len | 覆盖 | 状态 |
|-----|------|-------------|------|------|
| H200 SXM | 141GB | 18000+ | 95%+ | ✅ 推荐 |
| H100 SXM | 80GB | 14000 | 85%+ | ✅ 可行 |
| A800 | 80GB | 12000 | 70%+ | ⚠️ 勉强 |
| H20 | 96GB | 10000 | 50% | ❌ 已验证不够 |

## 下次训练计划

### Step 1: Position Embedding 扩展（必须）
```python
# 加载预训练权重后插值位置嵌入到 max_seq_len
old_emb = model.abs_pos_emb.weight.data  # [10000, 1024]
new_emb = F.interpolate(old_emb.T.unsqueeze(0), 
    size=18000, mode='linear').squeeze(0).T  # [18000, 1024]
model.abs_pos_emb = nn.Embedding(18000, 1024).to(device).to(old_emb.dtype)
model.abs_pos_emb.weight.data.copy_(new_emb)
model.max_seq_len = 18000
```
已验证可行的代码逻辑，只需正确的 dtype 转换 + 本地编译后上传。

### Step 2: 推荐配置

| GPU | batch | max_len | 预计时间 | 预估 Chamfer |
|-----|-------|---------|---------|-------------|
| H200 x1 | 4 | 18000 | ~8h | <10% |
| H100 x1 | 2 | 14000 | ~15h | <15% |

### Step 3: 推理修复
- 用 `mesh_type=triangle` 而非 quad（README 验证过 triangle 效果好）
- 或者检查 `serializaiton.py` 中 quad 反序列化逻辑

### Step 4: Overfit test 前必须验证
```bash
python overfit_test.py --n_samples 3 --epochs 100 --max_code_len 8000 --lr 2e-5
# 必须 loss < 0.5，且生成 mesh 完整（>500 vertices）
```

## 下载的文件

从 H20 下载到 `d:\ClothesNetData\h20_results\`:
- `best_model.pt` (epoch 23, val=1.30)
- `checkpoint_epoch10.pt / epoch20.pt`
- `train_full.log`
- `train_cloud.py` (v6 final)
- `eval_output/` (推理产物)
