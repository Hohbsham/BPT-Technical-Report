# BPT Garment Regularizer — AutoResearch Program

## Goal
Train a BPT model that reconstructs clean quad clothing meshes from noisy point clouds (2048 points).

## Pipeline
```
train_cloud.py → best_model.pt → infer_and_eval.py → metrics.json → feedback → retrain
```

## Training Config (editable by Agent)
All in `train_cloud.py`:
- `--lr` (default 5e-5)
- `--max_code_len` (default 2000, target 8000+)
- `--epochs` (default 50)
- `--batch_size` (default 1)
- `--grad_accum_steps` (default 8)
- `--augment` (default on)

## Evaluation Metrics (from infer_and_eval.py)
- `val_loss` — cross-entropy on validation set (LOWER is better)
- `chamfer_dist` — Chamfer distance %bbox (LOWER is better, < 2% = EXCELLENT)
- `hausdorff_dist` — Hausdorff distance %bbox (LOWER is better, < 5% = EXCELLENT)
- `face_coverage` — % of faces correctly reconstructed (HIGHER is better, > 80% = EXCELLENT)
- `quad_ratio` — % of quad faces (HIGHER is better, > 90% = EXCELLENT)

## AutoResearch Loop
1. Start training with current best config
2. Monitor val_loss each epoch
3. At checkpoint (every 5 epochs), run inference on 5 test samples
4. Compute all metrics
5. If val_loss improved AND chamfer_dist < 5%: commit + tag "good"
6. If val_loss improved but mesh quality bad: increase max_code_len
7. If val_loss plateaued (> 5 epochs no improvement): reduce lr, continue
8. If OOM: reduce max_code_len or batch_size

## Test Data
- `supervised_data_clean/` — 5 samples with noise-free point clouds (for generalization test)
- `supervised_data/` (val split) — noisy point clouds (for regularization test)

## Success Criteria
- val_loss < 0.5
- Chamfer distance < 2% bbox on clean inputs
- Face coverage > 80% on clean inputs
- Quad ratio > 90%
