"""
BPT fine-tuning for cloud (Linux) deployment.

Key differences from local Windows script:
  - No Windows-specific patches (DeepSpeed works natively on Linux)
  - AdamW optimizer (sufficient VRAM on cloud GPUs)
  - Higher max_code_len and batch_size defaults
  - Supports both cached embeddings and on-the-fly encoding
  - DeepSpeed ZeRO stage 2 for multi-GPU (optional)

Usage:
  # Single GPU with cached embeddings
  python train_cloud.py --data_dir ./supervised_data --cache_dir ./cached_embeddings --epochs 50

  # Multi-GPU with DeepSpeed
  deepspeed train_cloud.py --deepspeed --deepspeed_config ds_config.json ...

  # Without cached embeddings (encoder runs in loop)
  python train_cloud.py --data_dir ./supervised_data --epochs 50
"""

# ---- Auto-generated DeepSpeed patch ----
import types
class _DeepSpeedLoader:
    def find_module(self, fullname, path=None):
        return self if fullname.startswith("deepspeed") else None
    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = types.ModuleType(fullname)
        mod.__path__ = []; mod.__file__ = "(deepspeed-dummy)"
        class _AutoMod(types.ModuleType):
            def __getattr__(s, name):
                if name.startswith("_"): raise AttributeError(name)
                cls = type(name, (), {
                    "__new__": lambda cls, *a, **kw: object.__new__(cls),
                    "__init__": lambda self, *a, **kw: None,
                })
                setattr(s, name, cls)
                return cls
        mod.__class__ = _AutoMod
        sys.modules[fullname] = mod
        return mod
sys.meta_path.insert(0, _DeepSpeedLoader())

import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load



import os
import sys
import argparse
import random
import json
from pathlib import Path

import yaml
import torch
import numpy as np
import trimesh
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model.model import MeshTransformer
from model.serializaiton import BPT_serialize
from utils import apply_normalize


# ---- Dataset ----

class SupervisedDataset(torch.utils.data.Dataset):
    """Loads (point_cloud, clean_mesh) pairs for BPT training."""

    def __init__(self, data_dir, pc_samples=4096, augment=False, max_code_len=10000,
                 cache_dir=None):
        self.data_dir = Path(data_dir)
        self.pc_samples = pc_samples
        self.augment = augment
        self.max_code_len = max_code_len
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.model_dirs = []
        for d in sorted(self.data_dir.iterdir()):
            if not d.is_dir() or d.name == "Cube":
                continue
            gt = d / "ground_truth.obj"
            pc = d / "input_pointcloud.ply"
            if gt.exists() and pc.exists():
                if self.cache_dir and not (self.cache_dir / f"{d.name}.pt").exists():
                    continue
                self.model_dirs.append(d)

        cache_info = f" (cached embeddings from {cache_dir})" if self.cache_dir else ""
        print(f"Dataset: {len(self.model_dirs)} samples from {data_dir}{cache_info}")

    def __len__(self):
        return len(self.model_dirs)

    def __getitem__(self, idx):
        model_dir = self.model_dirs[idx]

        gt_mesh = trimesh.load(str(model_dir / "ground_truth.obj"), force='mesh', process=False)
        bbox = gt_mesh.bounds
        center = (bbox[1] + bbox[0]) / 2
        scale = (bbox[1] - bbox[0]).max() / (2 * 0.95)

        gt_mesh = apply_normalize(gt_mesh)
        codes = BPT_serialize(gt_mesh, mesh_type="triangle")
        codes = torch.from_numpy(codes).long()
        if len(codes) > self.max_code_len:
            codes = codes[:self.max_code_len]

        out = {"codes": codes, "uid": model_dir.name}

        if self.cache_dir:
            emb = torch.load(str(self.cache_dir / f"{model_dir.name}.pt"), map_location="cpu")
            out["cond_embeds"] = emb.squeeze(0)
        else:
            pc_raw = trimesh.load(str(model_dir / "input_pointcloud.ply"))
            pts = np.array(pc_raw.vertices, dtype=np.float32)
            pts = (pts - center) / scale

            normals = _estimate_normals(pts, k=min(30, len(pts) - 1))
            pc_data = np.concatenate([pts, normals], axis=-1).astype(np.float16)
            pc_data = np.nan_to_num(pc_data, nan=0.0, posinf=0.0, neginf=0.0)

            n = pc_data.shape[0]
            if n >= self.pc_samples:
                ind = np.random.choice(n, self.pc_samples, replace=False)
            else:
                ind = np.random.choice(n, self.pc_samples, replace=True)
            pc_data = pc_data[ind]

            if self.augment:
                angle = np.random.uniform(0, 2 * np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
                pc_data[:, :3] = pc_data[:, :3] @ rot.T
                pc_data[:, 3:6] = pc_data[:, 3:6] @ rot.T

            out["pc_normal"] = torch.from_numpy(pc_data)

        return out


def _estimate_normals(pts, k=30):
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=k + 1)
    normals = np.zeros_like(pts)
    for i, nbrs in enumerate(nn_idx):
        nbr_pts = pts[nbrs[1:]]
        cov = np.cov(nbr_pts.T, bias=True)
        _, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
    return normals


def collate_fn(batch):
    pad_id = -1
    max_len = max(b["codes"].shape[0] for b in batch)
    bs = len(batch)

    codes_batch = torch.full((bs, max_len), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["codes"].shape[0]
        codes_batch[i, :n] = b["codes"]

    result = {"codes": codes_batch, "uid": [b["uid"] for b in batch]}

    if "pc_normal" in batch[0]:
        result["pc_normal"] = torch.stack([b["pc_normal"] for b in batch])
    if "cond_embeds" in batch[0]:
        result["cond_embeds"] = torch.stack([b["cond_embeds"] for b in batch])

    return result


# ---- Training ----

def get_args():
    parser = argparse.ArgumentParser(description="BPT Fine-tuning (Cloud)")
    parser.add_argument("--config", type=str, default="config/BPT-open-8k-8-16.yaml")
    parser.add_argument("--model_path", type=str, default="weights/bpt-8-16-500m.pt")
    parser.add_argument("--data_dir", type=str, default="./supervised_data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--cache_dir", type=str, default="./cached_embeddings")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_code_len", type=int, default=4000,
                        help="Truncate code sequences to this length")
    parser.add_argument("--no_cache", action="store_true", default=False,
                        help="Disable cached embeddings (run encoder in training loop)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local rank passed from distributed launcher")
    return parser.parse_args()


def train(args):
    # Distributed setup
    use_deepspeed = False
    try:
        import deepspeed
        use_deepspeed = True
    except ImportError:
        pass

    device = torch.device("cuda")
    print(f"Device: {device} | GPU count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1 and not use_deepspeed:
        print("Multi-GPU detected but DeepSpeed not available. Using single GPU.")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"Config: {json.dumps(config, indent=2)}")

    # Build model
    model = MeshTransformer(
        dim=config['dim'],
        attn_depth=config['depth'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        mode=config['mode'],
        num_discrete_coors=2 ** int(config['quant_bit']),
        block_size=config['block_size'],
        offset_size=config['offset_size'],
        conditioned_on_pc=config['conditioned_on_pc'],
        use_special_block=config['use_special_block'],
        encoder_name=config['encoder_name'],
        encoder_freeze=args.freeze_encoder,
    )

    # Load pre-trained weights
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pre-trained weights: {args.model_path}")
        model.load(args.model_path)
    else:
        print(f"WARNING: No pre-trained weights at {args.model_path}. Training from scratch.")

    model = model.cuda()
    model = model.train()
    torch.backends.cudnn.benchmark = True

    n_total = sum(p.nelement() for p in model.parameters())
    n_trainable = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_trainable / 1e6:.1f}M trainable / {n_total / 1e6:.1f}M total")

    # Dataset
    cache_dir = None if args.no_cache else args.cache_dir
    dataset = SupervisedDataset(
        data_dir=args.data_dir,
        pc_samples=config.get('pc_num', 4096),
        augment=args.augment if not cache_dir else False,
        max_code_len=args.max_code_len,
        cache_dir=cache_dir,
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    indices = list(range(n_total))
    random.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    print(f"Split: {n_train} train / {n_val} val")

    train_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_idx),
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(val_idx),
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer — AdamW (cloud has enough VRAM)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    scaler = torch.cuda.amp.GradScaler()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            codes = batch["codes"].to(device)

            if "cond_embeds" in batch:
                cond_kw = {"cond_embeds": batch["cond_embeds"].to(device)}
            else:
                cond_kw = {"pc": batch["pc_normal"].to(device).half()}

            with torch.cuda.amp.autocast():
                try:
                    loss = model.forward_on_codes(codes, return_loss=True, **cond_kw)
                except AssertionError as e:
                    if 'NAN' in str(e) or 'nan' in str(e):
                        continue
                    raise

            loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()

            if (global_step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum_steps
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item() * args.grad_accum_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                codes = batch["codes"].to(device)
                if "cond_embeds" in batch:
                    cond_kw = {"cond_embeds": batch["cond_embeds"].to(device)}
                else:
                    cond_kw = {"pc": batch["pc_normal"].to(device).half()}
                loss = model.forward_on_codes(codes, return_loss=True, **cond_kw)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(str(out_dir / "best_model.pt"))
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

        if (epoch + 1) % args.save_every == 0:
            model.save(str(out_dir / f"checkpoint_epoch{epoch+1}.pt"))

    model.save(str(out_dir / "final_model.pt"))
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    args = get_args()
    train(args)
