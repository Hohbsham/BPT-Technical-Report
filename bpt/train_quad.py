"""
BPT fine-tuning: train autoregressive transformer to reconstruct clean meshes
from noisy sparse point clouds.

Architecture:
  Michelangelo encoder (frozen) → 256+1 latent embeddings
  BPT Transformer             → autoregressive token prediction
  Cross-entropy loss on block/offset token sequences

Data flow:
  input_pointcloud.ply (2048 noisy pts) → 4096 pts + PCA normals → encoder
  ground_truth.obj    (clean mesh)      → BPT_serialize          → target codes
  Loss = CE(predicted_token, target_token)

Usage:
  python train_quad.py --model_path weights/bpt-8-16-500m.pt --epochs 50
"""

import os
import sys
import types
import argparse
import random
import json
from pathlib import Path

# ---- Windows patches (must run before any model imports) ----
# Patch 1: dummy deepspeed modules (checkpoint used DeepSpeed, unavailable on Windows)
class _DeepSpeedLoader:
    def find_module(self, fullname, path=None):
        return self if fullname.startswith('deepspeed') else None
    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = types.ModuleType(fullname)
        mod.__path__ = []; mod.__file__ = '(deepspeed-dummy)'
        class _AutoMod(types.ModuleType):
            def __getattr__(s, name):
                if name.startswith('_'): raise AttributeError(name)
                cls = type(name, (), {
                    '__new__': lambda cls, *a, **kw: object.__new__(cls),
                    '__init__': lambda self, *a, **kw: None,
                })
                setattr(s, name, cls)
                return cls
        mod.__class__ = _AutoMod
        sys.modules[fullname] = mod
        return mod
sys.meta_path.insert(0, _DeepSpeedLoader())

# Patch 2: torch.load defaults to weights_only=True in PyTorch 2.6+, but the
# checkpoint contains deepspeed objects. Force weights_only=False.
import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

import yaml
import torch
import torch.nn.functional as F
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
    """Loads (noisy_point_cloud, clean_mesh) pairs for BPT training."""

    def __init__(self, data_dir, pc_samples=4096, augment=False, max_code_len=10000,
                 cache_dir=None):
        self.data_dir = Path(data_dir)
        self.pc_samples = pc_samples
        self.augment = augment
        self.max_code_len = max_code_len
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.model_dirs = []
        for d in sorted(self.data_dir.iterdir()):
            if not d.is_dir():
                continue
            if d.name == "Cube":  # skip Blender default
                continue
            gt = d / "ground_truth.obj"
            pc = d / "input_pointcloud.ply"
            if gt.exists() and pc.exists():
                # When using cache, skip samples without cached embedding
                if self.cache_dir:
                    emb = self.cache_dir / f"{d.name}.pt"
                    if not emb.exists():
                        continue
                self.model_dirs.append(d)

        if self.cache_dir:
            print(f"Dataset: {len(self.model_dirs)} samples from {data_dir} (cached embeddings from {cache_dir})")
        else:
            print(f"Dataset: {len(self.model_dirs)} samples from {data_dir}")

    def __len__(self):
        return len(self.model_dirs)

    def __getitem__(self, idx):
        model_dir = self.model_dirs[idx]

        # Load clean mesh, normalize, serialize to token codes
        gt_mesh = trimesh.load(str(model_dir / "ground_truth.obj"), force='mesh', process=False)

        bbox = gt_mesh.bounds
        center = (bbox[1] + bbox[0]) / 2
        scale = (bbox[1] - bbox[0]).max() / (2 * 0.95)

        gt_mesh = apply_normalize(gt_mesh)

        codes = BPT_serialize(gt_mesh, mesh_type="triangle")
        codes = torch.from_numpy(codes).long()

        if len(codes) > self.max_code_len:
            codes = codes[:self.max_code_len]

        out = {
            "codes": codes,
            "uid": model_dir.name,
        }

        # Load cached embedding if available
        if self.cache_dir:
            emb_file = self.cache_dir / f"{model_dir.name}.pt"
            emb = torch.load(emb_file, map_location="cpu")  # (1, 257, 1024) fp16
            out["cond_embeds"] = emb.squeeze(0)  # (257, 1024)
        else:
            # Real-time point cloud loading + encoding
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
    """Estimate point normals via local PCA (smallest eigenvector)."""
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

    uids = [b["uid"] for b in batch]

    result = {"codes": codes_batch, "uid": uids}

    if "pc_normal" in batch[0]:
        result["pc_normal"] = torch.stack([b["pc_normal"] for b in batch])

    if "cond_embeds" in batch[0]:
        result["cond_embeds"] = torch.stack([b["cond_embeds"] for b in batch])

    return result


# ---- Training ----

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. Training will be extremely slow.")

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
        print("WARNING: No pre-trained weights. Training from scratch.")
        if not args.model_path:
            print("  (no --model_path specified)")
        else:
            print(f"  (file not found: {args.model_path})")

    model = model.cuda()
    model = model.train()
    scaler = torch.cuda.amp.GradScaler()

    # Memory optimizations
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    n_total = sum(p.nelement() for p in model.parameters())
    n_trainable = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_trainable / 1e6:.1f}M trainable / {n_total / 1e6:.1f}M total")

    # Dataset
    cache_dir = getattr(args, 'cache_dir', None)
    dataset = SupervisedDataset(
        data_dir=args.data_dir,
        pc_samples=config.get('pc_num', 4096),
        augment=args.augment if not cache_dir else False,  # no augmentation with cached embeddings
        max_code_len=args.max_code_len,
        cache_dir=cache_dir,
    )

    # Train/val split (90/10)
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
    )
    val_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(val_idx),
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # Optimizer - use SGD with Nesterov momentum instead of AdamW to save ~2.1GB GPU memory
    # AdamW stores 2 buffers per param (m, v); SGD stores just 1 (momentum)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
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

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(str(out_dir / "best_model.pt"))
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            model.save(str(out_dir / f"checkpoint_epoch{epoch+1}.pt"))

    # Final save
    model.save(str(out_dir / "final_model.pt"))
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPT Fine-tuning on Supervised Data")
    parser.add_argument("--config", type=str, default="config/BPT-open-8k-8-16.yaml")
    parser.add_argument("--model_path", type=str, default="weights/bpt-8-16-500m.pt")
    parser.add_argument("--data_dir", type=str, default=r"D:\ClothesNetData\supervised_data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_code_len", type=int, default=4000,
                        help="Truncate token sequences to this length (model max_seq_len still loads from config)")
    parser.add_argument("--cache_dir", type=str, default="./cached_embeddings",
                        help="Directory with pre-computed Michelangelo embeddings (skip encoder if set)")
    args = parser.parse_args()
    train(args)
