"""
Overfit test: train on 3-5 samples, verify loss → 0.
Confirms pipeline correctness before full 50-epoch training.
Run on any GPU, even 8GB.

Usage:
  python overfit_test.py --n_samples 3 --epochs 100
"""

import os
import sys
import types
import argparse
from pathlib import Path

# ---- Windows/Linux patches ----
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

import torch as _torch
_orig_load = _torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
_torch.load = _patched_load

import yaml
import torch
import numpy as np
import trimesh
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model.model import MeshTransformer
from model.serializaiton import BPT_serialize
from utils import apply_normalize


class OverfitDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, n_samples=3, max_code_len=2000, cache_dir=None):
        self.data_dir = Path(data_dir)
        self.max_code_len = max_code_len
        self.cache_dir = Path(cache_dir) if cache_dir else None

        model_dirs = []
        for d in sorted(self.data_dir.iterdir()):
            if not d.is_dir() or d.name == "Cube":
                continue
            gt = d / "ground_truth.obj"
            pc = d / "input_pointcloud.ply"
            if gt.exists() and pc.exists():
                if self.cache_dir and not (self.cache_dir / f"{d.name}.pt").exists():
                    continue
                model_dirs.append(d)

        # Shuffle with fixed seed for consistent selection
        import random
        random.Random(42).shuffle(model_dirs)
        self.model_dirs = model_dirs[:n_samples]
        self.n_samples = len(self.model_dirs)
        print(f"Overfit dataset: {self.n_samples} samples")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        model_dir = self.model_dirs[idx]

        gt_mesh = trimesh.load(str(model_dir / "ground_truth.obj"), force='mesh', process=False)
        bbox = gt_mesh.bounds
        center = (bbox[1] + bbox[0]) / 2
        scale = (bbox[1] - bbox[0]).max() / (2 * 0.95)
        gt_mesh = apply_normalize(gt_mesh.copy())
        codes = BPT_serialize(gt_mesh, mesh_type="triangle")
        codes = torch.from_numpy(codes).long()
        if len(codes) > self.max_code_len:
            codes = codes[:self.max_code_len]

        out = {"codes": codes, "uid": model_dir.name}

        if self.cache_dir:
            emb = torch.load(str(self.cache_dir / f"{model_dir.name}.pt"), map_location="cpu")
            out["cond_embeds"] = emb.squeeze(0)
        else:
            # Quick PC loading without normals (just positions padded)
            pc_raw = trimesh.load(str(model_dir / "input_pointcloud.ply"))
            pts = np.array(pc_raw.vertices, dtype=np.float32)
            pts = (pts - center) / scale

            n = pts.shape[0]
            if n >= 4096:
                ind = np.random.choice(n, 4096, replace=False)
            else:
                ind = np.random.choice(n, 4096, replace=True)
            pts = pts[ind]
            zeros = np.zeros_like(pts)
            pc_data = np.concatenate([pts, zeros], axis=-1).astype(np.float16)
            out["pc_normal"] = torch.from_numpy(pc_data)

        return out


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


def main():
    parser = argparse.ArgumentParser(description="Overfit test for BPT pipeline")
    parser.add_argument("--config", type=str, default="config/BPT-open-8k-8-16.yaml")
    parser.add_argument("--model_path", type=str, default="weights/bpt-8-16-500m.pt")
    parser.add_argument("--data_dir", type=str, default="D:/ClothesNetData/supervised_data")
    parser.add_argument("--cache_dir", type=str, default="cached_embeddings")
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_code_len", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Overfit test: {args.n_samples} samples, {args.epochs} epochs")

    # Config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Model
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
        encoder_freeze=True,
    )

    if os.path.exists(args.model_path):
        print(f"Loading pretrained: {args.model_path}")
        model.load(args.model_path)
    else:
        print(f"WARNING: No pretrained weights at {args.model_path}")

    model = model.to(device).train()
    torch.backends.cudnn.benchmark = True

    # Dataset
    dataset = OverfitDataset(
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        max_code_len=args.max_code_len,
        cache_dir=args.cache_dir,
    )
    loader = DataLoader(
        dataset, batch_size=args.n_samples,
        collate_fn=collate_fn, shuffle=True,
    )

    # Optimizer — AdamW for fast overfitting (can OOM on <24GB GPUs)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n{'='*50}")
    print(f"Training {args.epochs} epochs on {dataset.n_samples} samples...")
    print(f"Token lengths: {[len(dataset[i]['codes']) for i in range(len(dataset))]}")
    print(f"Target: loss should drop to <0.5 (ideally <0.1)")
    print(f"{'='*50}\n")

    best_loss = float('inf')
    initial_loss = None

    for epoch in range(args.epochs):
        epoch_loss = 0.0

        for batch in loader:
            codes = batch["codes"].to(device)
            if "cond_embeds" in batch:
                cond_kw = {"cond_embeds": batch["cond_embeds"].to(device)}
            else:
                cond_kw = {"pc": batch["pc_normal"].to(device).half()}

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward_on_codes(codes, return_loss=True, **cond_kw)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        best_loss = min(best_loss, avg_loss)

        if initial_loss is None:
            initial_loss = avg_loss

        if epoch == 0 or (epoch + 1) % 10 == 0:
            reduction = initial_loss / max(avg_loss, 1e-8)
            print(f"Epoch {epoch+1:4d} | loss={avg_loss:.4f} | best={best_loss:.4f} | "
                  f"↓{reduction:.1f}x from epoch 0 | {'OK' if avg_loss < 0.5 else '...'}")

        if avg_loss < 0.05:
            print(f"\n  Converged at epoch {epoch+1} (loss < 0.05)")
            break

    # Verdict
    print(f"\n{'='*50}")
    if best_loss < 0.1:
        print(f"PASS: loss {best_loss:.4f} < 0.1 — pipeline works correctly")
    elif best_loss < 0.5:
        print(f"BORDERLINE: loss {best_loss:.4f} — check data/learning rate")
    elif best_loss < 1.0:
        print(f"WEAK: loss {best_loss:.4f} — possible issue with data alignment")
    else:
        print(f"FAIL: loss {best_loss:.4f} > 1.0 — pipeline has a bug")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
