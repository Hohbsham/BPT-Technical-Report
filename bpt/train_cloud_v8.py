"""
BPT Training v8 — Category Token Conditioning.
Key change from v7: category label prepended to token sequence as learned embedding.
"""
import os, sys, types, argparse, random, json
from pathlib import Path
import yaml, torch, numpy as np, trimesh
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

# ---- DeepSpeed Patch ----
class _DSLoader:
    def find_module(self, f, p=None): return self if f.startswith('deepspeed') else None
    def load_module(self, f):
        if f in sys.modules: return sys.modules[f]
        m = types.ModuleType(f); m.__path__ = []; m.__file__ = '(ds)'
        class A(types.ModuleType):
            def __getattr__(s, n):
                if n.startswith('_'): raise AttributeError(n)
                c = type(n, (), {'__new__':lambda c,*a,**k:object.__new__(c),'__init__':lambda s,*a,**k:None})
                setattr(s, n, c); return c
        m.__class__ = A; sys.modules[f] = m; return m
sys.meta_path.insert(0, _DSLoader())
import torch as _t; _o=_t.load
def _p(*a,**k): k.setdefault('weights_only',False); return _o(*a,**k)
_t.load = _p

sys.path.insert(0, str(Path(__file__).parent))
from model.model import MeshTransformer
from model.serializaiton import BPT_serialize
from utils import apply_normalize
from resize_pos_emb import resize as resize_pos_emb
from category_conditioner import load_category_conditioner

CATEGORY_DIM = 1024  # matches model hidden dim


class SupervisedDatasetV8(torch.utils.data.Dataset):
    """V8 dataset: loads mesh + point cloud + category ID."""

    def __init__(self, data_dir, pc_samples=4096, augment=False, max_code_len=18000,
                 cache_dir=None, category_mapper=None):
        self.data_dir = Path(data_dir)
        self.pc_samples = pc_samples
        self.augment = augment
        self.max_code_len = max_code_len
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.mapper = category_mapper

        self.samples = []
        for d in sorted(self.data_dir.iterdir()):
            if not d.is_dir() or d.name == "Cube":
                continue
            gt = d / "ground_truth.obj"
            pc = d / "input_pointcloud.ply"
            if not gt.exists() or not pc.exists():
                continue
            if self.cache_dir and not (self.cache_dir / f"{d.name}.pt").exists():
                continue
            cat_id = self.mapper.get_category_id(d.name) if self.mapper else 11
            self.samples.append((d, cat_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        model_dir, cat_id = self.samples[idx]
        gt_mesh = trimesh.load(str(model_dir / "ground_truth.obj"), force='mesh', process=False)
        codes = BPT_serialize(gt_mesh, mesh_type="triangle")
        codes = codes[:self.max_code_len]
        codes = torch.from_numpy(codes).long()

        # Load cached embedding (fp32)
        cond_embeds = None
        if self.cache_dir:
            emb = torch.load(str(self.cache_dir / f"{model_dir.name}.pt"), map_location="cpu")
            cond_embeds = emb.squeeze(0).float()

        # Load point cloud (fallback if no cache)
        pc_raw = trimesh.load(str(model_dir / "input_pointcloud.ply"))
        pts = np.array(pc_raw.vertices, dtype=np.float32)[:self.pc_samples]

        return {
            "codes": codes,
            "cond_embeds": cond_embeds,
            "pc_normal": torch.from_numpy(pts),
            "category_id": cat_id,
            "name": model_dir.name,
        }


def collate_v8(batch):
    """Collate function for v8 — pads codes, stacks embeddings."""
    max_len = max(b["codes"].shape[0] for b in batch)
    codes_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        codes_padded[i, :b["codes"].shape[0]] = b["codes"]

    result = {"codes": codes_padded, "category_ids": torch.tensor([b["category_id"] for b in batch])}

    if batch[0]["cond_embeds"] is not None:
        result["cond_embeds"] = torch.stack([b["cond_embeds"] for b in batch]).float()
    else:
        result["pc_normal"] = torch.stack([b["pc_normal"] for b in batch])

    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/BPT-open-8k-8-16.yaml")
    parser.add_argument("--model_path", default="weights/bpt-8-16-500m.pt")
    parser.add_argument("--data_dir", default="../supervised_data")
    parser.add_argument("--cache_dir", default="./cached_embeddings")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_code_len", type=int, default=18000)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--category_mapping", default="category_mapping.json")
    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPU count: {torch.cuda.device_count()}")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"Config: {json.dumps(config, indent=2)}")

    # Load category mapper
    mapper, cat_emb = load_category_conditioner(CATEGORY_DIM, args.category_mapping)
    print(f"Categories: {mapper.num_categories} known + 1 unknown")

    # Model
    model = MeshTransformer(
        dim=config["dim"], attn_depth=config["depth"],
        max_seq_len=config["max_seq_len"], dropout=config["dropout"],
        mode=config["mode"], num_discrete_coors=2**int(config["quant_bit"]),
        block_size=config["block_size"], offset_size=config["offset_size"],
        conditioned_on_pc=config["conditioned_on_pc"],
        use_special_block=config["use_special_block"],
        encoder_name=config["encoder_name"], encoder_freeze=config["encoder_freeze"])

    model_path = Path(args.model_path)
    if model_path.exists():
        print(f"Loading pre-trained weights: {args.model_path}")
        model.load(args.model_path)
    else:
        print(f"WARNING: No pre-trained weights at {args.model_path}")

    model = model.cuda()
    resize_pos_emb(model, args.max_code_len)
    model = model.train()
    cat_emb = cat_emb.cuda()

    n_trainable = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.nelement() for p in model.parameters())
    print(f"Parameters: {n_trainable:,} trainable / {n_total:,} total")

    # Dataset
    dataset = SupervisedDatasetV8(
        args.data_dir, augment=args.augment,
        max_code_len=args.max_code_len, cache_dir=args.cache_dir,
        category_mapper=mapper)
    print(f"Dataset: {len(dataset)} samples")

    # Train/val split
    n_val = len(dataset) // 10
    n_train = len(dataset) - n_val
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    print(f"Split: {n_train} train / {n_val} val")

    train_loader = DataLoader(dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(train_idx), collate_fn=collate_v8,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(val_idx), collate_fn=collate_v8,
        num_workers=args.num_workers, pin_memory=True)

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += [p for p in cat_emb.parameters()]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps

    # LR warmup
    warmup_steps = len(train_loader) // 2
    s1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=args.lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [s1, s2], milestones=[warmup_steps])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            codes = batch["codes"].to(device)
            cat_ids = batch["category_ids"].to(device)
            loss = None

            # Build condition kwargs with category embedding
            cond_kw = {}
            if "cond_embeds" in batch:
                base_cond = batch["cond_embeds"].to(device)
                cat_emb_batch = cat_emb(cat_ids)  # [B, 1, 1024]
                # Concat category embedding to condition features
                cond_kw["cond_embeds"] = torch.cat([cat_emb_batch, base_cond], dim=1)
            else:
                cond_kw["pc"] = batch["pc_normal"].to(device).half()
                # For PC mode, just use category embedding
                cond_kw["cat_embeds"] = cat_emb(cat_ids)

            # Forward
            try:
                with torch.cuda.amp.autocast():
                    loss = model.forward_on_codes(codes, return_loss=True, **cond_kw)
            except AssertionError as e:
                if 'NAN' in str(e) or 'nan' in str(e):
                    continue
                raise

            if loss is None:
                continue

            if torch.isfinite(loss):
                loss = loss / args.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                continue

            if (global_step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            lv = loss.item() * args.grad_accum_steps
            epoch_loss += lv if lv < 1e4 else 0.0
            global_step += 1
            pbar.set_postfix({"loss": f"{lv:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch in val_loader:
                    codes = batch["codes"].to(device)
                    cat_ids = batch["category_ids"].to(device)
                    cond_kw = {}
                    if "cond_embeds" in batch:
                        base_cond = batch["cond_embeds"].to(device)
                        cat_batch = cat_emb(cat_ids)
                        cond_kw["cond_embeds"] = torch.cat([cat_batch, base_cond], dim=1)
                    else:
                        cond_kw["pc"] = batch["pc_normal"].to(device).half()
                    loss = model.forward_on_codes(codes, return_loss=True, **cond_kw)
                    lv = loss.item()
                    val_loss += lv if lv < 1e4 else 0.0

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(str(out_dir / "best_model.pt"))
            torch.save(cat_emb.state_dict(), str(out_dir / "best_cat_emb.pt"))
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

        if (epoch + 1) % args.save_every == 0:
            model.save(str(out_dir / f"checkpoint_epoch{epoch+1}.pt"))

    model.save(str(out_dir / "final_model.pt"))
    torch.save(cat_emb.state_dict(), str(out_dir / "final_cat_emb.pt"))
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    args = get_args()
    train(args)
