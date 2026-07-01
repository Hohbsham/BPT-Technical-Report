"""
Pre-compute Michelangelo encoder embeddings for all samples (cloud version).
No Windows patches needed — runs natively on Linux.

Usage:
  python precompute_cloud.py --data_dir ./supervised_data --output_dir ./cached_embeddings
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import trimesh
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model.miche_conditioner import PointConditioner
from utils import apply_normalize


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


def precompute(args):
    device = torch.device("cuda")
    print(f"Device: {device}")

    encoder = PointConditioner(model_name='miche-256-feature', freeze=True)
    encoder = encoder.to(device)
    encoder.eval()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name == "Cube":
            continue
        gt = d / "ground_truth.obj"
        pc = d / "input_pointcloud.ply"
        if gt.exists() and pc.exists():
            emb_file = output_dir / f"{d.name}.pt"
            if not args.overwrite and emb_file.exists():
                continue
            model_dirs.append(d)

    print(f"Samples to process: {len(model_dirs)}")

    for model_dir in tqdm(model_dirs, desc="Computing embeddings"):
        gt_mesh = trimesh.load(str(model_dir / "ground_truth.obj"), force='mesh', process=False)
        bbox = gt_mesh.bounds
        center = (bbox[1] + bbox[0]) / 2
        scale = (bbox[1] - bbox[0]).max() / (2 * 0.95)

        pc_raw = trimesh.load(str(model_dir / "input_pointcloud.ply"))
        pts = np.array(pc_raw.vertices, dtype=np.float32)
        pts = (pts - center) / scale

        normals = _estimate_normals(pts, k=min(30, len(pts) - 1))
        pc_data = np.concatenate([pts, normals], axis=-1).astype(np.float16)
        pc_data = np.nan_to_num(pc_data, nan=0.0, posinf=0.0, neginf=0.0)

        n = pc_data.shape[0]
        pc_samples = min(n, 4096)
        ind = np.random.choice(n, pc_samples, replace=False)
        pc_data = pc_data[ind]

        if pc_samples < 4096:
            pad = np.zeros((4096 - pc_samples, 6), dtype=np.float16)
            pc_data = np.concatenate([pc_data, pad], axis=0)

        with torch.no_grad():
            pc_tensor = torch.from_numpy(pc_data).unsqueeze(0).to(device)
            embedding = encoder.embed_pc(pc_tensor)

        if torch.isnan(embedding).any():
            print(f"  [WARN] NaN in embedding for {model_dir.name}, skipping")
            continue

        emb_file = output_dir / f"{model_dir.name}.pt"
        torch.save(embedding.cpu().half(), emb_file)

    print(f"Done. Embeddings saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute Michelangelo encoder embeddings (Cloud)")
    parser.add_argument("--data_dir", type=str, default="./supervised_data")
    parser.add_argument("--output_dir", type=str, default="./cached_embeddings")
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()
    precompute(args)
