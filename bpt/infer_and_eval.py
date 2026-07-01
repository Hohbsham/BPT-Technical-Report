"""
Inference + Evaluation for BPT Garment Regularizer.
Runs trained model on test samples, generates meshes, evaluates quality.
Part of the AutoResearch feedback loop.

Usage:
  python infer_and_eval.py --model_path checkpoints/best_model.pt --test_dir supervised_data_clean --output_dir eval_results
"""
import os, sys, json, argparse
from pathlib import Path
import torch
import yaml
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parent))
from model.model import MeshTransformer
from model.serializaiton import BPT_deserialize
from utils import apply_normalize


def load_model(model_path, config_path="config/BPT-open-8k-8-16.yaml"):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = MeshTransformer(
        dim=config['dim'], attn_depth=config['depth'],
        max_seq_len=config['max_seq_len'], dropout=config['dropout'],
        mode=config['mode'], num_discrete_coors=2**int(config['quant_bit']),
        block_size=config['block_size'], offset_size=config['offset_size'],
        conditioned_on_pc=config['conditioned_on_pc'],
        use_special_block=config['use_special_block'],
        encoder_name=config['encoder_name'],
        encoder_freeze=config['encoder_freeze'],
    )
    model.load(model_path)
    model = model.eval().half().cuda()
    return model


def load_pointcloud(pc_path):
    pc = trimesh.load(str(pc_path))
    pts = np.array(pc.vertices, dtype=np.float32)
    # Estimate normals
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=min(30, len(pts) - 1) + 1)
    normals = np.zeros_like(pts)
    for i, nbrs in enumerate(nn_idx):
        nbr_pts = pts[nbrs[1:]]
        cov = np.cov(nbr_pts.T, bias=True)
        _, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)

    pc_data = np.concatenate([pts, normals], axis=-1).astype(np.float16)
    pc_data = np.nan_to_num(pc_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Resample to 4096
    n = pc_data.shape[0]
    if n >= 4096:
        ind = np.random.choice(n, 4096, replace=False)
    else:
        ind = np.random.choice(n, 4096, replace=True)
    return torch.from_numpy(pc_data[ind]).unsqueeze(0).cuda().half()


def evaluate_mesh(pred_mesh, gt_mesh):
    """Compute Chamfer distance, Hausdorff, face coverage, quad ratio."""
    # Sample points from both meshes
    n_samples = 10000
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, n_samples)
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, n_samples)

    # Chamfer distance
    from scipy.spatial import cKDTree
    tree_pred = cKDTree(pred_pts)
    tree_gt = cKDTree(gt_pts)

    dist_pred_to_gt, _ = tree_gt.query(pred_pts, k=1)
    dist_gt_to_pred, _ = tree_pred.query(gt_pts, k=1)

    chamfer = (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) / 2

    # Hausdorff
    hausdorff = max(dist_pred_to_gt.max(), dist_gt_to_pred.max())

    # As % of bounding box diagonal
    bbox_size = np.linalg.norm(gt_mesh.bounds[1] - gt_mesh.bounds[0])
    chamfer_pct = 100 * chamfer / bbox_size
    hausdorff_pct = 100 * hausdorff / bbox_size

    # Face coverage (rough)
    face_coverage = 100 * len(pred_mesh.faces) / max(1, len(gt_mesh.faces))

    # Quad ratio
    if len(pred_mesh.faces) > 0:
        quad_count = sum(1 for f in pred_mesh.faces if len(f) == 4)
        quad_ratio = 100 * quad_count / len(pred_mesh.faces)
    else:
        quad_ratio = 0

    return {
        "chamfer_pct": round(chamfer_pct, 3),
        "hausdorff_pct": round(hausdorff_pct, 3),
        "face_coverage": round(face_coverage, 1),
        "quad_ratio": round(quad_ratio, 1),
        "n_faces": len(pred_mesh.faces),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", default="../supervised_data_clean")
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--config", default="config/BPT-open-8k-8-16.yaml")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--mesh_type", default="quad", choices=["triangle", "quad"])
    args = parser.parse_args()

    model = load_model(args.model_path, args.config)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Find test samples
    samples = sorted([d for d in test_dir.iterdir() if d.is_dir() and d.name != "Cube"])
    if not samples:
        print(f"No samples found in {test_dir}")
        return

    print(f"Evaluating on {len(samples)} test samples...")
    print(f"Model: {args.model_path}")
    print(f"Temperature: {args.temperature}")
    print(f"Mesh type: {args.mesh_type}")
    print()

    all_metrics = []

    for i, sample_dir in enumerate(samples):
        name = sample_dir.name
        pc_path = sample_dir / "input_pointcloud.ply"
        gt_path = sample_dir / "ground_truth.obj"

        if not pc_path.exists() or not gt_path.exists():
            print(f"[{i+1}/{len(samples)}] {name}: SKIP (missing files)")
            continue

        print(f"[{i+1}/{len(samples)}] {name}...", end=" ", flush=True)

        # Load input
        pc = load_pointcloud(pc_path)
        gt_mesh = trimesh.load(str(gt_path), force='mesh', process=False)

        # Generate
        with torch.no_grad():
            codes = model.generate(pc=pc, temperature=args.temperature, max_seq_len=8000)

        # Decode
        pred_mesh = BPT_deserialize(codes[0].cpu().numpy(), mesh_type=args.mesh_type)

        # Save
        pred_path = out_dir / f"{name}_pred_{args.mesh_type}.obj"
        pred_mesh.export(str(pred_path))

        # Evaluate
        metrics = evaluate_mesh(pred_mesh, gt_mesh)
        metrics["name"] = name
        all_metrics.append(metrics)

        print(f"Chamfer={metrics['chamfer_pct']}% | Hausdorff={metrics['hausdorff_pct']}% | "
              f"Faces={metrics['n_faces']} ({metrics['face_coverage']:.0f}% cov) | "
              f"Quad={metrics['quad_ratio']:.0f}%")

    # Summary
    if all_metrics:
        avg = {
            "chamfer_pct": round(np.mean([m["chamfer_pct"] for m in all_metrics]), 3),
            "hausdorff_pct": round(np.mean([m["hausdorff_pct"] for m in all_metrics]), 3),
            "face_coverage": round(np.mean([m["face_coverage"] for m in all_metrics]), 1),
            "quad_ratio": round(np.mean([m["quad_ratio"] for m in all_metrics]), 1),
        }

        print(f"\n=== AVERAGE ===")
        print(f"Chamfer: {avg['chamfer_pct']}% | Hausdorff: {avg['hausdorff_pct']}% | "
              f"Coverage: {avg['face_coverage']}% | Quad: {avg['quad_ratio']}%")

        # Determine quality level
        if avg['chamfer_pct'] < 2 and avg['face_coverage'] > 80:
            quality = "EXCELLENT"
        elif avg['chamfer_pct'] < 5 and avg['face_coverage'] > 50:
            quality = "GOOD"
        elif avg['chamfer_pct'] < 10:
            quality = "FAIR"
        else:
            quality = "POOR"

        result = {
            "model_path": args.model_path,
            "test_dir": str(test_dir),
            "temperature": args.temperature,
            "mesh_type": args.mesh_type,
            "per_sample": all_metrics,
            "average": avg,
            "quality": quality,
        }

        result_path = out_dir / "metrics.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\nQuality: {quality}")
        print(f"Results saved to {result_path}")

        return result


if __name__ == "__main__":
    main()
