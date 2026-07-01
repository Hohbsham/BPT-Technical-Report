"""
AutoResearch Pipeline: Train → Infer → Eval → Feedback → Retrain
Orchestrates the full BPT garment regularizer training loop.

Usage:
  # Full auto loop
  python autoresearch_pipeline.py --mode auto --data_dir ../supervised_data --cache_dir ./cached_embeddings

  # Single iteration (train once, eval, report)
  python autoresearch_pipeline.py --mode single --data_dir ../supervised_data

  # Eval only (on existing checkpoint)
  python autoresearch_pipeline.py --mode eval --model_path checkpoints/best_model.pt
"""
import os, sys, json, time, subprocess, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

HISTORY_FILE = "autoresearch_history.json"
BEST_CONFIG_FILE = "autoresearch_best.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def load_best_config():
    if os.path.exists(BEST_CONFIG_FILE):
        with open(BEST_CONFIG_FILE) as f:
            return json.load(f)
    return {
        "lr": 5e-5,
        "max_code_len": 2000,
        "epochs": 50,
        "batch_size": 1,
        "grad_accum_steps": 8,
        "augment": True,
        "best_val_loss": float('inf'),
        "best_chamfer": float('inf'),
    }


def save_best_config(config):
    with open(BEST_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def run_training(config, experiment_id):
    """Run train_cloud.py with given config, return val_loss."""
    print(f"\n{'='*60}")
    print(f"Experiment {experiment_id}: Training...")
    print(f"Config: lr={config['lr']}, max_code_len={config['max_code_len']}, "
          f"epochs={config['epochs']}, grad_accum={config['grad_accum_steps']}")
    print(f"{'='*60}")

    augment_flag = "--augment" if config.get("augment", True) else "--no-augment"

    cmd = [
        sys.executable, "train_cloud.py",
        "--config", "config/BPT-open-8k-8-16.yaml",
        "--model_path", "weights/bpt-8-16-500m.pt",
        "--data_dir", config["data_dir"],
        "--cache_dir", config.get("cache_dir", "./cached_embeddings"),
        "--output_dir", f"checkpoints/exp_{experiment_id}",
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--max_code_len", str(config["max_code_len"]),
        "--grad_accum_steps", str(config["grad_accum_steps"]),
        "--lr", str(config["lr"]),
        "--save_every", str(config.get("save_every", 5)),
        augment_flag,
    ]

    train_log = f"train_exp_{experiment_id}.log"
    print(f"Log: {train_log}")
    print(f"Cmd: {' '.join(cmd)}")

    with open(train_log, 'w') as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)

    # Parse val_loss from log
    val_loss = float('inf')
    with open(train_log) as f:
        for line in f:
            if "val_loss" in line:
                try:
                    val_loss = float(line.split("val_loss=")[1].split(",")[0])
                except:
                    pass

    return val_loss


def run_eval(model_path, test_dirs):
    """Run inference + evaluation, return metrics."""
    results = {}
    for test_name, test_dir in test_dirs.items():
        out_dir = f"eval_results/{test_name}"
        cmd = [
            sys.executable, "infer_and_eval.py",
            "--model_path", model_path,
            "--test_dir", test_dir,
            "--output_dir", out_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)

        # Parse metrics
        metrics_path = Path(out_dir) / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                results[test_name] = json.load(f)

    return results


def decide_next_config(current_config, val_loss, eval_results, history):
    """AutoResearch decision logic: adjust config based on results."""
    next_config = current_config.copy()
    best = load_best_config()

    action = "CONTINUE"
    reasons = []

    # Check if this is the best run
    if val_loss < best["best_val_loss"]:
        best["best_val_loss"] = val_loss
        action = "SAVE_BEST"
        reasons.append(f"New best val_loss: {val_loss:.4f}")

    # Check mesh quality from eval
    clean_metrics = eval_results.get("clean", {})
    if clean_metrics and "average" in clean_metrics:
        chamfer = clean_metrics["average"]["chamfer_pct"]
        if chamfer < best["best_chamfer"]:
            best["best_chamfer"] = chamfer
            reasons.append(f"New best Chamfer: {chamfer:.3f}%")

        # Quality-based adjustments
        if chamfer > 10 and current_config["max_code_len"] < 8000:
            next_config["max_code_len"] = min(current_config["max_code_len"] + 1000, 8000)
            next_config["batch_size"] = 1  # compensate
            next_config["grad_accum_steps"] = 16
            action = "RETRAIN"
            reasons.append("High Chamfer → increasing max_code_len")
        elif chamfer < 2 and current_config["lr"] > 1e-5:
            next_config["lr"] = current_config["lr"] * 0.7
            next_config["epochs"] = min(current_config["epochs"] + 10, 100)
            action = "RETRAIN"
            reasons.append("Good Chamfer → fine-tuning with lower lr")

    # Loss plateau check
    recent_losses = [h["val_loss"] for h in history[-5:]] if len(history) >= 5 else []
    if len(recent_losses) >= 5 and max(recent_losses) - min(recent_losses) < 0.01:
        if current_config["lr"] > 1e-6:
            next_config["lr"] = current_config["lr"] * 0.5
            action = "RETRAIN"
            reasons.append("Loss plateau → reducing lr")

    # OOM handling
    last_error = history[-1].get("error") if history else None
    if last_error and "OOM" in str(last_error):
        next_config["max_code_len"] = max(current_config["max_code_len"] - 1000, 1000)
        next_config["grad_accum_steps"] = current_config.get("grad_accum_steps", 8) * 2
        action = "RETRAIN"
        reasons.append("OOM → reducing max_code_len")

    save_best_config(best)
    return next_config, action, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="single",
                        choices=["auto", "single", "eval"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_dir", default="../supervised_data")
    parser.add_argument("--cache_dir", default="./cached_embeddings")
    parser.add_argument("--max_iterations", type=int, default=50,
                        help="Max AutoResearch iterations")
    args = parser.parse_args()

    if args.mode == "eval":
        model_path = args.model_path or "checkpoints/best_model.pt"
        test_dirs = {
            "clean": "../supervised_data_clean",
        }
        results = run_eval(model_path, test_dirs)
        print(json.dumps(results, indent=2))
        return

    # Load best config
    config = load_best_config()
    config["data_dir"] = args.data_dir
    config["cache_dir"] = args.cache_dir

    history = load_history()

    for iteration in range(1, args.max_iterations + 1):
        timestamp = datetime.now().isoformat()
        experiment_id = f"{timestamp[:10]}_{iteration:03d}"

        print(f"\n{'#'*60}")
        print(f"# AutoResearch Iteration {iteration}/{args.max_iterations}")
        print(f"{'#'*60}")

        entry = {
            "iteration": iteration,
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "config": config.copy(),
        }

        # Step 1: Train
        try:
            val_loss = run_training(config, experiment_id)
            entry["val_loss"] = val_loss
            print(f"\nTraining complete. val_loss = {val_loss:.4f}")
        except Exception as e:
            entry["error"] = str(e)
            print(f"Training failed: {e}")
            history.append(entry)
            save_history(history)

            # Try to recover
            config, _, reasons = decide_next_config(config, float('inf'), {}, history)
            for r in reasons:
                print(f"Recovery: {r}")
            continue

        # Step 2: Inference + Evaluation
        model_path = f"checkpoints/exp_{experiment_id}/best_model.pt"
        if not os.path.exists(model_path):
            model_path = f"checkpoints/exp_{experiment_id}/final_model.pt"

        if os.path.exists(model_path):
            test_dirs = {"clean": "../supervised_data_clean"}
            eval_results = run_eval(model_path, test_dirs)
            entry["eval_results"] = eval_results
        else:
            eval_results = {}
            print("No checkpoint found, skipping eval")

        # Step 3: Decide next action
        next_config, action, reasons = decide_next_config(
            config, val_loss, eval_results, history
        )

        entry["action"] = action
        entry["reasons"] = reasons
        history.append(entry)
        save_history(history)

        print(f"\nDecision: {action}")
        for r in reasons:
            print(f"  → {r}")
        print(f"Next config: {next_config}")

        if args.mode == "single":
            break  # Only one iteration in single mode

        if action == "STOP":
            print("\nAutoResearch converged. Best config found.")
            break

        # Prepare for next iteration
        config = next_config
        for k in ["data_dir", "cache_dir"]:
            config[k] = getattr(args, k)

    print(f"\n{'='*60}")
    print(f"AutoResearch complete. {len(history)} experiments run.")
    best = load_best_config()
    print(f"Best val_loss: {best['best_val_loss']:.4f}")
    print(f"Best Chamfer: {best['best_chamfer']:.3f}%")
    print(f"Best config: {json.dumps(best, indent=2)}")


if __name__ == "__main__":
    main()
