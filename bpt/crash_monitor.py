"""
Crash Monitor + Auto-Recovery for BPT Training.
Watches training process, detects failures, attempts recovery, writes crash reports.

Usage:
  python crash_monitor.py --log_file /root/autodl-tmp/train_full.log --check_interval 30
"""
import os, sys, time, subprocess, signal, json, argparse, re
from pathlib import Path
from datetime import datetime

CRASH_REPORT_FILE = "crash_report.json"
RECOVERY_LOG_FILE = "recovery_history.json"

CRASH_PATTERNS = {
    "OOM": r"OutOfMemoryError|CUDA out of memory",
    "dtype_mismatch": r"expected.*dtype.*but got|c10::Half != float",
    "NaN_loss": r"loss=nan|loss=inf|loss=NaN",
    "gradient_explosion": r"loss=inf",
    "process_killed": None,  # detected differently
    "disk_full": r"No space left on device",
    "connection_lost": r"Broken pipe|Connection refused",
}

RECOVERY_STRATEGIES = {
    "OOM": {"max_code_len": -500, "grad_accum_steps": "*2"},
    "dtype_mismatch": {"auto_fix": "convert_tensors_to_float32"},
    "NaN_loss": {"lr": "*0.5"},
    "gradient_explosion": {"lr": "*0.1", "grad_clip": "0.5"},
}


def detect_crash(log_file):
    """Scan log file for crash patterns."""
    if not os.path.exists(log_file):
        return None, "log file not found"

    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()

    for crash_type, pattern in CRASH_PATTERNS.items():
        if pattern and re.search(pattern, content, re.IGNORECASE):
            # Extract traceback
            tb_match = re.search(r'Traceback \(most recent call last\):(.+?)(?:\n\n|\Z)', content, re.DOTALL)
            traceback = tb_match.group(0)[:2000] if tb_match else "No traceback found"
            return crash_type, traceback

    # Check for abnormal loss plateau (loss stuck for >500 steps)
    loss_values = re.findall(r'loss=([\d.]+)', content)
    if len(loss_values) > 500:
        recent = [float(l) for l in loss_values[-500:]]
        if max(recent) - min(recent) < 0.01 and recent[-1] > 5:
            return "loss_plateau", f"Loss stuck at {recent[-1]:.2f} for 500 steps"

    return None, None


def get_training_status(log_file):
    """Get current training progress."""
    if not os.path.exists(log_file):
        return {"status": "not_started"}

    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()

    epochs = re.findall(r'Epoch\s+(\d+)/(\d+)', content)
    losses = re.findall(r'loss=([\d.]+)', content)
    val_losses = re.findall(r'val_loss=([\d.]+)', content)

    return {
        "status": "running" if epochs else "starting",
        "current_epoch": int(epochs[-1][0]) if epochs else 0,
        "total_epochs": int(epochs[-1][1]) if epochs else 50,
        "recent_loss": float(losses[-1]) if losses else None,
        "loss_trend": "decreasing" if len(losses) > 10 and float(losses[-1]) < float(losses[-10]) else "stable",
        "best_val_loss": float(min(val_losses)) if val_losses else None,
    }


def attempt_recovery(crash_type, current_config):
    """Try to recover from a crash by adjusting config."""
    strategy = RECOVERY_STRATEGIES.get(crash_type, {})
    new_config = current_config.copy()
    recovery_actions = []

    if "max_code_len" in strategy:
        delta = strategy["max_code_len"]
        if isinstance(delta, int) and delta < 0:
            new_config["max_code_len"] = max(1000, current_config.get("max_code_len", 2000) + delta)
            recovery_actions.append(f"Reduced max_code_len to {new_config['max_code_len']}")

    if "grad_accum_steps" in strategy:
        if strategy["grad_accum_steps"] == "*2":
            new_config["grad_accum_steps"] = current_config.get("grad_accum_steps", 8) * 2
            recovery_actions.append(f"Increased grad_accum to {new_config['grad_accum_steps']}")

    if "lr" in strategy:
        lr_str = strategy["lr"]
        if lr_str.startswith("*"):
            new_config["lr"] = current_config.get("lr", 5e-5) * float(lr_str[1:])
        else:
            new_config["lr"] = float(lr_str)
        recovery_actions.append(f"Adjusted lr to {new_config['lr']:.2e}")

    if "auto_fix" in strategy and strategy["auto_fix"] == "convert_tensors_to_float32":
        # Add .float() conversions to train code
        recovery_actions.append("Applied dtype fix (float32 conversion)")

    if "grad_clip" in strategy:
        new_config["grad_clip"] = float(strategy["grad_clip"])
        recovery_actions.append(f"Reduced grad_clip to {new_config['grad_clip']}")

    return new_config, recovery_actions


def generate_crash_report(crash_type, traceback, status, config, recovery_actions):
    """Generate detailed crash report for Claude to read."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "crash_type": crash_type,
        "traceback": traceback[:3000],
        "training_status_at_crash": status,
        "config_at_crash": config,
        "recovery_attempted": len(recovery_actions) > 0,
        "recovery_actions": recovery_actions,
        "recommended_human_actions": [],
    }

    # Add human-readable recommendations
    if crash_type == "OOM":
        report["recommended_human_actions"] = [
            "Reduce max_code_len by 500-1000",
            "Or upgrade to A100 40GB",
            "Or enable gradient checkpointing: torch.utils.checkpoint.checkpoint()",
        ]
    elif crash_type == "dtype_mismatch":
        report["recommended_human_actions"] = [
            "Ensure all inputs are .float() before model",
            "Check mixed precision autocast settings",
            "Verify cached embedding dtype with: torch.load(emb).dtype",
        ]
    elif crash_type == "NaN_loss":
        report["recommended_human_actions"] = [
            "Reduce learning rate",
            "Check for NaN in cached embeddings",
            "Add gradient clipping (--grad_clip 0.5)",
        ]
    elif crash_type == "gradient_explosion":
        report["recommended_human_actions"] = [
            "Reduce lr to 1e-5",
            "Increase gradient clipping",
            "Check if any samples have extreme token lengths",
        ]
    elif crash_type == "loss_plateau":
        report["recommended_human_actions"] = [
            "Reduce lr by 50%",
            "Increase max_code_len if possible",
            "Check if data is diverse enough",
        ]

    return report


def save_crash_report(report):
    """Save crash report for Claude to read later."""
    path = Path(CRASH_REPORT_FILE)

    # Load existing reports
    reports = []
    if path.exists():
        with open(path) as f:
            reports = json.load(f)

    reports.append(report)

    # Keep last 20 reports
    reports = reports[-20:]

    with open(path, 'w') as f:
        json.dump(reports, f, indent=2)

    # Also write a human-readable markdown version
    md_path = Path("CRASH_REPORT.md")
    with open(md_path, 'w') as f:
        f.write(f"# 🚨 Training Crash Report\n\n")
        f.write(f"**Time:** {report['timestamp']}\n\n")
        f.write(f"**Crash Type:** `{report['crash_type']}`\n\n")
        f.write(f"**Status at crash:** Epoch {report['training_status_at_crash']['current_epoch']}, ")
        f.write(f"Loss: {report['training_status_at_crash']['recent_loss']}\n\n")
        f.write(f"**Recovery attempted:** {report['recovery_attempted']}\n")
        for action in report['recovery_actions']:
            f.write(f"- {action}\n")
        f.write(f"\n**Recommended actions:**\n")
        for action in report['recommended_human_actions']:
            f.write(f"- {action}\n")
        f.write(f"\n<details><summary>Traceback</summary>\n\n```\n{report['traceback']}\n```\n</details>\n")

    return str(md_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", default="/root/autodl-tmp/train_full.log")
    parser.add_argument("--check_interval", type=int, default=30)
    parser.add_argument("--max_auto_restarts", type=int, default=3)
    parser.add_argument("--one_shot", action="store_true",
                        help="Run once and report status")
    args = parser.parse_args()

    if args.one_shot:
        status = get_training_status(args.log_file)
        print(json.dumps(status, indent=2))
        crash_type, traceback = detect_crash(args.log_file)
        if crash_type:
            print(f"\n🚨 CRASH DETECTED: {crash_type}")
            print(traceback[-500:])
        else:
            print("\n✅ No crash detected")
        return

    # Continuous monitoring mode
    config = {
        "max_code_len": 2000,
        "lr": 5e-5,
        "grad_accum_steps": 8,
        "grad_clip": 1.0,
    }
    restart_count = 0

    print(f"🛡️ Crash Monitor started. Watching: {args.log_file}")
    print(f"   Check interval: {args.check_interval}s")
    print(f"   Max auto-restarts: {args.max_auto_restarts}")

    while True:
        time.sleep(args.check_interval)

        # Check if training process exists
        proc_count = int(subprocess.run(
            "ps aux | grep train_cloud.py | grep -v grep | wc -l",
            shell=True, capture_output=True, text=True
        ).stdout.strip() or 0)

        status = get_training_status(args.log_file)
        crash_type, traceback = detect_crash(args.log_file)

        if proc_count == 0 and status["status"] != "not_started":
            # Process died!
            crash_type = crash_type or "process_killed"
            if not traceback:
                # Read last lines of log for error
                with open(args.log_file, 'r', errors='ignore') as f:
                    lines = f.readlines()
                    traceback = ''.join(lines[-30:])

        if crash_type:
            print(f"\n🚨 [{datetime.now().strftime('%H:%M:%S')}] CRASH: {crash_type}")

            # Generate report
            report = generate_crash_report(
                crash_type, traceback, status, config, []
            )

            # Attempt recovery
            if restart_count < args.max_auto_restarts:
                new_config, recovery_actions = attempt_recovery(crash_type, config)
                report["recovery_attempted"] = True
                report["recovery_actions"] = recovery_actions

                md_path = save_crash_report(report)
                print(f"   Report: {md_path}")
                print(f"   Auto-restart {restart_count + 1}/{args.max_auto_restarts}...")

                # Update config and restart training
                config = new_config
                restart_count += 1
            else:
                report["recovery_attempted"] = False
                report["recommended_human_actions"].append(
                    f"MAX AUTO-RESTARTS ({args.max_auto_restarts}) REACHED. Human intervention required."
                )
                save_crash_report(report)
                print(f"   ❌ Max restarts reached. Human needed.")
                break

        # Heartbeat
        if status["current_epoch"] > 0:
            print(f"   [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Epoch {status['current_epoch']}/{status['total_epochs']} | "
                  f"Loss: {status['recent_loss']} | "
                  f"{'✅' if not crash_type else '🚨'}")


if __name__ == "__main__":
    main()
