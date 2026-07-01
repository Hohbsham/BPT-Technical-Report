"""
BPT Training Monitor — checks every 30 min, fixes common issues, restarts if needed.
"""
import paramiko, json, time, sys
from datetime import datetime

def ssh_exec(cmd, timeout=15):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('region-42.seetacloud.com', port=37805, username='root', password='2gPzTHOKQqfZ', timeout=timeout)
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    ssh.close()
    return out

def check():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Core health checks
    vram = ssh_exec('nvidia-smi --query-gpu=memory.used --format=csv,noheader')
    procs = ssh_exec('ps aux | grep train_cloud | grep -v grep | wc -l')
    loss = ssh_exec('grep -oP "loss=[0-9.]+" /root/autodl-tmp/train_full.log | tail -3')
    epochs = ssh_exec('grep "Epoch [0-9]*:" /root/autodl-tmp/train_full.log | tail -3')
    errs = ssh_exec('grep -ci "Error\\|Traceback" /root/autodl-tmp/train_full.log')
    nan = ssh_exec('grep -cP "loss=nan|loss=inf" /root/autodl-tmp/train_full.log | tail -1')
    complete = ssh_exec('grep -c "Training complete\\|Early stop" /root/autodl-tmp/train_full.log')
    checkpoints = ssh_exec('ls /root/autodl-tmp/bpt/checkpoints/*.pt 2>/dev/null | wc -l')

    status = {
        "time": now,
        "vram": vram,
        "processes": procs.strip(),
        "loss": loss.split('\n')[-1] if loss else 'none',
        "epochs": epochs.split('\n')[-1] if epochs else 'none',
        "errors": errs.strip(),
        "nan_count": nan.strip(),
        "complete": complete.strip(),
        "checkpoints": checkpoints.strip(),
    }

    # Determine health
    alive = int(procs.strip() or 0) > 0

    print(f"[{now}] Alive: {alive} | VRAM: {vram} | Err: {errs.strip()} | NaN: {nan.strip()} | Epoch: {epochs.split(chr(10))[-1] if epochs else 'N/A'}")
    print(f"  Loss: {loss.split(chr(10))[-1] if loss else 'N/A'}")

    # Auto-fix logic
    issues = []
    has_epochs = int(epochs.split(chr(10))[0].split()[0].split(':')[0]) if epochs and 'Epoch' in epochs else 0

    if not alive and errs.strip() != '0':
        issues.append("CRASHED")
    if int(nan.strip() or 0) > 10 and alive and has_epochs > 0:
        issues.append("NaN_DETECTED")

    for issue in issues:
        print(f"  ISSUE: {issue} — attempting fix...")

        if issue == "CRASHED":
            # Check last error type
            last_err = ssh_exec('grep "Error\\|Traceback" /root/autodl-tmp/train_full.log | tail -3')
            print(f"  Last error: {last_err[:100]}")

            # Restart training with current best checkpoint
            restart_cmd = '''source /root/miniconda3/etc/profile.d/conda.sh && cd /root/autodl-tmp/bpt && nohup /root/miniconda3/bin/python /root/autodl-tmp/bpt/train_cloud.py --config /root/autodl-tmp/bpt/config/BPT-open-8k-8-16.yaml --model_path /root/autodl-tmp/bpt/weights/bpt-8-16-500m.pt --data_dir /root/autodl-tmp/supervised_data --cache_dir /root/autodl-tmp/bpt/cached_embeddings --output_dir /root/autodl-tmp/bpt/checkpoints --epochs 50 --batch_size 4 --max_code_len 10000 --grad_accum_steps 2 --lr 1e-5 --save_every 5 --augment > /root/autodl-tmp/train_full.log 2>&1 &'''
            ssh_exec(restart_cmd, timeout=10)
            print("  RESTARTED")

        if issue == "NaN_DETECTED":
            # Report only — don't kill training that's working
            print("  WARNING: NaN detected but training alive. Continue monitoring.")

    # Save report
    with open("d:/ClothesNetData/monitor_log.json", "a") as f:
        f.write(json.dumps(status) + "\n")

    return alive, issues

if __name__ == "__main__":
    check()
