"""Monitor v8 training on AutoDL A100. FIXED: never deletes checkpoints."""
import paramiko, json, time
from datetime import datetime

SSH_HOST = 'region-9.autodl.pro'
SSH_PORT = 28256
SSH_USER = 'root'
SSH_PASS = 'Xt45iO3Fgvnt'

def ssh_exec(cmd, timeout=15):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS, timeout=timeout)
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    ssh.close()
    return out

def check():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    vram = ssh_exec('nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader')
    procs = ssh_exec('ps aux | grep train_cloud | grep -v grep | wc -l')
    epochs = ssh_exec('grep "Epoch [0-9]*:" /root/autodl-tmp/train_full.log | tail -5')
    errs = ssh_exec('grep -ci "Error\\|Traceback" /root/autodl-tmp/train_full.log')
    nan = ssh_exec('grep -cP "loss=nan|loss=inf" /root/autodl-tmp/train_full.log')
    ckpts = ssh_exec('ls /root/autodl-tmp/bpt/checkpoints/*.pt 2>/dev/null | wc -l')

    alive = int(procs.strip() or 0) > 0
    has_errs = int(errs.strip() or 0) > 0
    completed = ssh_exec('grep -c "Training complete" /root/autodl-tmp/train_full.log 2>/dev/null')
    is_done = int(completed.strip() or 0) > 0

    print(f"[{now}] Alive:{alive} | Done:{is_done} | VRAM:{vram} | Err:{errs.strip()} | NaN:{nan.strip()} | Ckpt:{ckpts.strip()}")
    if epochs:
        print(f"  {epochs.split(chr(10))[-1] if epochs else 'N/A'}")

    # === THREE RULES ===

    # RULE 1: Training completed normally → report, NEVER restart, NEVER delete
    if is_done:
        best = ssh_exec('grep "Best model\\|Training complete" /root/autodl-tmp/train_full.log | tail -3')
        print(f"  COMPLETED. {best}")
        return True

    # RULE 2: Training running → nothing to do
    if alive:
        return True

    # RULE 3: Process dead + has errors → genuine crash → restart
    if not has_errs:
        print("  Process stopped without errors. Check manually.")
        return False

    last_err = ssh_exec('tail -30 /root/autodl-tmp/train_full.log 2>/dev/null | grep -A2 "Error\\|Traceback" | tail -5')
    print(f"  CRASH: {last_err[:150]}")

    deployed = ssh_exec('ls /root/autodl-tmp/bpt/train_cloud_v8.py 2>/dev/null')
    if not deployed:
        print("  Code not deployed.")
        return False

    # Restart — KEEP existing checkpoints
    print("  Restarting (preserving checkpoints)...")
    restart_cmd = 'cd /root/autodl-tmp/bpt && nohup /root/miniconda3/bin/python -B train_cloud_v8.py --config config/BPT-open-8k-8-16.yaml --model_path weights/bpt-8-16-500m.pt --data_dir ../supervised_data --cache_dir ./cached_embeddings --output_dir ./checkpoints --epochs 30 --batch_size 1 --max_code_len 14000 --grad_accum_steps 8 --lr 1e-5 --save_every 5 --augment > /root/autodl-tmp/train_full.log 2>&1 &'
    ssh_exec(restart_cmd, timeout=10)
    print("  Restarted.")
    time.sleep(60)
    vram = ssh_exec('nvidia-smi --query-gpu=memory.used --format=csv,noheader')
    print(f"  VRAM: {vram}")
    return True

if __name__ == "__main__":
    check()
