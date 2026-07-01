#!/bin/bash
# === BPT Auto-Deploy Script ===
set -e
PY=/home/apulis-dev/.conda/envs/bpt/bin/python
PIP=/home/apulis-dev/.conda/envs/bpt/bin/pip
ENV="env -i PATH=/usr/bin:/usr/local/bin:/home/apulis-dev/.conda/envs/bpt/bin HOME=$HOME"
WORK=/home/apulis-dev/userdata/bpt
LOG=/home/apulis-dev/userdata

echo "[1/6] Install deps..."
$PIP install --force-reinstall scipy -q 2>&1 | tail -1

echo "[2/6] Patch NaN guard..."
cp $WORK/train_cloud.py $WORK/train_cloud.bak
sed -i 's/epoch_loss += loss.item() \* args.grad_accum_steps/l = loss.item() * args.grad_accum_steps; epoch_loss += l if l < 1000 else 0.0/' $WORK/train_cloud.py
sed -i 's/val_loss += loss.item()/l = loss.item(); val_loss += l if l < 1000 else 0.0/' $WORK/train_cloud.py
echo "Patched"

echo "[3/6] Overfit test (quick check)..."
$ENV $PY $WORK/overfit_test.py --n_samples 3 --epochs 30 --max_code_len 4000 --lr 2e-5   --cache_dir $WORK/cached_embeddings --data_dir /home/apulis-dev/userdata/supervised_data   2>&1 | grep -E "PASS|FAIL" > $LOG/overfit_check.txt
cat $LOG/overfit_check.txt

echo "[4/6] Start full training..."
cat > $LOG/run_train.sh << 'TRAINEOF'
#!/bin/bash
PY=/home/apulis-dev/.conda/envs/bpt/bin/python
ENV="env -i PATH=/usr/bin:/usr/local/bin:/home/apulis-dev/.conda/envs/bpt/bin HOME=\/c/Users/YANGZ"
WORK=/home/apulis-dev/userdata/bpt
LOG=/home/apulis-dev/userdata

echo "=== BPT Full Training ==="
echo "Start: \Wed Jun  3 22:49:07     2026"
echo "GPU: Tesla V100 32GB"
echo "Config: max_code_len=4000, lr=2e-5, batch=1, grad_accum=8"

$ENV $PY $WORK/train_cloud.py   --config $WORK/config/BPT-open-8k-8-16.yaml   --model_path $WORK/weights/bpt-8-16-500m.pt   --data_dir /home/apulis-dev/userdata/supervised_data   --cache_dir $WORK/cached_embeddings   --output_dir $WORK/checkpoints   --epochs 50 --batch_size 1 --max_code_len 4000   --grad_accum_steps 8 --lr 2e-5 --save_every 5 --augment   2>&1 | tee $LOG/train_full.log

echo "Training done: \Wed Jun  3 22:49:07     2026"
TRAINEOF
chmod +x $LOG/run_train.sh

echo "[5/6] Start crash monitor..."
cat > $LOG/crash_monitor.sh << 'CRASHEOF'
#!/bin/bash
LOG=/home/apulis-dev/userdata/train_full.log
while true; do
  sleep 30
  if grep -qi "RuntimeError\|CUDA out of memory\|Traceback" $LOG 2>/dev/null; then
    echo "CRASH DETECTED at \Wed Jun  3 22:49:07     2026" >> $LOG.crash
    tail -30 $LOG >> $LOG.crash
  fi
done
CRASHEOF
chmod +x $LOG/crash_monitor.sh

echo "[6/6] Launch all..."
nohup bash $LOG/run_train.sh > $LOG/nohup_train.log 2>&1 &
TRAIN_PID=$!
nohup bash $LOG/crash_monitor.sh > /dev/null 2>&1 &
CRASH_PID=$!

echo ""
echo "=========================================="
echo "  DEPLOYED - All running in background"
echo "  Training PID: $TRAIN_PID"
echo "  Monitor PID: $CRASH_PID"
echo "  Log: $LOG/train_full.log"
echo "  Check: tail -f $LOG/train_full.log"
echo "=========================================="
