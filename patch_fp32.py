"""Convert train_cloud.py to pure fp32 training."""
with open("/root/autodl-tmp/bpt/train_cloud.py", "r") as f:
    code = f.read()

# 1. Embedding loading: fp16 on disk -> fp32 in memory
code = code.replace('emb.squeeze(0)', 'emb.squeeze(0).float()')

# 2. Model to fp32
code = code.replace('model = model.cuda()', 'model = model.cuda().float()')

# 3. Input to fp32 (not half)
code = code.replace('.to(device).half()', '.to(device).float()')

# 4. Disable autocast in training (replace the with-block comment)
code = code.replace(
    'with torch.cuda.amp.autocast():',
    '#with torch.cuda.amp.autocast():  # fp32 training')

# 5. Disable autocast in validation
code = code.replace(
    'with torch.no_grad(), torch.cuda.amp.autocast():',
    'with torch.no_grad():  # fp32 val')

# 6. Disable scaler
code = code.replace(
    'scaler = torch.cuda.amp.GradScaler()',
    'scaler = None  # fp32')

# 7. scaler.scale -> direct backward
code = code.replace('scaler.scale(loss).backward()', 'loss.backward()')

# 8. scaler.unscale_ -> None
code = code.replace('scaler.unscale_(optimizer)', '#scaler.unscale_ removed')

# 9. scaler.step -> optimizer.step
code = code.replace('scaler.step(optimizer)', 'optimizer.step()')

# 10. scaler.update -> None
code = code.replace('scaler.update()', '#scaler.update removed')

# 11. loss = None before if/else
code = code.replace(
    'codes = batch["codes"].to(device)\n\n            if "cond_embeds" in batch:',
    'codes = batch["codes"].to(device)\n\n            loss = None\n            if "cond_embeds" in batch:')

# 12. NaN guard
code = code.replace(
    'epoch_loss += loss.item() * args.grad_accum_steps',
    'lv=loss.item()*args.grad_accum_steps; epoch_loss+=lv if lv<1e4 else 0.0')
code = code.replace(
    'val_loss += loss.item()',
    'lv=loss.item(); val_loss+=lv if lv<1e4 else 0.0')

# 13. Conda collate to fp32
code = code.replace(
    'torch.stack([b["cond_embeds"] for b in batch])',
    'torch.stack([b["cond_embeds"] for b in batch]).float()')

with open("/root/autodl-tmp/bpt/train_cloud.py", "w") as f:
    f.write(code)

compile(code, "test", "exec")
print("FP32 PATCH OK")
for p in [".float()", "loss = None", "lv<1e4", "# fp32"]:
    print(f"  {p}: {code.count(p)}")
