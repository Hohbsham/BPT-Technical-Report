"""BPT Fine-tuning — Final Version. 3 fixes only: loss init, NaN guard, val autocast."""
with open('d:/ClothesNetData/bpt/train_cloud.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Fix 1: loss = None before if/else (prevent UnboundLocalError on NaN batches)
code = code.replace(
    'codes = batch["codes"].to(device)\n\n            if "cond_embeds" in batch:',
    'codes = batch["codes"].to(device)\n\n            loss = None\n            if "cond_embeds" in batch:')

# Fix 2: NaN guard — filter inf/NaN from epoch loss averages
code = code.replace(
    'epoch_loss += loss.item() * args.grad_accum_steps',
    'lv = loss.item() * args.grad_accum_steps; epoch_loss += lv if lv < 1e4 else 0.0')
code = code.replace(
    'val_loss += loss.item()',
    'lv = loss.item(); val_loss += lv if lv < 1e4 else 0.0')

# Fix 3: Add autocast to validation (prevents fp16/fp32 dtype mismatch)
code = code.replace(
    'with torch.no_grad():',
    'with torch.no_grad(), torch.cuda.amp.autocast():')

# Verify
compile(code, 'test.py', 'exec')
checks = {
    'loss = None': code.count('loss = None'),
    'NaN guard (train)': code.count('lv < 1e4'),
    'val autocast': code.count('no_grad(), torch.cuda.amp.autocast()'),
    'autocast (original kept)': code.count('with torch.cuda.amp.autocast():'),
    'GradScaler (original kept)': code.count('GradScaler()'),
}
all_ok = True
for name, count in checks.items():
    ok = count >= 1
    if not ok: all_ok = False
    print(f'  {"OK" if ok else "MISSING"} {name}: {count}')

print(f'\nAll checks: {\"PASS\" if all_ok else \"FAIL\"}')
print(f'File size: {len(code)} bytes')

with open('d:/ClothesNetData/bpt/train_cloud_final_v4.py', 'w', encoding='utf-8') as f:
    f.write(code)
