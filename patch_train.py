"""Patch train_cloud.py with minimal fixes."""
import re

with open("/root/autodl-tmp/bpt/train_cloud.py", "r") as f:
    code = f.read()

# Fix 1: loss = None before if/else
code = code.replace(
    'codes = batch["codes"].to(device)\n\n            if "cond_embeds" in batch:',
    'codes = batch["codes"].to(device)\n\n            loss = None\n            if "cond_embeds" in batch:')

# Fix 2: NaN guard for training loss
code = code.replace(
    'epoch_loss += loss.item() * args.grad_accum_steps',
    'lv=loss.item()*args.grad_accum_steps; epoch_loss+=lv if lv<1e4 else 0.0')

# Fix 3: NaN guard for validation loss
code = code.replace(
    'val_loss += loss.item()',
    'lv=loss.item(); val_loss+=lv if lv<1e4 else 0.0')

# Fix 4: Early stopping
code = code.replace(
    'best_val_loss = float("inf")',
    'best_val_loss = float("inf"); pc=0')

old_es = '''if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(str(out_dir / "best_model.pt"))
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")'''

new_es = '''if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(str(out_dir / "best_model.pt"))
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")
            pc = 0
        else:
            pc += 1
            print(f"  stagnant ({pc}/3)")
            if pc >= 3:
                print(f"Early stop epoch {epoch+1}")
                break'''

code = code.replace(old_es, new_es)

with open("/root/autodl-tmp/bpt/train_cloud.py", "w") as f:
    f.write(code)

compile(code, "test", "exec")
print("PATCH OK")
for p in ["loss = None", "lv<1e4", "pc=0", "Early stop"]:
    print(f"  {p}: {code.count(p)}")
