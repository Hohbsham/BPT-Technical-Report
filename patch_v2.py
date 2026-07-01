"""Patch train_cloud.py: keep autocast + scaler. Only add loss init, NaN guard, early stop."""
with open("/root/autodl-tmp/bpt/train_cloud.py", "r") as f:
    code = f.read()

# 1. loss = None before if/else
code = code.replace(
    'codes = batch["codes"].to(device)\n\n            if "cond_embeds" in batch:',
    'codes = batch["codes"].to(device)\n\n            loss = None\n            if "cond_embeds" in batch:')

# 2. NaN guard
code = code.replace(
    'epoch_loss += loss.item() * args.grad_accum_steps',
    'lv=loss.item()*args.grad_accum_steps; epoch_loss+=lv if lv<1e4 else 0.0')
code = code.replace(
    'val_loss += loss.item()',
    'lv=loss.item(); val_loss+=lv if lv<1e4 else 0.0')

# 3. init pc counter
code = code.replace(
    "best_val_loss = float('inf')",
    'best_val_loss = float("inf"); pc=0')

# 4. Early stop block
old = """if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(str(out_dir / \"best_model.pt\"))
            print(f\"  -> Best model saved (val_loss={best_val_loss:.4f})\")"""

new = """if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(str(out_dir / \"best_model.pt\"))
            print(f\"  -> Best model saved (val_loss={best_val_loss:.4f})\")
            pc = 0
        else:
            pc += 1
            print(f\"  stagnant ({pc}/3)\")
            if pc >= 3:
                print(f\"Early stop epoch {epoch+1}\")
                break"""

if old in code:
    code = code.replace(old, new)
    print("early stop applied")
else:
    # Try with escaped quotes
    old2 = 'if avg_val_loss < best_val_loss:\n            best_val_loss = avg_val_loss\n            model.save(str(out_dir / "best_model.pt"))\n            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")'
    new2 = 'if avg_val_loss < best_val_loss:\n            best_val_loss = avg_val_loss\n            model.save(str(out_dir / "best_model.pt"))\n            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")\n            pc = 0\n        else:\n            pc += 1\n            print(f"  stagnant ({pc}/3)")\n            if pc >= 3:\n                print(f"Early stop epoch {epoch+1}")\n                break'
    if old2 in code:
        code = code.replace(old2, new2)
        print("early stop applied (v2)")

with open("/root/autodl-tmp/bpt/train_cloud.py", "w") as f:
    f.write(code)

compile(code, "test", "exec")
print("PATCH OK")
for p in ["loss = None", "lv<1e4", "pc=0", "Early stop"]:
    print(f"  {p}: {code.count(p)}")
