"""Add position embedding resize to train_cloud.py so it can handle longer max_code_len."""
import re

with open("/root/autodl-tmp/bpt/train_cloud.py", "r") as f:
    code = f.read()

old = "model.load(args.model_path)"
new = """model.load(args.model_path)
    # Resize position embeddings if max_code_len > pretrained max_seq_len
    if args.max_code_len > model.max_seq_len:
        import torch.nn.functional as F
        old_weights = model.abs_pos_emb.weight.data
        new_weights = F.interpolate(
            old_weights.T.unsqueeze(0),
            size=args.max_code_len,
            mode='linear'
        ).squeeze(0).T
        model.abs_pos_emb = torch.nn.Embedding(args.max_code_len, model.dim).cuda()
        model.abs_pos_emb.weight.data.copy_(new_weights)
        model.max_seq_len = args.max_code_len
        print(f"Pos emb resized: {old_weights.shape[0]} -> {args.max_code_len}")"""

code = code.replace(old, new)
compile(code, "test", "exec")
print("pos emb resize OK")
with open("/root/autodl-tmp/bpt/train_cloud.py", "w") as f:
    f.write(code)
