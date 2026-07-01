"""Scan training samples for NaN loss."""
import sys; sys.path.insert(0, ".")
import torch, yaml, os, random, trimesh
from model.model import MeshTransformer
from model.serializaiton import BPT_serialize

with open("config/BPT-open-8k-8-16.yaml") as f:
    c = yaml.load(f, Loader=yaml.FullLoader)

print("Loading model...")
model = MeshTransformer(
    dim=c["dim"], attn_depth=c["depth"],
    max_seq_len=c["max_seq_len"], dropout=c["dropout"],
    mode=c["mode"], num_discrete_coors=2**int(c["quant_bit"]),
    block_size=c["block_size"], offset_size=c["offset_size"],
    conditioned_on_pc=c["conditioned_on_pc"],
    use_special_block=c["use_special_block"],
    encoder_name=c["encoder_name"], encoder_freeze=c["encoder_freeze"])
model.load("weights/bpt-8-16-500m.pt")
model = model.cuda().float().eval()
print(f"Model loaded, dtype: {next(model.parameters()).dtype}")

dd = "../supervised_data"
samples = [d for d in os.listdir(dd) if os.path.isdir(os.path.join(dd, d)) and d != "Cube"]
random.seed(42)
random.shuffle(samples)

bad = 0
total = 0
losses = []
for s in samples[:100]:
    gt = os.path.join(dd, s, "ground_truth.obj")
    ef = os.path.join("cached_embeddings", f"{s}.pt")
    if not os.path.exists(gt) or not os.path.exists(ef):
        continue

    mesh = trimesh.load(gt, force="mesh", process=False)
    codes = BPT_serialize(mesh, mesh_type="triangle")
    token_len = len(codes)
    codes = codes[:5000] if len(codes) > 5000 else codes
    codes_t = torch.from_numpy(codes).long().unsqueeze(0).cuda()
    emb = torch.load(ef, map_location="cpu").float().cuda()

    total += 1
    with torch.no_grad():
        loss = model.forward_on_codes(codes_t, return_loss=True, cond_embeds=emb)
    val = loss.item()
    finite = torch.isfinite(loss).item()
    losses.append((s, val, finite, token_len))
    if not finite:
        bad += 1
        print(f"BAD: {s} loss={val} tokens={token_len}")

# Summary
losses.sort(key=lambda x: x[1])
print(f"\nTotal: {bad}/{total} NaN")
print(f"Loss range: {losses[0][1]:.2f} to {losses[-1][1]:.2f}")
print(f"Median loss: {losses[len(losses)//2][1]:.2f}")
print(f"Token lengths: {min(l[3] for l in losses)} to {max(l[3] for l in losses)}")
print("DONE")
