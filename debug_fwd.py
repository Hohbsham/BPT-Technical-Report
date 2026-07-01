import sys; sys.path.insert(0,".")
import torch, yaml
from model.model import MeshTransformer

with open("config/BPT-open-8k-8-16.yaml") as f:
    c = yaml.load(f, Loader=yaml.FullLoader)

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
print(f"dtype: {next(model.parameters()).dtype}")

emb = torch.load("cached_embeddings/1dlg_Dress101.pt", map_location="cpu").float().cuda()
codes = torch.randint(0, 5000, (1, 500)).cuda()

with torch.no_grad():
    loss = model.forward_on_codes(codes, return_loss=True, cond_embeds=emb)
    print(f"loss: {loss.item():.4f}, finite: {torch.isfinite(loss).item()}")
    logits = model.forward_on_codes(codes, return_loss=False, cond_embeds=emb)
    print(f"logits finite: {torch.isfinite(logits).all().item()}")
    print(f"logits range: {logits.min().item():.2f} to {logits.max().item():.2f}")

# Test with real training data
import trimesh
from model.serializaiton import BPT_serialize
import os

dd = "../supervised_data"
samples = [d for d in os.listdir(dd) if os.path.isdir(os.path.join(dd,d)) and d != "Cube"]
for s in samples[:5]:
    gt_path = os.path.join(dd, s, "ground_truth.obj")
    if not os.path.exists(gt_path): continue
    mesh = trimesh.load(gt_path, force="mesh", process=False)
    codes_real = BPT_serialize(mesh, mesh_type="triangle")
    codes_t = torch.from_numpy(codes_real[:500]).long().unsqueeze(0).cuda()

    emb_f = os.path.join("cached_embeddings", f"{s}.pt")
    if os.path.exists(emb_f):
        e = torch.load(emb_f, map_location="cpu").float().cuda()
        with torch.no_grad():
            loss = model.forward_on_codes(codes_t, return_loss=True, cond_embeds=e)
            print(f"{s}: loss={loss.item():.4f}, finite={torch.isfinite(loss).item()}")

print("DONE")
