"""Debug NaN source: test forward pass with actual data."""
import sys, os
sys.path.insert(0, "/root/autodl-tmp/bpt")
import torch, yaml, trimesh
from model.model import MeshTransformer
from model.serializaiton import BPT_serialize

os.chdir("/root/autodl-tmp/bpt")

with open("config/BPT-open-8k-8-16.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model = MeshTransformer(
    dim=config["dim"], attn_depth=config["depth"],
    max_seq_len=config["max_seq_len"], dropout=config["dropout"],
    mode=config["mode"], num_discrete_coors=2**int(config["quant_bit"]),
    block_size=config["block_size"], offset_size=config["offset_size"],
    conditioned_on_pc=config["conditioned_on_pc"],
    use_special_block=config["use_special_block"],
    encoder_name=config["encoder_name"], encoder_freeze=config["encoder_freeze"])
model.load("weights/bpt-8-16-500m.pt")
model = model.cuda().float().eval()
print(f"Model dtype: {next(model.parameters()).dtype}")

# Test with real embedding
emb = torch.load("cached_embeddings/1dlg_Dress101.pt", map_location="cpu")
print(f"Emb dtype: {emb.dtype}, shape: {emb.shape}")
emb = emb.float().cuda()

# Test with real mesh codes
mesh = trimesh.load("../supervised_data/1dlg_Dress101/ground_truth.obj", force="mesh", process=False)
codes = BPT_serialize(mesh, mesh_type="triangle")
codes = torch.from_numpy(codes).long().unsqueeze(0).cuda()
codes = codes[:, :200]  # short test
print(f"Codes shape: {codes.shape}, max: {codes.max()}, min: {codes.min()}")

# Forward pass
with torch.no_grad():
    loss = model.forward_on_codes(codes, return_loss=True, cond_embeds=emb)
    print(f"Loss: {loss.item():.4f}, finite: {torch.isfinite(loss).item()}")

    # Check logits for NaN
    logits = model.forward_on_codes(codes, return_loss=False, cond_embeds=emb)
    print(f"Logits finite: {torch.isfinite(logits).all().item()}")
    print(f"Logits min/max: {logits.min().item():.2f} / {logits.max().item():.2f}")

print("DEBUG DONE")
