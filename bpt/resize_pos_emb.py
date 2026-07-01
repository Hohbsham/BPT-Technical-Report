"""
Position Embedding Resize Utility
Extends BPT pretrained position embeddings from 10000 to target length.
Usage: import resize_pos_emb; resize_pos_emb.resize(model, 18000)
"""
import torch
import torch.nn.functional as F


def resize(model, target_len, verbose=True):
    """
    Resize model.abs_pos_emb from its current size to target_len.
    Uses linear interpolation (F.interpolate with mode='linear').

    Args:
        model: MeshTransformer instance (after model.load() has been called)
        target_len: desired max_seq_len (e.g., 18000)
        verbose: print status messages

    Returns:
        model with resized position embeddings
    """
    old_weights = model.abs_pos_emb.weight.data  # [old_len, dim]
    old_len, dim = old_weights.shape

    if old_len >= target_len:
        if verbose:
            print(f"Position embeddings already at {old_len} >= {target_len}, skipping")
        return model

    # Linear interpolation: [old_len, dim] -> [target_len, dim]
    # F.interpolate expects [N, C, L] format
    new_weights = F.interpolate(
        old_weights.T.unsqueeze(0),   # [1, dim, old_len]
        size=target_len,
        mode='linear',
        align_corners=True
    ).squeeze(0).T                    # [target_len, dim]

    # Create new embedding with same dtype and device
    device = old_weights.device
    dtype = old_weights.dtype
    model.abs_pos_emb = torch.nn.Embedding(target_len, dim).to(device=device, dtype=dtype)
    model.abs_pos_emb.weight.data.copy_(new_weights)
    model.max_seq_len = target_len

    if verbose:
        print(f"Position embeddings resized: {old_len} -> {target_len}")

    return model


if __name__ == "__main__":
    # Test: load model, resize, verify
    import sys, yaml
    sys.path.insert(0, ".")
    from model.model import MeshTransformer

    print("Loading model...")
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
    print(f"Original max_seq_len: {model.max_seq_len}")
    print(f"Original pos_emb shape: {model.abs_pos_emb.weight.shape}")

    # Resize
    resize(model, 18000)
    print(f"New max_seq_len: {model.max_seq_len}")
    print(f"New pos_emb shape: {model.abs_pos_emb.weight.shape}")

    # Quick forward test
    codes = torch.randint(0, 5000, (1, 200))
    with torch.no_grad():
        loss = model.forward_on_codes(codes, return_loss=True)
        print(f"Forward test loss: {loss.item():.4f}")

    print("ALL TESTS PASSED")
