#!/usr/bin/env python3
import torch, argparse
from pathlib import Path
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="outputs/m6/torch_fusion_best.pth")
parser.add_argument("--out", default="outputs/m6/fusion_model.pt")
parser.add_argument("--in_dim", type=int, default=None)
args = parser.parse_args()

sd_path = Path(args.ckpt)
if not sd_path.exists():
    raise RuntimeError("Checkpoint not found: " + str(sd_path))

sd = torch.load(str(sd_path), map_location="cpu")
# infer in_dim from first linear weight
in_dim = args.in_dim
if in_dim is None:
    for k,v in sd.items():
        if "weight" in k and v.ndim == 2:
            in_dim = v.shape[1]
            break
if in_dim is None:
    raise RuntimeError("Could not infer input dim; pass --in_dim")

# build same architecture
class FusionNet(nn.Module):
    def __init__(self, in_dim, hidden=256, num_out=None):
        super().__init__()
        # infer num_out from state_dict last linear if possible
        # Find last linear weight shape
        out_dim = None
        for k,v in sd.items():
            if "net.3.weight" in k or "net.2.weight" in k:
                out_dim = v.shape[0]
        # fallback: find any weight with ndim==2 and row count matches
        if out_dim is None:
            for k,v in sd.items():
                if v.ndim == 2:
                    out_dim = v.shape[0]
                    break
        if num_out is not None:
            out_dim = num_out
        if out_dim is None:
            raise RuntimeError("Could not infer output classes from checkpoint")
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=1)

# instantiate and load
m = FusionNet(in_dim)
m.load_state_dict(sd)
m.eval()

# trace and save
example = torch.randn(1, in_dim)
traced = torch.jit.trace(m, example)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
traced.save(args.out)
print("Saved TorchScript fusion model to", args.out)
