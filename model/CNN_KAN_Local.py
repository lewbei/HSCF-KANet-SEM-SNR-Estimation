# model_raw_plus_nvar_scalar.py

import torch, torch.nn as nn
import torch.nn.functional as F
from kan import KANLayer

# ───────────────────────────────────────── Blocks ─────────────────────────────────────────
class GCU(nn.Module):
    def forward(self, x): return x * torch.cos(x)

class ECA(nn.Module):
    def __init__(self, C, k=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=(k-1)//2, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sig(y)

def conv_block(inp, out, pool=True):
    layers = [
        nn.Conv2d(inp, out, 3, 1, 1, bias=False),
        nn.GroupNorm(8, out),
        GCU()
    ]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# ───────────────────────────────────────── Model ────────────────────────────────────────────
class RawPlusNVarScalar(nn.Module):
    """
    • raw image → CNN → 64‐d feature
    • n local‐variance scalars appended to the KAN head
    """
    def __init__(self, ksizes=(3,5,7), kan_hidden=32, kan_grid=16):
        super().__init__()
        self.ksizes = ksizes
        n_vars = len(ksizes)

        # CNN backbone on raw input only
        self.conv1 = conv_block(1, 16)
        self.conv2 = conv_block(16, 32); self.eca2 = ECA(32)
        self.conv3 = conv_block(32, 64); self.eca3 = ECA(64)
        self.pool  = nn.AdaptiveAvgPool2d(1)

        # KAN head now sees 64 + n_vars inputs
        in_dim = 64 + n_vars
        self.kan = KANLayer(in_dim, kan_hidden, num=kan_grid)
        self.out = nn.Linear(kan_hidden, 1)

    @staticmethod
    def local_var(x, k):
        pad = k // 2
        m   = F.avg_pool2d(x, k, 1, pad)
        m2  = F.avg_pool2d(x * x, k, 1, pad)
        return m2 - m * m

    def forward(self, x):
        # 1) CNN feature
        z = self.conv1(x)
        z = self.eca2(self.conv2(z))
        z = self.eca3(self.conv3(z))
        z = self.pool(z).flatten(1)            # [B,64]

        # 2) compute n global var scalars
        var_feats = []
        for k in self.ksizes:
            vmap = self.local_var(x, k)        # [B,1,H,W]
            var_feats.append(vmap.mean(dim=(-2,-1)))  # [B,1]
        var_feats = torch.cat(var_feats, dim=1)      # [B, n_vars]

        # 3) append to CNN features → KAN
        feat = torch.cat([z, var_feats], dim=1)      # [B,64+n_vars]
        y, *_ = self.kan(feat)
        return self.out(y)

model  = RawPlusNVarScalar(ksizes=(3,5,7,9)).to(device)
