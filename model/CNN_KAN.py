import torch
import torch.nn as nn
from kan import KANLayer                 #  pip install pykan
# (If this import fails:  pip install git+https://github.com/KindXiaoming/pykan.git)

# ─────────────────────────────────────────────────────────────
# 1. Growing-Cosine Unit φ(z)=z·cos z
# ─────────────────────────────────────────────────────────────
class GCU(nn.Module):
    def forward(self, x):
        return x * torch.cos(x)

# ─────────────────────────────────────────────────────────────
# 2. Efficient Channel Attention (ECA)
# ─────────────────────────────────────────────────────────────
class ECA(nn.Module):
    def __init__(self, C, k_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k_size,
                              padding=(k_size - 1) // 2,
                              bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        # x:[B,C,H,W] → y:[B,C,1,1]
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sig(y)

# ─────────────────────────────────────────────────────────────
# 3. Conv → GN → GCU → (optional) MaxPool
# ─────────────────────────────────────────────────────────────
def conv_block(inp: int, out: int, pool: bool = True):
    layers = [
        nn.Conv2d(inp, out, 3, padding=1, bias=False),
        nn.GroupNorm(8, out),
        GCU()
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# ─────────────────────────────────────────────────────────────
# 4. Backbone + KAN head
# ─────────────────────────────────────────────────────────────
class CNN_ECA_GCU_KAN(nn.Module):
    """
    3-stage conv backbone (GCU + ECA) → global-avg-pool (64-D)
    → 1-hidden-layer Kolmogorov–Arnold head → scalar SNR (dB)
    """
    def __init__(self, kan_hidden: int = 32, kan_grid: int = 16):
        super().__init__()

        # Backbone
        self.conv1 = conv_block(1, 16)     # 256 → 128
        self.conv2 = conv_block(16, 32)    # 128 →  64
        self.eca2  = ECA(32)
        self.conv3 = conv_block(32, 64)    #  64 →  32
        self.eca3  = ECA(64)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)   # [B,64,1,1] → [B,64]

        # KAN head (returns 4 values: y, preacts, postacts, postspline)
        self.kan  = KANLayer(
            in_dim  = 64,
            out_dim = kan_hidden,
            num     = kan_grid                 # spline intervals
        )
        self.fc   = nn.Linear(kan_hidden, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.eca2(self.conv2(x))
        x = self.eca3(self.conv3(x))
        x = self.pool(x).flatten(1)            # [B,64]

        y, *_ = self.kan(x)                    # keep only first output
        return self.fc(y)                      # [B,1]

# ─────────────────────────────────────────────────────────────
# 5. Instantiate
# ─────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = CNN_ECA_GCU_KAN().to(device)
print('Model ready on', device)
