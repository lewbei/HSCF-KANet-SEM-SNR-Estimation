import torch, torch.nn as nn, torch.nn.functional as F
from kan import KANLayer


# ────────────────────────── Blocks
class GCU(nn.Module):
    def forward(self, x):
        return x * torch.cos(x)


class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, C, k=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=(k - 1) // 2, bias=False)
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
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


# ────────────────────────── Model
class RawPlusPSDAndNVarScalar(nn.Module):
    """
    raw image → CNN → 64-d vector
              + 4 PSD scalars
              + n local-variance scalars
              → LayerNorm → KAN → 1 regression output
    """
    def __init__(
        self,
        ksizes=(3, 5, 7),
        kan_hidden=32,
        kan_grid=16,
        dropout_p=0.0
    ):
        super().__init__()
        # handcrafted-feature config
        self.ksizes = ksizes
        self.n_vars = len(ksizes)
        self.n_psd  = 4

        # CNN backbone
        self.conv1 = conv_block(1, 16)                 # 256→128
        self.conv2 = conv_block(16, 32); self.eca2 = ECA(32)
        self.conv3 = conv_block(32, 64); self.eca3 = ECA(64)
        self.pool  = nn.AdaptiveAvgPool2d(1)

        # feature normaliser (learnable)
        in_dim = 64 + self.n_psd + self.n_vars
        self.feat_norm = nn.LayerNorm(in_dim, elementwise_affine=True)

        # optional dropout
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        # KAN head
        self.kan = KANLayer(in_dim, kan_hidden, num=kan_grid)
        self.out = nn.Linear(kan_hidden, 1)

    # ────────── PSD descriptors
    @staticmethod
    def psd_stats(x, eps=1e-8):
        B, _, H, W = x.shape
        fft   = torch.fft.fft2(x, norm='ortho')
        power = torch.abs(fft)**2 + eps

        ky = torch.fft.fftfreq(H, d=1.0 / H, device=x.device).reshape(1, 1, H, 1)
        kx = torch.fft.fftfreq(W, d=1.0 / W, device=x.device).reshape(1, 1, 1, W)
        k_mag = torch.sqrt(kx ** 2 + ky ** 2)

        hf_mask = (k_mag > 0.25).float()
        lf_mask = 1.0 - hf_mask

        p_tot = power.mean(dim=(-2, -1))
        p_hf  = (power * hf_mask).sum(dim=(-2, -1)) / hf_mask.sum()
        p_lf  = (power * lf_mask).sum(dim=(-2, -1)) / lf_mask.sum()

        ratio    = p_hf / (p_lf + eps)
        centroid = (power * k_mag).mean(dim=(-2, -1)) / p_tot

        return torch.cat([p_tot.log(), p_hf.log(), ratio.log(), centroid], dim=1)  # [B,4]

    # ────────── local variance
    @staticmethod
    def local_var(x, k):
        pad = k // 2
        m   = F.avg_pool2d(x, k, 1, pad)
        m2  = F.avg_pool2d(x * x, k, 1, pad)
        return m2 - m * m

    # ────────── forward
    def forward(self, x):                               # x: [B,1,H,W]
        # (1) CNN branch
        z = self.conv1(x)
        z = self.eca2(self.conv2(z))
        z = self.eca3(self.conv3(z))
        z = self.pool(z).flatten(1)                     # [B,64]

        # (2) handcrafted scalars (no scaling yet)
        psd_s = self.psd_stats(x)                       # [B,4]

        var_feats = [
            self.local_var(x, k).mean(dim=(-2, -1)) for k in self.ksizes
        ]
        var_feats = torch.cat(var_feats, dim=1)         # [B,n_vars]

        # (3) concatenate and **LayerNorm**
        feat = torch.cat([z, psd_s, var_feats], dim=1)  # [B, in_dim]
        feat = self.feat_norm(feat)                     # learnable normalisation
        feat = self.dropout(feat)

        # (4) KAN head
        y, *_ = self.kan(feat)
        return self.out(y)


model  = RawPlusPSDAndNVarScalar(
    ksizes=(3, 5, 7, 9),
    kan_hidden=32,   # start small again
    kan_grid=16,
    dropout_p=0.0    # disable dropout while testing
).to(device)
