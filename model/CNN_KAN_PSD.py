# model_raw_plus_psd_scalar.py
# ───────────────────────────
# Same backbone as CNN_ECA_GCU_KAN + 4 scalar PSD descriptors

import torch, torch.nn as nn, torch.nn.functional as F
from kan import KANLayer

# ───────────────────────────────────────── Blocks
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
    layers = [nn.Conv2d(inp, out, 3, 1, 1, bias=False),
              nn.GroupNorm(8, out),
              GCU()]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# ──────────────────────────────────────── Model
class RawPlusPSDScalar(nn.Module):
    """
    • raw image → CNN → 64-d feature
    • four global PSD scalars appended to the KAN head
         – total power
         – high-freq power (|k| > 0.25·Nyquist)
         – HF / LF ratio
         – spectral centroid (weighted mean |k|)
    """
    def __init__(self, kan_hidden=32, kan_grid=16):
        super().__init__()

        # backbone
        self.conv1 = conv_block(1, 16)    # 256→128
        self.conv2 = conv_block(16, 32); self.eca2 = ECA(32)
        self.conv3 = conv_block(32, 64); self.eca3 = ECA(64)
        self.pool  = nn.AdaptiveAvgPool2d(1)

        # KAN head
        in_dim = 64 + 4                    # 4 PSD scalars
        self.kan = KANLayer(in_dim, kan_hidden, num=kan_grid)
        self.out = nn.Linear(kan_hidden, 1)

    # ───────────── PSD descriptors ─────────────
    @staticmethod
    def psd_stats(x, eps=1e-8):
        # x : [B,1,H,W], already in [0,1] or z-scored
        B, _, H, W = x.shape
        fft = torch.fft.fft2(x, norm='ortho')        # complex
        power = torch.abs(fft)**2 + eps              # [B,1,H,W]

        # radially averaged frequency magnitude |k|
        ky = torch.fft.fftfreq(H, d=1./H, device=x.device).reshape(1, 1, H, 1)
        kx = torch.fft.fftfreq(W, d=1./W, device=x.device).reshape(1, 1, 1, W)
        k_mag = torch.sqrt(kx**2 + ky**2)            # [1,1,H,W]

        hf_mask = (k_mag > 0.25).float()             # > ¼ Nyquist
        lf_mask = 1. - hf_mask

        p_tot = power.mean(dim=(-2, -1))
        p_hf  = (power * hf_mask).sum(dim=(-2, -1)) / hf_mask.sum()
        p_lf  = (power * lf_mask).sum(dim=(-2, -1)) / lf_mask.sum()

        ratio = p_hf / (p_lf + eps)
        centroid = (power * k_mag).mean(dim=(-2, -1)) / p_tot

        # four scalars per sample → [B,4]
        return torch.cat([p_tot.log(), p_hf.log(),
                          ratio.log(), centroid], dim=1)

    # ───────────────────────── forward ────────────────────────
    def forward(self, x):
        # raw spatial stream
        z = self.conv1(x)
        z = self.eca2(self.conv2(z))
        z = self.eca3(self.conv3(z))
        z = self.pool(z).flatten(1)               # [B,64]

        # PSD *scalars*
        psd_s = self.psd_stats(x)                 # [B,4]

        z = torch.cat([z, psd_s], dim=1)          # [B,68]
        y, *_ = self.kan(z)
        return self.out(y)

# ──────────────────────────────────────── Instantiate (in your script)
# device  = 'cuda' if torch.cuda.is_available() else 'cpu'
model   = RawPlusPSDScalar().to(device)
