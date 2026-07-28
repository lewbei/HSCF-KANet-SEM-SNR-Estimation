"""
CNN + MLP Regression Head (replaces KAN with standard MLP)
Ablation variant: tests if KAN provides benefit over standard MLP

Reviewer requested comparison with "Standard MLP regression heads"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCU(nn.Module):
    def forward(self, x):
        return x * torch.cos(x)


class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, C, k=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=(k - 1) // 2, bias=False)
        self.sig = nn.Sigmoid()

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


class CNNMLP(nn.Module):
    """
    CNN backbone → 64-d vector + 4 PSD scalars + n local-variance scalars
    → LayerNorm → MLP → 1 regression output

    Same architecture as HSCF-KANet but replaces KAN with standard MLP.
    """
    def __init__(self, ksizes=(3, 5, 7, 9), hidden_dim=32, dropout_p=0.0):
        super().__init__()
        self.ksizes = ksizes
        self.n_vars = len(ksizes)
        self.n_psd = 4

        # CNN backbone
        self.conv1 = conv_block(1, 16)
        self.conv2 = conv_block(16, 32)
        self.eca2 = ECA(32)
        self.conv3 = conv_block(32, 64)
        self.eca3 = ECA(64)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Feature normalizer
        in_dim = 64 + self.n_psd + self.n_vars
        self.feat_norm = nn.LayerNorm(in_dim, elementwise_affine=True)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        # MLP head (replaces KAN)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    @staticmethod
    def psd_stats(x, eps=1e-8):
        B, _, H, W = x.shape
        fft = torch.fft.fft2(x, norm='ortho')
        power = torch.abs(fft)**2 + eps

        ky = torch.fft.fftfreq(H, d=1.0 / H, device=x.device).reshape(1, 1, H, 1)
        kx = torch.fft.fftfreq(W, d=1.0 / W, device=x.device).reshape(1, 1, 1, W)
        k_mag = torch.sqrt(kx**2 + ky**2)

        hf_mask = (k_mag > 0.25).float()
        lf_mask = 1.0 - hf_mask

        p_tot = power.mean(dim=(-2, -1))
        p_hf = (power * hf_mask).sum(dim=(-2, -1)) / hf_mask.sum()
        p_lf = (power * lf_mask).sum(dim=(-2, -1)) / lf_mask.sum()

        ratio = p_hf / (p_lf + eps)
        centroid = (power * k_mag).mean(dim=(-2, -1)) / p_tot

        return torch.cat([p_tot.log(), p_hf.log(), ratio.log(), centroid], dim=1)

    @staticmethod
    def local_var(x, k):
        pad = k // 2
        m = F.avg_pool2d(x, k, 1, pad)
        m2 = F.avg_pool2d(x * x, k, 1, pad)
        return m2 - m * m

    def forward(self, x):
        # CNN branch
        z = self.conv1(x)
        z = self.eca2(self.conv2(z))
        z = self.eca3(self.conv3(z))
        z = self.pool(z).flatten(1)  # [B, 64]

        # PSD features
        psd_s = self.psd_stats(x)  # [B, 4]

        # Local variance features
        var_feats = [self.local_var(x, k).mean(dim=(-2, -1)) for k in self.ksizes]
        var_feats = torch.cat(var_feats, dim=1)  # [B, n_vars]

        # Concatenate and normalize
        feat = torch.cat([z, psd_s, var_feats], dim=1)
        feat = self.feat_norm(feat)
        feat = self.dropout(feat)

        # MLP regression head
        return self.mlp(feat)


model = CNNMLP(ksizes=(3, 5, 7, 9), hidden_dim=32, dropout_p=0.0).to(device)
