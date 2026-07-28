import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 1. Growing-Cosine Unit
class GCU(nn.Module):
    def forward(self, x):
        return x * torch.cos(x)

# 2. Efficient Channel Attention (ECA)
class ECA(nn.Module):
    def __init__(self, C, k_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k_size, padding=(k_size - 1) // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sig(y)

# 3. Conv block
def conv_block(inp, out, pool=True):
    layers = [
        nn.Conv2d(inp, out, 3, padding=1, bias=False),
        nn.GroupNorm(8, out),
        GCU()
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# 4. CNN Feature Extractor (same backbone as CNN_KAN)
class CNN_FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(1, 16)
        self.conv2 = conv_block(16, 32)
        self.eca2 = ECA(32)
        self.conv3 = conv_block(32, 64)
        self.eca3 = ECA(64)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.eca2(self.conv2(x))
        x = self.eca3(self.conv3(x))
        x = self.pool(x).flatten(1)
        return x

# 5. Instantiate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN_FeatureExtractor().to(device)
print('CNN_RF Model ready on', device)
