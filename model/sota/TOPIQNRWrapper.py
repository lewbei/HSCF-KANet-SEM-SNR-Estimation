from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


MODEL_DIRECTORY = Path(__file__).resolve().parent
TOPIQ_REPOSITORY = MODEL_DIRECTORY / "github" / "IQA-PyTorch"

if not TOPIQ_REPOSITORY.is_dir():
    raise FileNotFoundError(
        f"IQA-PyTorch repository was not found at {TOPIQ_REPOSITORY}"
    )

repository_string = str(TOPIQ_REPOSITORY)

if repository_string not in sys.path:
    sys.path.insert(0, repository_string)

from pyiqa.archs.topiq_arch import CFANet


class CFANetFromScratch(CFANet):
    def fix_bn(self, model: nn.Module):
        return None


class TOPIQNRSNRWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = CFANetFromScratch(
            semantic_model_name="resnet50",
            model_name="cfanet_nr_koniq_res50",
            backbone_pretrain=False,
            in_size=None,
            use_ref=False,
            num_class=1,
            num_crop=1,
            crop_size=256,
            inter_dim=256,
            num_heads=4,
            num_attn_layers=1,
            dprate=0.1,
            activation="gelu",
            pretrained=False,
            pretrained_model_path=None,
            out_act=False,
            block_pool="weighted_avg",
            test_img_size=None,
            align_crop_face=False,
        )

    @staticmethod
    def _convert_to_rgb(image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected 4D tensor but got {image.shape}")
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels but got {image.shape[1]}")
        return image

    def train(self, mode: bool = True):
        super().train(mode)
        self.network.train(mode)
        if mode:
            self.network.semantic_model.train(True)
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self._convert_to_rgb(image)
        if self.training:
            self.network.semantic_model.train(True)
        return self.network(image, return_mos=False, return_dist=True)
