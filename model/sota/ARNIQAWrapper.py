from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


MODEL_DIRECTORY = Path(__file__).resolve().parent
ARNIQA_REPOSITORY = MODEL_DIRECTORY / "github" / "ARNIQA"

if not ARNIQA_REPOSITORY.is_dir():
    raise FileNotFoundError(
        f"ARNIQA repository was not found at {ARNIQA_REPOSITORY}"
    )

repository_string = str(ARNIQA_REPOSITORY)

if repository_string not in sys.path:
    sys.path.insert(0, repository_string)

from models.resnet import ResNet


class ARNIQASNRWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        upstream_encoder = ResNet(
            embedding_dim=128,
            pretrained=False,
            use_norm=True,
        )

        self.encoder = upstream_encoder.model
        self.feature_dimension = upstream_encoder.feat_dim

        self.regressor = nn.Linear(
            self.feature_dimension * 2,
            1,
        )

        self.register_buffer(
            "input_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "input_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
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

    def _normalise(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.input_mean) / self.input_std

    def _extract_features(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        features = features.flatten(1)
        return F.normalize(features, dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self._convert_to_rgb(image)

        half_scale_image = F.interpolate(
            image, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True,
        )

        full_scale_image = self._normalise(image)
        half_scale_image = self._normalise(half_scale_image)

        full_scale_features = self._extract_features(full_scale_image)
        half_scale_features = self._extract_features(half_scale_image)

        combined_features = torch.cat([full_scale_features, half_scale_features], dim=1)

        return self.regressor(combined_features)
