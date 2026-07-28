import torch.nn as nn
import timm


class ResNeXt50Regression(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        # in_chans=1 rewires the stem correctly, whatever its internal layout
        self.model = timm.create_model(
            'resnext50d_32x4d',
            pretrained=pretrained,
            in_chans=1,
        )

        # Replace the classification head with a regression head
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.model(x)

model = ResNeXt50Regression().to(device)
