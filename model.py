from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # extra over the original U-Net
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # extra over the original U-Net
            nn.ReLU(inplace=True),
        )

        self.device = "cpu"

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 5,
        features: list[int] = [64, 128, 256, 512],
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # The encoder part of the U-Net
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # The decoder part of the U-Net
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # The encoder part of the U-Net
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        # The decoder part of the U-Net
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

    @override
    def to(self, device: str) -> nn.Module:
        self.device = device
        return super().to(device)

    @torch.inference_mode()
    def segment(self, image) -> torch.Tensor:
        """
        A method for inference, which returns a predicted segmentation mask.
        """
        self.eval()
        image = image.unsqueeze(0).to(self.device)
        logits = self(image)
        probs = F.softmax(logits, dim=1)
        mask = torch.argmax(probs, dim=1)
        return mask.squeeze(0).cpu().detach().numpy()
