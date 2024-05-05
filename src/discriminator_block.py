import sys
import os
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph
from collections import OrderedDict

sys.path.append("src")

from utils import config


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        super(DiscriminatorBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel = 3
        self.stride = 2
        self.padding = 1

        self.discriminator = self.block()

    def block(self):
        layers = OrderedDict()

        layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )

        layers["batch_norm"] = nn.BatchNorm2d(self.out_channels)

        layers["relu"] = nn.ReLU(inplace=True)

        return nn.Sequential(layers)

    def forward(self, x):
        if x is not None:
            return self.discriminator(x)
        else:
            raise ValueError("Input is not found".capitalize())

    @staticmethod
    def total_params(model=None):
        if model is not None:
            return sum(params.numel() for params in model.parameters())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discriminator Block".title())
    parser.add_argument(
        "--in_channels", type=int, default=1, help="Input channels".capitalize()
    )
    parser.add_argument(
        "--out_channels", type=int, default=512, help="Output channels".capitalize()
    )
    args = parser.parse_args()

    in_channels = args.in_channels
    out_channels = args.out_channels

    layers = []
    config = config()

    for _ in range(4):
        layers.append(
            DiscriminatorBlock(in_channels=in_channels, out_channels=out_channels)
        )

        in_channels = out_channels
        out_channels //= 2

    model = nn.Sequential(*layers)

    print(summary(model=model, input_size=(1, 128, 128)))

    draw_graph(model=model, input_data=torch.randn(4, 1, 64, 64)).visual_graph.render(
        filename=(
            os.path.join(config["path"]["files_path"], "netD_block")
            if config["path"]["files_path"]
            else "Cannot be saved the file".capitalize()
        ),
        format="jpeg",
    )
