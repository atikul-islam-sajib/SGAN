import sys
import os
import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph
from collections import OrderedDict

sys.path.append("src")

from utils import config
from generator_block import GeneratorBlock


class Generator(nn.Module):
    def __init__(self, latent_space=100, out_channels=1):
        super(Generator, self).__init__()

        self.in_channels = latent_space
        self.out_channels = out_channels * 64

        self.kernel = 4
        self.stride = 1
        self.padding = 0

        self.layers = []

        self.layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel,
                    stride=self.stride,
                    padding=self.padding,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )
        self.in_channels = self.out_channels

        for idx in tqdm(range(4)):
            self.layers.append(
                GeneratorBlock(
                    in_channels=self.in_channels,
                    out_channels=1 if idx == (4 - 1) else self.in_channels * 2,
                    is_last=True if idx == (4 - 1) else False,
                )
            )

            self.in_channels *= 2

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        if x is not None:
            return self.model(x)

        else:
            raise ValueError("Input is not found".capitalize())

    @staticmethod
    def total_params(model=None):
        if model is not None:
            return sum(params.numel() for params in model.parameters())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the Generator for SGAN".title()
    )
    parser.add_argument(
        "--latent_space",
        type=int,
        default=100,
        help="Define the latent space".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=1,
        help="Define the number of channels".capitalize(),
    )

    args = parser.parse_args()
    config = config()

    netG = Generator(latent_space=args.latent_space, out_channels=args.out_channels)

    print(netG(torch.randn(4, 100, 1, 1)).size())

    print(summary(model=netG, input_size=(100, 1, 1)))

    draw_graph(model=netG, input_data=torch.randn(4, 100, 1, 1)).visual_graph.render(
        filename=(
            os.path.join(config["path"]["files_path"], "netG")
            if os.path.exists(config["path"]["files_path"])
            else "Cannot save the image netD".capitalize()
        ),
        format="jpeg",
    )
