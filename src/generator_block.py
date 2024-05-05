import sys
import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torchsummary import summary
from torchview import draw_graph
from collections import OrderedDict

sys.path.append("src")

from utils import config


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, is_last=False):
        super(GeneratorBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_last = is_last

        self.kernel = 4
        self.stride = 2
        self.padding = 1

        self.generator = self.block()

    def block(self):
        layers = OrderedDict()

        layers["convTranspose"] = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
        )

        if self.is_last:
            layers["tanh"] = nn.Tanh()

        else:
            layers["batch_norm"] = nn.BatchNorm2d(self.out_channels)
            layers["leaky_relu"] = nn.LeakyReLU(0.2, inplace=True)

        return nn.Sequential(layers)

    def forward(self, x):
        if x is not None:
            return self.generator(x)

        else:
            raise ValueError("Input is not found".capitalize())

    @staticmethod
    def total_params(model=None):
        if model is not None:
            return sum(params.numel() for params in model.parameters())

        else:
            raise ValueError("Model is not found".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generator Block".title())

    parser.add_argument(
        "--in_channels", type=int, default=64, help="Input channels".capitalize()
    )
    parser.add_argument(
        "--out_channels", type=int, default=128, help="Output channels".capitalize()
    )

    args = parser.parse_args()

    latent_space = 100
    out_channels = 64
    kernel = 4
    stride = 1
    padding = 0
    num_repetitive = 4

    layers = []
    config = config()

    layers.append(
        nn.ConvTranspose2d(
            in_channels=latent_space,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=False,
        )
    )

    in_channels = out_channels

    for idx in tqdm(range(num_repetitive)):
        layers.append(
            GeneratorBlock(
                in_channels=in_channels,
                out_channels=1 if idx == (num_repetitive - 1) else in_channels * 2,
                is_last=True if idx == (num_repetitive - 1) else False,
            )
        )

        in_channels *= 2

    model = nn.Sequential(*layers)

    print(model(torch.randn(4, 100, 1, 1)).size())

    print(summary(model=model, input_size=(100, 1, 1)))

    draw_graph(model=model, input_data=torch.randn(4, 100, 1, 1)).visual_graph.render(
        filename=(
            os.path.join(config["path"]["files_path"], "netG_block")
            if os.path.exists(config["path"]["files_path"])
            else "Cannot be saved the model architecture".capitalize()
        ),
        format="jpeg",
    )
