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
from discriminator_block import DiscriminatorBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = []

        for _ in tqdm(range(4)):
            self.layers.append(
                DiscriminatorBlock(
                    in_channels=self.in_channels, out_channels=self.out_channels
                )
            )

            self.in_channels = self.out_channels
            self.out_channels //= 2

        self.model = nn.Sequential(*self.layers)

        self.classification = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=2, bias=False),
            nn.Softmax(dim=1),
        )

        self.validity = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x is not None:
            x = self.model(x)

            classification = self.classification(x.view(x.size(0), -1))
            validity = self.validity(x.view(x.size(0), -1))

            return classification, validity

        else:
            raise ValueError("Input is not found".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discriminator for SGAN".title())

    parser.add_argument(
        "--in_channels", type=int, default=1, help="Input channels".capitalize()
    )

    parser.add_argument(
        "--out_channels", type=int, default=512, help="Output channels".capitalize()
    )

    args = parser.parse_args()

    config = config()

    netD = Discriminator(in_channels=args.in_channels, out_channels=args.out_channels)

    classification, validity = netD(torch.randn(4, 1, 64, 64))

    print(classification.size(), validity.size())

    print(summary(model=netD, input_size=(1, 64, 64)))

    draw_graph(model=netD, input_data=torch.randn(4, 1, 64, 64)).visual_graph.render(
        filename=(
            os.path.join(config["path"]["files_path"], "netD")
            if os.path.exists(config["path"]["files_path"])
            else "Cannot save the image netD".capitalize()
        ),
        format="jpeg",
    )
