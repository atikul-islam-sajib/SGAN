import argparse
import torch
import torch.nn as nn


class ValidityLoss(nn.Module):
    def __init__(self):
        super(ValidityLoss, self).__init__()

        self.name = "Validity loss".capitalize()

        self.criterion = nn.BCELoss()

    def forward(self, predict, actual):
        if (predict is not None) and (actual is not None):
            return self.criterion(predict, actual)

        else:
            raise ValueError("Input is not found".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GAN - Discriminator loss for SGAN".title()
    )

    parser.add_argument(
        "--criterion", action="store_true", help="Define classification".capitalize()
    )

    args = parser.parse_args()

    if args.criterion:
        validity_loss = ValidityLoss()

        actual = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])
        predict = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0])

        print(validity_loss(predict, actual))

    else:
        raise ValueError("Define the arguments properly".capitalize())
