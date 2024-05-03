import argparse
import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()

        self.name = "adversarial loss".capitalize()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predict, actual):
        if (predict is not None) and (actual is not None):
            return self.criterion(predict, actual)

        else:
            raise ValueError("Input is not found".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification loss for SGAN".title())

    parser.add_argument(
        "--clf", action="store_true", help="Define classification".capitalize()
    )

    args = parser.parse_args()

    if args.clf:
        adversarial_loss = ClassificationLoss()

        actual = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        predict = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])

        print(adversarial_loss(predict, actual))
