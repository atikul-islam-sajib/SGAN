import sys
import os
import torch
import unittest

sys.path.append("src")

from utils import config, dump, load
from dataloader import Loader
from discriminator import Discriminator


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.config = config()
        self.train_dataloader = load(
            os.path.join(self.config["path"]["processed_path"], "train_dataloader.pkl")
        )

        self.test_dataloader = load(
            os.path.join(self.config["path"]["processed_path"], "test_dataloader.pkl")
        )

        self.val_dataloader = load(
            os.path.join(self.config["path"]["processed_path"], "val_dataloader.pkl")
        )

        self.netD = Discriminator(in_channels=1, out_channels=512)

    def tearDown(self) -> None:
        return super().tearDown()

    def test_total_train_data(self):
        self.assertEqual(
            sum(data.size(0) for data, labels in self.train_dataloader), 61
        )

    def test_total_test_data(self):
        self.assertEqual(sum(data.size(0) for data, labels in self.test_dataloader), 61)

    def test_total_val_data(self):
        self.assertEqual(sum(data.size(0) for data, labels in self.val_dataloader), 123)

    def test_total_data(self):
        self.total_data = sum(
            [
                sum([data.size(0) for data, labels in dataloader])
                for dataloader in [
                    self.train_dataloader,
                    self.test_dataloader,
                    self.val_dataloader,
                ]
            ]
        )

        self.assertEqual(self.total_data, 245)

    def test_train_data_size(self):
        data, labels = next(iter(self.train_dataloader))

        self.assertEqual(data.size(), torch.Size([4, 3, 64, 64]))

    def test_test_data_size(self):
        data, labels = next(iter(self.test_dataloader))

        self.assertEqual(data.size(), torch.Size([16, 3, 64, 64]))

    def test_val_data_size(self):
        data, labels = next(iter(self.val_dataloader))

        self.assertEqual(data.size(), torch.Size([4, 3, 64, 64]))

    def test_classification_shape(self):
        classification, _ = self.netD(torch.randn(4, 1, 64, 64))

        self.assertEqual(classification.size(), torch.Size([4, 2]))

    def test_validity_shape(self):
        _, validity = self.netD(torch.randn(4, 1, 64, 64))

        self.assertEqual(validity.size(), torch.Size([4, 1]))


if __name__ == "__main__":
    unittest.main()
