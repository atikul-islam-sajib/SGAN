import sys
import os
import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

sys.path.append("src/")

from utils import device_init, config, load
from discriminator import Discriminator


class TestModel:
    def __init__(self, model=None, device="mps"):
        self.model = model
        self.device = device

        self.device = device_init(device=self.device)
        self.config = config()
        self.netD = Discriminator(in_channels=1, out_channels=512)
        self.netD.to(self.device)

    def select_best_model(self):
        if self.model is None:
            load_state = torch.load(
                os.path.join(self.config["path"]["best_model"], "best_model.pth")
            )

            self.netD.load_state_dict(load_state["model"])

        else:
            self.netD.load_state_dict(torch.load(self.model))

    def load_dataloader(self):
        if os.path.exists(self.config["path"]["processed_path"]):
            self.val_dataloader = load(
                filename=os.path.join(
                    self.config["path"]["processed_path"], "val_dataloader.pkl"
                )
            )

            return self.val_dataloader

        else:
            raise Exception("Cannot be loaded the dataloader".capitalize())

    def test(self):
        try:
            self.select_best_model()
        except Exception as e:
            print(e)

        else:
            real_labels = []
            pred_labels = []

            for images, labels in self.load_dataloader():
                images = images.to(self.device)
                labels = labels.to(self.device)

                clf_pred, _ = self.netD(images)
                clf_labels = torch.argmax(clf_pred, dim=1)
                clf_labels = clf_labels.cpu().detach().flatten().numpy()

                real_labels.extend(labels.cpu().detach().flatten().numpy())
                pred_labels.extend(clf_labels)

            print("Accuracy # {}".format(accuracy_score(real_labels, pred_labels)))
            print("Precision # {}".format(precision_score(real_labels, pred_labels)))
            print("Recall # {}".format(recall_score(real_labels, pred_labels)))
            print("F1 # {}".format(f1_score(real_labels, pred_labels)))
            print("ROC_AUC # {}".format(roc_auc_score(real_labels, pred_labels)))

            pd.DataFrame(
                {
                    "accuracy": accuracy_score(real_labels, pred_labels),
                    "precision": precision_score(real_labels, pred_labels),
                    "recall": recall_score(real_labels, pred_labels),
                    "f1": f1_score(real_labels, pred_labels),
                    "roc_auc": roc_auc_score(real_labels, pred_labels),
                },
                index=["Metrics".capitalize()],
            ).T.to_csv(os.path.join(self.config["path"]["results_path"], "results.csv"))

            sns.heatmap(confusion_matrix(real_labels, pred_labels), annot=True, fmt="d")
            plt.savefig(
                os.path.join(
                    self.config["path"]["results_path"], "confusion_matrix.png"
                )
            )
            print("*" * 100, "\n")
            print(classification_report(real_labels, pred_labels))
            print("*" * 100, "\n")
            print(confusion_matrix(real_labels, pred_labels))

            print("metrics results is saved to " + self.config["path"]["results_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Model for SGAN".title())

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Define the model".capitalize(),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="mps",
    )
    args = parser.parse_args()

    test_model = TestModel(model=args.model, device=args.device)
    test_model.test()
