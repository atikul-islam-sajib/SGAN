import sys
import os
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR

import torch

sys.path.append("src/")

from utils import dump, config, weight_init, device_init
from helpers import helper


class Trainer:
    def __init__(
        self,
        epochs=200,
        lr=0.0002,
        latent=100,
        adam=True,
        SGD=False,
        device="mps",
        lr_scheduler=False,
        l1_loss=False,
        l2_loss=False,
        elastic_net=False,
        is_display=False,
        is_weight_init=True,
        is_weight_clip=False,
    ):

        self.epochs = epochs
        self.lr = lr
        self.latent = latent
        self.adam = adam
        self.SGD = SGD
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.l1_loss = l1_loss
        self.l2_loss = l2_loss
        self.elastic = elastic_net
        self.is_display = is_display
        self.weight_init = is_weight_init
        self.is_weight_clip = is_weight_clip

        self.device = device_init(device=device)

        init = helper(
            lr=self.lr,
            adam=self.adam,
            SGD=self.SGD,
        )

        self.train_dataloader = init["train_dataloader"]
        self.test_dataloader = init["test_dataloader"]
        self.val_dataloader = init["val_dataloader"]

        self.netG = init["netG"]
        self.netD = init["netD"]

        if self.weight_init:
            self.netG.apply(weight_init)
            self.netD.apply(weight_init)

        self.netG.to(self.device)
        self.netD.to(self.device)

        self.optimizerG = init["optimizerG"]
        self.optimizerD = init["optimizerD"]

        self.adversarial_loss = init["adversarial_loss"]
        self.criterion_loss = init["criterion_loss"]

        if self.lr_scheduler:
            self.schedulerG = StepLR(self.optimizerG, step_size=15, gamma=0.80)
            self.schedulerD = StepLR(self.optimizerD, step_size=15, gamma=0.80)

        self.check_accuracy = float("inf")

        self.config = config()
        self.total_netG_loss = []
        self.total_netD_loss = []
        self.total_train_accuracy = []
        self.total_test_accuracy = []
        self.total_model_history = {
            "netG_loss": list(),
            "netD_loss": list(),
            "train_accuracy": list(),
            "test_accuracy": list(),
        }

    def l1(self, model):
        if model is not None:
            return sum(torch.norm(params, 1) for params in model.parameters())

        else:
            raise ValueError("Model is not found".capitalize())

    def l2(self, model):
        if model is not None:
            return sum(torch.norm(params, 2) for params in model.parameters())
        else:
            raise ValueError("Model is not found".capitalize())

    def elastic_net(self, model):
        if model is not None:
            l1 = self.l1(model=model)
            l2 = self.l2(model=model)

            return 0.01 * (l1 + l2)

        else:
            raise ValueError("Model is not found".capitalize())

    def saved_checkpoints(self, **kwargs):
        if os.path.exists(self.config["path"]["train_models"]):
            torch.save(
                self.netD.state_dict(),
                os.path.join(
                    self.config["path"]["train_models"],
                    "model{}.pth".format(kwargs["epoch"]),
                ),
            )
        else:
            raise Exception("Cannot be saved the model".capitalize())

        if os.path.exists(self.config["path"]["best_model"]):
            if self.check_accuracy > kwargs["train_accuracy"]:
                self.check_accuracy = kwargs["train_accuracy"]

                torch.save(
                    {
                        "model": self.netD.state_dict(),
                        "accuracy": kwargs["train_accuracy"],
                        "test_accuracy": kwargs["test_accuracy"],
                        "epoch": kwargs["epoch"],
                    },
                    os.path.join(self.config["path"]["best_model"], "best_model.pth"),
                )

        else:
            raise Exception("Cannot be saved the model".capitalize())

    def saved_metrics(self, **kwargs):

        self.total_model_history["netG_loss"].extend(kwargs["total_netG_loss"])
        self.total_model_history["netD_loss"].extend(kwargs["total_netD_loss"])
        self.total_model_history["train_accuracy"].extend(
            kwargs["total_train_accuracy"]
        )
        self.total_model_history["test_accuracy"].extend(kwargs["total_test_accuracy"])

        dump(
            value=self.total_model_history,
            filename=(
                os.path.join(self.config["path"]["metrics_path"], "metrics.pkl")
                if os.path.exists(self.config["path"]["metrics_path"])
                else "Cannot be saved the model metrics".capitalize()
            ),
        )

        if os.path.exists(self.config["path"]["files_path"]):
            pd.DataFrame(
                {
                    "netG_loss": self.total_model_history["netG_loss"],
                    "netD_loss": self.total_model_history["netD_loss"],
                    "train_accuracy": self.total_model_history["train_accuracy"],
                    "test_accuracy": self.total_model_history["test_accuracy"],
                }
            ).to_csv(
                os.path.join(self.config["path"]["files_path"], "model_history.csv"),
                index=False,
            )

        else:
            raise Exception("Cannot be saved the model metrics".capitalize())

    def saved_train_images(self, **kwargs):
        if os.path.exists(self.config["path"]["train_images"]):

            fake_images = torch.randn((16, 100, 1, 1)).to(self.device)
            predicted_images = self.netG(fake_images)

            save_image(
                predicted_images,
                os.path.join(
                    self.config["path"]["train_images"],
                    "image{}.png".format(kwargs["epoch"]),
                ),
                nrow=4,
            )

        else:
            raise Exception("Cannot be saved the model images".capitalize())

    def update_discriminator_model(self, **kwargs):
        self.optimizerD.zero_grad()

        fake_clf_pred, fake_validity_pred = self.netD(self.netG(kwargs["fake_samples"]))
        fake_validity_loss = self.criterion_loss(
            fake_validity_pred, kwargs["fake_zeros"]
        )
        fake_clf_loss = self.adversarial_loss(fake_clf_pred, kwargs["fake_labels"])

        real_clf_pred, real_validity_pred = self.netD(kwargs["images"])
        real_validity_loss = self.criterion_loss(
            real_validity_pred, kwargs["real_ones"]
        )
        real_clf_loss = self.adversarial_loss(real_clf_pred, kwargs["labels"])

        total_loss = 0.5 * (
            fake_validity_loss + fake_clf_loss + real_validity_loss + real_clf_loss
        )

        if self.l1_loss:
            total_loss += 0.001 * self.l1(model=self.netD)

        if self.l2_loss:

            total_loss += 0.001 * self.l2(model=self.netD)

        if self.elastic_net:
            total_loss += 0.001 * self.elastic_net(model=self.netD)

        total_loss.backward()
        self.optimizerD.step()

        train_accuracy = accuracy_score(
            torch.argmax(real_clf_pred, dim=1).cpu().detach().numpy(),
            kwargs["labels"].cpu().detach().numpy(),
        )

        return {"loss": total_loss.item(), "accuracy": train_accuracy}

    def update_generator_model(self, **kwargs):
        self.netG.zero_grad()

        _, generated_predict = self.netD(self.netG(kwargs["fake_samples"]))

        loss = self.criterion_loss(
            torch.sigmoid(generated_predict), kwargs["real_ones"]
        )

        if self.l1_loss:
            loss += 0.001 * self.l1(model=self.netG)

        if self.l2_loss:
            loss += 0.001 * self.l2(model=self.netG)

        if self.elastic_net:
            loss += 0.001 * self.elastic_net(model=self.netG)

        if self.is_weight_clip:
            for params in self.netG.parameters():
                params.data.clamp_(-0.01, 0.01)

        loss.backward()
        self.optimizerG.step()

        return {"loss": loss.item()}

    def show_progress(self, **kwargs):
        if self.is_display:
            print(
                "Epochs - [{}/{}] - netG_loss: [{:.4f}] - netD_loss: [{:.4f}] - train_acc: [{:.4f}] - test_accu: [{:.4f}]".format(
                    kwargs["epoch"],
                    self.epochs,
                    kwargs["netG_loss"],
                    kwargs["netD_loss"],
                    kwargs["train_accuracy"],
                    kwargs["test_accuracy"],
                )
            )

        else:
            print("Epoch -[{}/{}] is completed.".format(kwargs["epoch"], self.epochs))

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            self.netG_loss = []
            self.netD_loss = []
            self.train_accuracy = []
            self.test_accuracy = []

            for _, (images, labels) in enumerate(self.train_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                batch = images.size(0)
                fake_samples = torch.randn((batch, self.latent, 1, 1)).to(self.device)
                fake_labels = torch.randint(0, 2, (batch, 2), dtype=torch.float).to(
                    self.device
                )
                real_ones = torch.ones((batch, 1)).to(self.device)
                fake_zeros = torch.zeros((batch, 1)).to(self.device)

                netD = self.update_discriminator_model(
                    images=images,
                    labels=labels,
                    fake_samples=fake_samples,
                    fake_labels=fake_labels,
                    real_ones=real_ones,
                    fake_zeros=fake_zeros,
                )

                self.netD_loss.append(netD["loss"])
                self.train_accuracy.append(netD["accuracy"])

                netG = self.update_generator_model(
                    fake_samples=fake_samples,
                    real_ones=real_ones,
                )

                self.netG_loss.append(netG["loss"])

            for _, (images, labels) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                clf_pred, _ = self.netD(images)
                clf_labels = torch.argmax(clf_pred, dim=1)
                clf_labels = clf_labels.cpu().detach().numpy()

                self.test_accuracy.append(
                    accuracy_score(clf_labels, labels.cpu().detach().numpy())
                )

            if self.lr_scheduler:
                self.schedulerG.step()
                self.schedulerD.step()

            if self.is_display:
                self.show_progress(
                    epoch=epoch + 1,
                    netG_loss=np.mean(self.netG_loss),
                    netD_loss=np.mean(self.netD_loss),
                    train_accuracy=np.mean(self.train_accuracy),
                    test_accuracy=np.mean(self.test_accuracy),
                )

            self.saved_checkpoints(
                epoch=epoch + 1,
                train_accuracy=np.mean(self.train_accuracy),
                test_accuracy=np.mean(self.test_accuracy),
            )

            self.total_netD_loss.append(np.mean(self.netD_loss))
            self.total_netG_loss.append(np.mean(self.netG_loss))
            self.total_train_accuracy.append(np.mean(self.train_accuracy))
            self.total_test_accuracy.append(np.mean(self.test_accuracy))

            try:
                # self.saved_train_images(epoch=epoch + 1)
                pass

            except Exception as e:
                print("The exception is in saved_train_images {}".format(e))

        try:

            self.saved_metrics(
                total_netD_loss=self.total_netD_loss,
                total_netG_loss=self.total_netG_loss,
                total_train_accuracy=self.total_train_accuracy,
                total_test_accuracy=self.total_test_accuracy,
            )

        except Exception as e:
            print("The exception is {}".format(e))

    @staticmethod
    def plot_history():
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training for SGAN".title())
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to train the model".title(),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Learning rate".title(),
    )

    parser.add_argument(
        "--adam", type=bool, default=True, help="Use adam optimizer".title()
    )
    parser.add_argument(
        "--SGD", type=bool, default=False, help="Use SGD optimizer".title()
    )
    parser.add_argument(
        "--l1_loss", type=bool, default=True, help="Use l1 loss".title()
    )
    parser.add_argument(
        "--l2_loss", type=bool, default=True, help="Use l2 loss".title()
    )
    parser.add_argument(
        "--elastic_net", type=bool, default=True, help="Use elastic net".title()
    )
    parser.add_argument(
        "--is_weight_clip", type=bool, default=True, help="Use weight clip".title()
    )
    parser.add_argument(
        "--is_display", type=bool, default=True, help="Use display".title()
    )
    parser.add_argument(
        "--lr_scheduler", type=bool, default=True, help="Use lr scheduler".title()
    )
    parser.add_argument(
        "--latent", type=int, default=100, help="Latent dimension".title()
    )
    parser.add_argument(
        "--is_weight_init", type=bool, default=True, help="Use weight init".title()
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Define the device".title()
    )

    args = parser.parse_args()

    trainer = Trainer(
        epochs=args.epochs,
        lr=args.lr,
        adam=args.lr,
        SGD=args.SGD,
        l2_loss=args.l2_loss,
        elastic_net=args.elastic_net,
        is_display=args.is_display,
        is_weight_clip=args.is_weight_clip,
        is_weight_init=args.is_weight_init,
        device=args.device,
        lr_scheduler=args.lr_scheduler,
        latent=args.latent,
    )
    trainer.train()
