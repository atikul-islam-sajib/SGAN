import sys
import os
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sys.path.append("src/")

from utils import config, load
from generator import Generator
from discriminator import Discriminator
from loss.clf_loss import ClassificationLoss
from loss.validity_loss import ValidityLoss


def load_dataloader():
    config_files = config()

    if os.path.exists(config_files["path"]["processed_path"]):
        train_dataloader = load(
            os.path.join(config_files["path"]["processed_path"], "train_dataloader.pkl")
        )
        test_dataloader = load(
            os.path.join(config_files["path"]["processed_path"], "test_dataloader.pkl")
        )
        val_dataloader = load(
            os.path.join(config_files["path"]["processed_path"], "val_dataloader.pkl")
        )

        return {
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
            "val_dataloader": val_dataloader,
        }

    else:
        raise FileNotFoundError("Could not find the dataloader".capitalize())


def init_loss():
    adversarial_loss = ClassificationLoss()
    criterion_loss = ValidityLoss()

    return {"adversarial_loss": adversarial_loss, "criterion_loss": criterion_loss}


def helper(**kwargs):
    lr = kwargs["lr"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]

    netG = Generator(latent_space=100, out_channels=1)
    netD = Discriminator(in_channels=1, out_channels=512)

    if adam:
        optimizerG = optim.Adam(params=netG.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerD = optim.Adam(params=netD.parameters(), lr=lr, betas=(0.5, 0.999))

    if SGD:
        optimizerG = optim.SGD(params=netG.parameters(), lr=lr, momentum=0.85)
        optimizerD = optim.SGD(params=netD.parameters(), lr=lr, momentum=0.85)

    try:
        loss = init_loss()
        dataloader = load_dataloader()

    except AttributeError as e:
        print("The exception raised {}".format(e))

    finally:
        print("All are extracted...".capitalize())

    return {
        "train_dataloader": dataloader["train_dataloader"],
        "test_dataloader": dataloader["test_dataloader"],
        "val_dataloader": dataloader["val_dataloader"],
        "netG": netG,
        "netD": netD,
        "optimizerG": optimizerG,
        "optimizerD": optimizerD,
        "adversarial_loss": loss["adversarial_loss"],
        "criterion_loss": loss["criterion_loss"],
    }


if __name__ == "__main__":
    init = helper(
        lr=0.0002,
        adam=True,
        SGD=False,
    )

    print(init["train_dataloader"])
    print(init["test_dataloader"])
    print(init["val_dataloader"])
    print(init["netG"])
    print(init["netD"])
    print(init["optimizerG"])
    print(init["optimizerD"])
    print(init["adversarial_loss"])
    print(init["criterion_loss"])
    print(init["adversarial_loss"])
