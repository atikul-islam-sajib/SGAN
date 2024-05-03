import os
import cv2
import sys
import argparse
import zipfile
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

sys.path.append("src/")

from utils import config, dump, load


class Loader:
    def __init__(self, image_path=None, image_size=64, batch_size=4, split_size=0.50):
        self.image_path = image_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

        self.config = config()

        self.images = []
        self.labels = []

    def split_images(self, **kwargs):
        X_train, X_test, y_train, y_test = train_test_split(
            kwargs["X"], kwargs["y"], test_size=self.split_size, random_state=42
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.5,
                    ],
                    std=[
                        0.5,
                    ],
                ),
            ]
        )

    def unzip_folder(self):
        if os.path.exists(self.config["path"]["raw_path"]):

            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(self.config["path"]["raw_path"])

        else:
            raise FileNotFoundError(
                "Raw path is not found for the directory".capitalize()
            )

    def extract_feature(self):

        self.directory = os.path.join(self.config["path"]["raw_path"], "brain")
        self.categories = [file for file in os.listdir(self.directory)]

        for category in self.categories:
            full_path = os.path.join(self.directory, category)

            for image in os.listdir(full_path):
                full_image_path = os.path.join(full_path, image)

                image_object = Image.fromarray(cv2.imread(full_image_path))
                self.images.append(self.transforms()(image_object))
                self.labels.append(self.categories.index(category))

        dataset = self.split_images(X=self.images, y=self.labels)

        X_limit = round(len(dataset["X_train"]) * 1.0)
        y_limit = round(len(dataset["y_train"]) * 1.0)

        sub_dataset = self.split_images(
            X=dataset["X_train"][0:X_limit], y=dataset["y_train"][0:y_limit]
        )

        return {
            "X_train": sub_dataset["X_train"],
            "y_train": sub_dataset["y_train"],
            "X_test": sub_dataset["X_test"],
            "y_test": sub_dataset["y_test"],
            "val_train": dataset["X_test"],
            "val_test": dataset["y_test"],
        }

    def create_dataloader(self):
        dataset = self.extract_feature()

        self.train_dataloader = DataLoader(
            dataset=list(zip(dataset["X_train"], dataset["y_train"])),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.test_dataloader = DataLoader(
            dataset=list(zip(dataset["X_test"], dataset["y_test"])),
            batch_size=self.batch_size * 4,
            shuffle=True,
        )

        self.val_dataloader = DataLoader(
            dataset=list(zip(dataset["val_train"], dataset["val_test"])),
            batch_size=self.batch_size,
            shuffle=True,
        )
        if os.path.exists(self.config["path"]["processed_path"]):

            dump(
                value=self.train_dataloader,
                filename=os.path.join(
                    self.config["path"]["processed_path"], "train_dataloader.pkl"
                ),
            )

            dump(
                value=self.test_dataloader,
                filename=os.path.join(
                    self.config["path"]["processed_path"], "test_dataloader.pkl"
                ),
            )

            dump(
                value=self.val_dataloader,
                filename=os.path.join(
                    self.config["path"]["processed_path"], "val_dataloader.pkl"
                ),
            )

        else:
            raise FileNotFoundError(
                "Processed path is not found for the directory".capitalize()
            )

    @staticmethod
    def dataset_details():
        config_files = config()

        if os.path.exists(config_files["path"]["processed_path"]):

            train_dataloader = load(
                filename=os.path.join(
                    config_files["path"]["processed_path"], "train_dataloader.pkl"
                ),
            )

            test_dataloader = load(
                filename=os.path.join(
                    config_files["path"]["processed_path"], "test_dataloader.pkl"
                ),
            )

            val_dataloader = load(
                filename=os.path.join(
                    config_files["path"]["processed_path"], "val_dataloader.pkl"
                ),
            )

            pd.DataFrame(
                {
                    "total_data(Train)": sum(
                        data.size(0) for data, _ in train_dataloader
                    ),
                    "total_data(Test)": sum(
                        data.size(0) for data, _ in test_dataloader
                    ),
                    "total_data(Val)": sum(data.size(0) for data, _ in val_dataloader),
                    "total_batch(Train)": str(len(train_dataloader)),
                    "total_batch(Test)": str(len(test_dataloader)),
                    "total_data": sum(
                        [
                            sum([data.size(0) for data, _ in dataloader])
                            for dataloader in [
                                train_dataloader,
                                test_dataloader,
                                val_dataloader,
                            ]
                        ]
                    ),
                    "train_shape": str(
                        [data.size() for _, (data, _) in enumerate(train_dataloader)][0]
                    ),
                    "test_shape": str(
                        [data.size() for _, (data, _) in enumerate(test_dataloader)][0]
                    ),
                    "val_shape": str(
                        [data.size() for _, (data, _) in enumerate(val_dataloader)][0]
                    ),
                },
                index=["quantity"],
            ).T.to_csv(
                os.path.join(config_files["path"]["files_path"], "dataset_details.csv")
                if os.path.exists(config_files["path"]["files_path"])
                else os.makedirs(config_files["path"]["files_path"])
            )

        else:
            raise FileNotFoundError(
                "Processed path is not found for the directory".capitalize()
            )

    @staticmethod
    def plot_images():
        config_files = config()

        plt.figure(figsize=(20, 10))

        if os.path.exists(config_files["path"]["processed_path"]):
            test_dataloader = load(
                filename=os.path.join(
                    config_files["path"]["processed_path"], "test_dataloader.pkl"
                )
            )

            images, labels = next(iter(test_dataloader))

            for index, image in enumerate(images):
                image = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                image = (image - image.min()) / (image.max() - image.min())

                plt.subplot(4, 4, index + 1)

                plt.imshow(image, cmap="gray")
                plt.title("Yes" if labels[index] == 0 else "No")
                plt.axis("off")

            plt.tight_layout()

            (
                plt.savefig(
                    os.path.join(config_files["path"]["files_path"], "images.png")
                )
                if os.path.exists(config_files["path"]["files_path"])
                else "Cannot save the images to".capitalize()
            )
            plt.show()

        else:
            raise FileNotFoundError(
                "Processed path is not found for the directory".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataloader function for SGAN".title())

    parser.add_argument(
        "--image_path",
        type=str,
        help="Image path for the Image".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        choices=[64, 128],
        help="Image size for the Image".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        choices=[4, 8, 16, 32],
        help="Batch size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=0.2,
        choices=[0.2, 0.25, 0.30],
        help="Split of the dataset".capitalize(),
    )
    args = parser.parse_args()

    if args.image_path:
        loader = Loader(
            image_path=args.image_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            split_size=args.split_size,
        )

        loader.unzip_folder()
        # loader.create_dataloader()

        # loader.dataset_details()
        # loader.plot_images()

    else:
        raise ValueError("Image path is not found".capitalize())
