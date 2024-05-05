import sys
import argparse

sys.path.append("src/")

from dataloader import Loader
from trainer import Trainer
from test import TestModel


def cli():
    parser = argparse.ArgumentParser(description="CLI for SGAN".title())

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
        choices=[1, 4, 8, 16, 32],
        help="Batch size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=0.50,
        choices=[0.40, 0.50, 0.60, 0.80],
        help="Split of the dataset".capitalize(),
    )
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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Define the model".capitalize(),
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Help to train the model".capitalize(),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Help to test the model".capitalize(),
    )

    args = parser.parse_args()

    if args.train:
        if args.image_path:
            loader = Loader(
                image_path=args.image_path,
                image_size=args.image_size,
                batch_size=args.batch_size,
                split_size=args.split_size,
            )

            loader.unzip_folder()
            loader.create_dataloader()

            loader.dataset_details()
            loader.plot_images()

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

        else:
            raise ValueError("Define the image path first".capitalize())

    elif args.test:
        test_model = TestModel(model=args.model, device=args.device)
        test_model.test()


if __name__ == "__main__":
    cli()
