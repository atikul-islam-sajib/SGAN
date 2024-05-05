# SGAN - emi-Supervised Generative Adversarial Network

## Overview

Semi-Supervised Generative Adversarial Network (SGAN) which achieves better classification performance than the standard supervised algorithms using majority unlabelled datasets.

## Model architecture


![AC-GAN - Medical Image Dataset Generator with class labels: Gif file](https://drek4537l1klr.cloudfront.net/langr/v-6/Figures/07_02.png)


## Features

- Utilizes PyTorch for implementing GAN models.
- Provides scripts for easy training and generating synthetic images.
- Command Line Interface for easy interaction.
- Includes a custom data loader for the custom medical image dataset.
- Customizable training parameters for experimenting with GAN.

## Installation

Clone the repository:

```
git clone https://github.com/atikul-islam-sajib/SGAN.git
cd SGAN
```

# Install dependencies

```
pip install -r requirements.txt
```

## Usage

Examples of commands - test and their explanations(Do the training and testing).

```bash
python /path/to/SGAN/src/cli.py --help
```


### Options

| Argument          | Description                                               | Type     | Default |
| ----------------- | --------------------------------------------------------- | -------- | ------- |
| `--image_path`       | Path to the zip file containing the dataset.              | `str`    | None    |
| `--image_size`    | Size of the images to be used.                            | `int`    | 64      |
| `--batch_size`    | Number of images per training batch.                      | `int`    | 64      |
| `--split_size`    | Split size of the Image.           | `floar`   | 0.50    |
| `--epochs`    | The number of epochs for training.                        | `int`    | 200     |
| `--latent`  | Dimensionality of the latent space.                       | `int`    | 50      |
| `--in_channels`   | The number of input channels for the model.               | `int`    | 1       |
| `--lr` | Learning rate for the optimizer.                          | `float`  | 0.0002  |
| `--is_display`       | Whether to display training progress and output images.   | `bool`   | True    |
| `--is_weight_clip`       | Whether to use weight clipping for training smooth.   | `bool`   | True    |
| `--is_weight_init`       | Whether to use weight init for training smooth.   | `bool`   | True    |
| `--l1_loss`       | Whether to use l1 use to prevent overfitting.   | `bool`   | True    |
| `--l2_loss`       | Whether to use l2 use to prevent overfitting.   | `bool`   | True    |
| `--elastic_net`       | Whether to use elastic(l1+l2) use to prevent overfitting.   | `bool`   | True    |
| `--device`        | The device to run the model on ('cuda', 'cpu', or 'mps'). | `str`    | "cuda"  |
| `--train`         | Flag to initiate training mode.                           | `action` | N/A     |
| `--test`          | Flag to initiate testing mode.                            | `action` | N/A     |
| `--model`    | Path to the pre-trained model for testing.                | `str`    | None    |
| `--label`         | Label for the samples to be generated during testing.     | `int`    | 0       |

## Training and Generating Images - CLI

#### Training the GAN Model with CUDA, MPS, CPU

| Action       | Device | Command                                                                                                                                                                                                         |
| ------------ | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Training** | CUDA   | `python cli_script.py --train --image_path path/to/dataset.zip --image_size 64 --batch_size 64 --epochs 200 --latent 100 --in_channels 1 --lr 0.0002 --l1_loss True --is_display True --device cuda` |
|              | MPS    | `python cli_script.py --train --image_path path/to/dataset.zip --image_size 64 --batch_size 64 --epochs 200 --latent 100 --in_channels 1 --lr 0.0002 --l1_loss True --is_display True --device mps`  |
|              | CPU    | ``python cli_script.py --train --image_path path/to/dataset.zip --image_size 64 --batch_size 64 --epochs 200 --latent 100 --in_channels 1 --lr 0.0002 --l1_loss True --is_display True --device cpu`  |
| **Testing**  | CUDA   | `python cli_script.py --test --model_path path/to/model.pth --device cuda`                                                                                          |
|              | MPS    | `python cli_script.py --test --model_path path/to/model.pth --device mps`                                                                                           |
|              | CPU    | `python cli_script.py --test --model_path path/to/model.pth --device cpu`                                                                                           |


## Training and Generating Images - Importing the Modules

### Using CUDA

To leverage CUDA for accelerated computing, follow the instructions below. This setup is ideal for training and testing models with NVIDIA GPUs, ensuring fast processing and efficient handling of large datasets.

**Prerequisites:**

- Ensure CUDA-compatible hardware is available and properly configured.
- Verify that the necessary CUDA drivers and libraries are installed on your system.

**Script Execution:**

1. **Data Preparation:**

   - Initialize and configure the data loader for image preprocessing and dataset creation.

   ```python
   from src.dataloader import Loader

   loader = Loader(image_path="/path/to/dataset.zip", batch_size=64, image_size=64, split_size=0.5)
   loader.unzip_images()
   loader.create_dataloader()
   ```

2. **Model Training:**

   - Set up and initiate the training process using specified parameters.

   ```python
   from src.trainer import Trainer

   trainer = Trainer(
    epochs=200,
    latent=100,
    in_channels=1,
    lr=0.0002,
    l1_loss=True,             # Use 'l2_loss' or 'elastic_net' for other loss functions
    is_display=True,
    device="cuda",            # Use 'mps' or 'cpu'
    is_weight_init=True,      # Use 'False'
    is_weight_clip=True,      # Use 'False'
    lr_scheduler=True,        # Use 'False'
    adam=True                 # Use 'SGD'
   )
   ```

3. **Model Testing:**

   - Execute model testing to evaluate performance and generate synthetic images.

   ```python
   from src.test import Test

   test = TestModel(best_model_path="/checkpoints/best_models/netD_51.pth", device="cuda") # Use 'cuda' or 'mps'
   test.test()  # It will return model accuracy, precision, recall, f1 score along with clf report and confusion metrics
   ```


## Contributing

Contributions to improve the project are welcome. Please follow the standard procedures for contributing to open-source projects.

## License

This project is licensed under [MIT LICENSE](./LICENSE). Please see the LICENSE file for more details.

## Acknowledgements

Thanks to all contributors and users of the SGAN project. Special thanks to those who have provided feedback and suggestions for improvements.

## Contact

For any inquiries or suggestions, feel free to reach out to [atikulislamsajib137@gmail.com].

## Additional Information

- This project is a work in progress and subject to changes.
- Feedback and suggestions are highly appreciated.
- Courtesy: Atikul Islam Sajib
