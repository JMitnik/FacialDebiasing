from typing import NamedTuple
import torch
import datetime
import os
import argparse

# Default device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run folder name
FOLDER_NAME = "images_{}".format(datetime.datetime.now())
os.makedirs("images/"+ FOLDER_NAME + '/best_and_worst')
os.makedirs("images/"+ FOLDER_NAME + '/base_vs_our')
os.makedirs("images/"+ FOLDER_NAME + '/reconstructions')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int,
                    help='size of batch')
parser.add_argument('--epochs', type=int,
                    help='max number of epochs')
parser.add_argument('--zdim', type=int,
                    help='dimensionality of latent space')
parser.add_argument('--alpha', type=float,
                    help='importance of debiasing')
parser.add_argument('--dataset_size', type=int,
                    help='total size of database')
parser.add_argument('--eval_freq', type=int,
                    help='total size of database')
ARGS = parser.parse_args()

class Config(NamedTuple):
    # Path to CelebA images
    path_to_celeba_images: str = 'data/celeba/images'
    # Path to CelebA bounding-boxes
    path_to_celeba_bbox_file: str = 'data/celeba/list_bbox_celeba.txt'
    # Path to ImageNet images
    path_to_imagenet_images: str = 'data/imagenet'
    # Random seed for reproducability
    random_seed: int = 0
    # Device to use
    device: torch.device = DEVICE
    # Folder name of the run
    run_folder: str = FOLDER_NAME
    # Batch size
    batch_size: int = ARGS.batch_size or 128
    # Epochs
    epochs: int = ARGS.epochs or 10
    # Z dimension
    zdim: int = ARGS.zdim or 200
    # Alpha value
    alpha: float = ARGS.alpha or 0.0
    # Dataset size
    dataset_size: int = ARGS.dataset_size or 10000
    # Eval frequence
    eval_freq: int = ARGS.eval_freq or 5

config = Config()
