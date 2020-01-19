from typing import NamedTuple
import torch
import datetime
import os
import argparse
from typing import Optional

# Default device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define folder name
FOLDER_NAME = "{}".format(datetime.datetime.now().strftime("%d_%m_%Y---%H_%M_%S"))

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
parser.add_argument('--debias_type', type=str,
                    help='type of debiasing used')
parser.add_argument("--path_to_model", type=str,
                        help='Path to stored model')
parser.add_argument("--num_workers", type=int,
                        help='Path to stored model')
ARGS = parser.parse_args()

num_workers = 5 if ARGS.num_workers is None else ARGS.num_workers

class Config(NamedTuple):
    # Path to CelebA images
    path_to_celeba_images: str = 'data/celeba/images'
    # Path to CelebA bounding-boxes
    path_to_celeba_bbox_file: str = 'data/celeba/list_bbox_celeba.txt'
    # Path to ImageNet images
    path_to_imagenet_images: str = 'data/imagenet'
    # Path to evaluation images (Faces)
    path_to_eval_face_images: str = 'data/ppb/imgs'
    # Path to evaluation metadata
    path_to_eval_metadata: str = 'data/ppb/PPB-2017-metadata.csv'
    # Path to evaluation images (Nonfaces such as Imagenet)
    path_to_eval_nonface_images: str = 'data/eval_imagenet'
    # Path to stored model
    path_to_model: Optional[str] = ARGS.path_to_model or None
    # Type of debiasing used
    debias_type: str = ARGS.debias_type or 'none'
    # Random seed for reproducability
    random_seed: int = 0
    # Device to use
    device: torch.device = DEVICE
    # Folder name of the run
    run_folder: str = FOLDER_NAME
    # Batch size
    batch_size: int = ARGS.batch_size or 256
    # Epochs
    epochs: int = ARGS.epochs or 10
    # Z dimension
    zdim: int = ARGS.zdim or 200
    # Alpha value
    alpha: float = ARGS.alpha or 0.0
    # Dataset size
    dataset_size: int = ARGS.dataset_size or -1
    # Eval frequence
    eval_freq: int = ARGS.eval_freq or 5
    # Number workers
    num_workers: int = 5 if ARGS.num_workers is None else ARGS.num_workers
    # Image size
    image_size: int = 64
    # Number windows evaluation
    eval_nr_windows: int = 15
    # Evaluation window minimum
    eval_min_size: int = 30
    # Evaluation window maximum
    eval_max_size: int = 64

config = Config()

# Write run-folder name
if not os.path.exists("results"):
    os.makedirs("results")

os.makedirs("results/"+ FOLDER_NAME + '/best_and_worst')
os.makedirs("results/"+ FOLDER_NAME + '/bias_probs')
os.makedirs("results/"+ FOLDER_NAME + '/reconstructions')

with open("results/" + FOLDER_NAME + "/flags.txt", "w") as write_file:
    write_file.write(f"zdim = {config.zdim}\n")
    write_file.write(f"alpha = {config.alpha}\n")
    write_file.write(f"epochs = {config.epochs}\n")
    write_file.write(f"batch size = {config.batch_size}\n")
    write_file.write(f"eval frequency = {config.eval_freq}\n")
    write_file.write(f"dataset size = {config.dataset_size}\n")
    write_file.write(f"debiasing type = {config.debias_type}\n")

with open("results/" + FOLDER_NAME + "/training_results.csv", "w") as write_file:
    write_file.write("epoch,train_loss,valid_loss,train_acc,valid_acc\n")


print(f"Config => {config}")
