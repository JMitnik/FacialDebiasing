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
parser.add_argument("--debug_mode", type=bool,
                        help='Debug mode')
parser.add_argument("--use_h5", type=bool,
                        help='Use h5')
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
    # Path to h5
    path_to_h5_train: str = 'data/h5_train/train_face.h5'
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
    alpha: float = ARGS.alpha or 0.01
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
    # Uses h5 instead of the imagenet files
    use_h5: bool = False if ARGS.use_h5 is None else ARGS.use_h5
    # Debug mode prints several statistics
    debug_mode: bool = False if ARGS.debug_mode is None else ARGS.debug_mode

config = Config()

def init_trainining_results():
    # Write run-folder name
    if not os.path.exists("results"):
        os.makedirs("results")

    os.makedirs("results/"+ config.run_folder + '/best_and_worst')
    os.makedirs("results/"+ config.run_folder + '/bias_probs')
    os.makedirs("results/"+ config.run_folder + '/reconstructions')

    if config.debug_mode:
        os.makedirs(f"results/{config.run_folder}/debug")

    with open("results/" + config.run_folder + "/training_results.csv", "a+") as write_file:
        write_file.write("epoch,train_loss,valid_loss,train_acc,valid_acc\n")

print(f"Config => {config}")
