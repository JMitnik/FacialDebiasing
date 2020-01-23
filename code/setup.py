from typing import NamedTuple
import torch
import datetime
import os
import argparse
from typing import Optional

# Default device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'

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
parser.add_argument('--num_bins', type=float,
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
parser.add_argument("--folder_name", type=str, default="{}".format(datetime.datetime.now().strftime("%d_%m_%Y---%H_%M_%S")),
                        help='folder_name_to_save in')
parser.add_argument("--eval_name", type=str, default="evaluation_results.txt",
                        help='eval name')
parser.add_argument('--stride', type=float,
                    help='importance of debiasing')
parser.add_argument('--eval_dataset', type=str,
                    help='Name of eval dataset [ppb/h5_imagenet/h5]')
parser.add_argument('--save_sub_images', type=bool,
                    help='Save images')
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
    path_to_eval_nonface_images: str = 'data/imagenet'
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
    run_folder: str = ARGS.folder_name
    # eval file name
    eval_name: str = ARGS.eval_name
    # Batch size
    batch_size: int = ARGS.batch_size or 256
    # Number of bins
    num_bins: int = ARGS.num_bins or 10
    # Epochs
    epochs: int = ARGS.epochs or 50
    # Z dimension
    zdim: int = ARGS.zdim or 200
    # Alpha value
    alpha: float = ARGS.alpha or 0.01
    # stride used for evaluation windows
    stride: float = ARGS.stride or 0.2
    # Dataset size
    dataset_size: int = ARGS.dataset_size or -1
    # Eval frequence
    eval_freq: int = ARGS.eval_freq or 5
    # Number workers
    num_workers: int = 5 if ARGS.num_workers is None else ARGS.num_workers
    # Debug mode
    debug_mode: bool = False if ARGS.debug_mode is None else ARGS.debug_mode
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
    # Dataset for evaluation
    eval_dataset: str = ARGS.eval_dataset or 'ppb'
    # Images to save
    save_sub_images: bool = False if ARGS.save_sub_images is None else ARGS.save_sub_images

config = Config()

def init_trainining_results():
    # Write run-folder name
    if not os.path.exists("results"):
        os.makedirs("results")

    os.makedirs("results/"+ config.run_folder + '/best_and_worst')
    os.makedirs("results/"+ config.run_folder + '/bias_probs')
    os.makedirs("results/"+ config.run_folder + '/reconstructions')

    with open(f"results/{config.run_folder}/flags.txt", "w") as write_file:
      write_file.write(f"zdim = {config.zdim}\n")
      write_file.write(f"alpha = {config.alpha}\n")
      write_file.write(f"epochs = {config.epochs}\n")
      write_file.write(f"batch size = {config.batch_size}\n")
      write_file.write(f"eval frequency = {config.eval_freq}\n")
      write_file.write(f"dataset size = {config.dataset_size}\n")
      write_file.write(f"debiasing type = {config.debias_type}\n")


    if config.debug_mode:
        os.makedirs(f"results/{config.run_folder}/debug")

    with open(f"results/{config.run_folder}/training_results.csv", "a+") as write_file:
        write_file.write("epoch,train_loss,valid_loss,train_acc,valid_acc\n")

    with open(f"results/{config.run_folder}/flags.txt", "w") as wf:
        wf.write(f"debias_type: {config.debias_type}\n")
        wf.write(f"alpha: {config.alpha}\n")
        wf.write(f"zdim: {config.zdim}\n")
        wf.write(f"batch_size: {config.batch_size}\n")
        wf.write(f"dataset_size: {config.dataset_size}\n")
        wf.write(f"use_h5: {config.use_h5}\n")

print(f"Config => {config}")
