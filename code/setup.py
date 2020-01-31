from typing import NamedTuple
from types import SimpleNamespace
import torch
import datetime
import os
import argparse
from logger import logger
from typing import Optional
from dataclasses import dataclass, field

# Default device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int,
                    help='size of batch')
parser.add_argument('--epochs', type=int,
                    help='max number of epochs')
parser.add_argument('--z_dim', type=int,
                    help='dimensionality of latent space')
parser.add_argument('--alpha', type=float,
                    help='importance of debiasing')
parser.add_argument('--num_bins', type=int,
                    help='importance of debiasing')
parser.add_argument('--max_images', type=int,
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
parser.add_argument("--folder_name", type=str,
                        help='folder_name_to_save in')
parser.add_argument("--eval_name", type=str,
                        help='eval name')
parser.add_argument('--stride', type=float,
                    help='importance of debiasing')
parser.add_argument('--eval_dataset', type=str,
                    help='Name of eval dataset [ppb/h5_imagenet/h5]')
parser.add_argument('--save_sub_images', type=bool,
                    help='Save images')
parser.add_argument('--model_name', type=str,
                    help='name of the model to evaluate')
parser.add_argument('--hist_size', type=bool,
                    help='Number of histogram')
parser.add_argument('--run_mode', type=str,
                    help='Type of main.py run')
parser.add_argument('-f', type=str,
                    help='Path to kernel json')


class EmptyObject():
    def __getattribute__(self, idx):
        return None

ARGS, unknown = parser.parse_known_args()
if len(unknown) > 0:
    logger.warning(f'There are some unknown args: {unknown}')

num_workers = 5 if ARGS.num_workers is None else ARGS.num_workers

def create_folder_name(foldername):
    if foldername == "":
        return foldername

    suffix = ''
    count = 0
    while True:
        if not os.path.isdir(f"results/{foldername}{suffix}"):
            foldername = f'{foldername}{suffix}'
            return foldername
        else:
            count += 1
            suffix = f'_{count}'

def create_run_folder(folder_name):
    if len(folder_name) > 0:
        return create_folder_name(folder_name)

    return create_folder_name(str(datetime.datetime.now().strftime("%d_%m_%Y---%H_%M_%S")))

@dataclass
class Config:
    # Running main for train, eval or both
    run_mode: str = 'both' if ARGS.run_mode is None else ARGS.run_mode
    # Folder name of the run
    run_folder: str = '' if ARGS.folder_name is None else ARGS.folder_name
    # Path to CelebA images
    path_to_celeba_images: str = 'data/celeba/images'
    # Path to CelebA bounding-boxes
    path_to_celeba_bbox_file: str = 'data/celeba/list_bbox_celeba.txt'
    # Path to ImageNet images
    path_to_imagenet_images: str = 'data/imagenet'
    # Path to evaluation images (Faces)
    path_to_eval_face_images: str = 'data/ppb/PPB-2017/imgs'
    # Path to evaluation metadata
    path_to_eval_metadata: str = 'data/ppb/PPB-2017/PPB-2017-metadata.csv'
    # Path to evaluation images (Nonfaces such as Imagenet)
    path_to_eval_nonface_images: str = 'data/imagenet'
    # Path to stored model
    path_to_model: Optional[str] = ARGS.path_to_model
    # Path to h5
    path_to_h5_train: str = 'data/h5_train/train_face.h5'
    # Type of debiasing used
    debias_type: str = ARGS.debias_type or 'none'
    # name of the model to evaluate
    model_name: str = ARGS.model_name or 'model.pt'
    # Random seed for reproducability
    random_seed: int = 0
    # Device to use
    device: torch.device = DEVICE
    # eval file name
    eval_name: str = ARGS.eval_name or "evaluation_results.txt"
    # Batch size
    batch_size: int = ARGS.batch_size or 256
    # Number of bins
    num_bins: int = ARGS.num_bins or 10
    # Epochs
    epochs: int = ARGS.epochs or 50
    # Z dimension
    z_dim: int = ARGS.z_dim or 200
    # Alpha value
    alpha: float = ARGS.alpha or 0.01
    # stride used for evaluation windows
    stride: float = ARGS.stride or 0.2
    # Dataset size
    max_images: int = ARGS.max_images or -1
    # Eval frequence
    eval_freq: int = ARGS.eval_freq or 5
    # Number workers
    num_workers: int = 5 if ARGS.num_workers is None else ARGS.num_workers
    # Image size
    image_size: int = 64
    # Number windows evaluation
    sub_images_nr_windows: int = 15
    # Evaluation window minimum
    eval_min_size: int = 30
    # Evaluation window maximum
    eval_max_size: int = 64
    # Uses h5 instead of the imagenet files
    use_h5: bool = True if ARGS.use_h5 is None else ARGS.use_h5
    # Debug mode prints several statistics
    debug_mode: bool = False if ARGS.debug_mode is None else ARGS.debug_mode
    # Dataset for evaluation
    eval_dataset: str = ARGS.eval_dataset or 'ppb'
    # Images to save
    save_sub_images: bool = False if ARGS.save_sub_images is None else ARGS.save_sub_images
    # Hist size
    hist_size: int = 1000 if ARGS.hist_size is None else ARGS.hist_size
    # Batch size for how many sub images to batch
    sub_images_batch_size: int = 10
    # Minimum size for sub images
    sub_images_min_size: int = 30
    # Maximum size for sub images
    sub_images_max_size: int = 64
    # Stride of sub images
    sub_images_stride: float = 0.2

    def __post_init__(self, printing=False):
        self.run_folder = create_run_folder(self.run_folder)
        if printing:
            logger.save(f"Saving new run files to {self.run_folder}")


def init_trainining_results(config: Config):
    # Write run-folder name
    if not os.path.exists("results"):
        os.makedirs("results")

    config.__post_init__(printing=True)
    os.makedirs("results/"+ config.run_folder + '/best_and_worst')
    os.makedirs("results/"+ config.run_folder + '/bias_probs')
    os.makedirs("results/"+ config.run_folder + '/reconstructions')

    with open(f"results/{config.run_folder}/flags.txt", "w") as write_file:
      write_file.write(f"z_dim = {config.z_dim}\n")
      write_file.write(f"alpha = {config.alpha}\n")
      write_file.write(f"epochs = {config.epochs}\n")
      write_file.write(f"batch size = {config.batch_size}\n")
      write_file.write(f"eval frequency = {config.eval_freq}\n")
      write_file.write(f"max images = {config.max_images}\n")
      write_file.write(f"debiasing type = {config.debias_type}\n")


    if config.debug_mode:
        os.makedirs(f"results/{config.run_folder}/debug")

    with open(f"results/{config.run_folder}/training_results.csv", "a+") as write_file:
        write_file.write("epoch,train_loss,valid_loss,train_acc,valid_acc\n")

    with open(f"results/{config.run_folder}/flags.txt", "w") as wf:
        wf.write(f"debias_type: {config.debias_type}\n")
        wf.write(f"alpha: {config.alpha}\n")
        wf.write(f"z_dim: {config.z_dim}\n")
        wf.write(f"batch_size: {config.batch_size}\n")
        wf.write(f"max_images: {config.max_images}\n")
        wf.write(f"use_h5: {config.use_h5}\n")

default_config = Config()
