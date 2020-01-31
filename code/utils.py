from typing import List
from logger import logger
from datasets.data_utils import DatasetOutput
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import gc
from collections import Counter
from PIL import Image

from dataset import sample_dataset, sample_idxs_from_loader, sample_idxs_from_loaders

def calculate_accuracy(labels, pred):
    """Calculates accuracy given labels and predictions."""
    return float(((pred > 0) == (labels > 0)).sum()) / labels.size()[0]

def get_best_and_worst_predictions(labels, pred, device):
    """Returns indices of the best and worst predicted faces."""
    n_rows = 4
    n_samples = n_rows**2

    logger.info(f"Face percentage: {float(labels.sum().item())/len(labels)}")
    indices = torch.tensor([i for i in range(len(labels))]).long().to(device)

    faceslice = labels == 1
    faces,       other       = pred[faceslice],    pred[~faceslice]
    faces_index, other_index = indices[faceslice], indices[~faceslice]

    worst_faces = faces_index[faces.argsort()[:n_samples]]
    best_faces = faces_index[faces.argsort(descending=True)[:n_samples]]

    worst_other = other_index[other.argsort(descending=True)[:n_samples]]
    best_other = other_index[other.argsort()[:n_samples]]

    return best_faces, worst_faces, best_other, worst_other

def remove_frame(plt):
    """Removes frames from a pyplot plot. """ 
    # TODO: Add annotation
    frame = plt.gca()
    for xlabel_i in frame.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    for xlabel_i in frame.axes.get_yticklabels():
        xlabel_i.set_fontsize(0.0)
        xlabel_i.set_visible(False)
    for tick in frame.axes.get_xticklines():
        tick.set_visible(False)
    for tick in frame.axes.get_yticklines():
        tick.set_visible(False)

def concat_batches(batch_a: DatasetOutput, batch_b: DatasetOutput):
    """Concatenates two batches of data of shape image x label x idx."""
    images: torch.Tensor = torch.cat((batch_a.image, batch_b.image), 0)
    labels: torch.Tensor = torch.cat((batch_a.label, batch_b.label), 0)
    idxs: torch.Tensor = torch.cat((batch_a.idx, batch_b.idx), 0)

    return images, labels, idxs


def read_image(path_to_image):
    """Reads an image into memory and transform to a tensor.""" 
    img: Image = Image.open(path_to_image)

    transforms = default_transforms()
    img_tensor: torch.Tensor = transforms(img)

    return img_tensor

def read_flags(path_to_model):
    """"""
    path_to_flags = f"results/{path_to_model}/flags.txt"

    with open(path_to_flags, 'r') as f:
        data = f.readlines()

def find_face_in_subimages(model, sub_images: torch.Tensor, device: str):
    """Finds a face in a tensor of subimages using a models' evaluation method."""
    model.eval()

    for images in sub_images:
        if len(images.shape) == 5:
            images = images.squeeze(dim=0)

        # If one image
        if len(images.shape) == 3:
            images = images.view(1, 3, 64, 64)
        images = images.to(device)
        pred = model.forward_eval(images)

        # If face
        if (pred > 0).any():
            return True

    return False


def default_transforms():
    """Transforms a transform object to a 64 by 64 tensor."""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

def visualize_tensor(img_tensor: torch.Tensor):
    """Visualizes a image tensor."""
    pil_transformer = transforms.ToPILImage()
    pil_transformer(img_tensor).show()