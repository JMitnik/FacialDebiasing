import torch
from torch.utils.data import Dataset as TorchDataset, ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler, BatchSampler, SequentialSampler
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import RandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import numpy as np
import pandas as pd
import uuid
from PIL import Image
from typing import Callable, Optional, List, NamedTuple
from enum import Enum

from torch import float64, floor

class GenderEnum(Enum):
    MEN = "men"
    WOMEN = "woman"

class DataLabel(Enum):
    POSITIVE = 1
    NEGATIVE = 0

class CountryEnum(Enum):
    FINLAND = "Finland"
    ICELAND = "Iceland"
    SWEDEN = "Sweden"
    RWANDA = "Rwanda"
    SENEGAL = "Senegal"
    SOUTHAFRICA = "South Africa"

class SkinColorEnum(Enum):
    DARKER = "darker"
    LIGHTER = "lighter"

class DataLoaderTuple(NamedTuple):
    faces: DataLoader
    nonfaces: DataLoader

class DatasetOutput(NamedTuple):
    image: torch.FloatTensor
    label: int
    idx: int
    sub_images: Optional[torch.Tensor] = None

# Default transform
default_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def slide_windows_over_img(
    img: torch.Tensor,
    min_win_size: int,
    max_win_size: int,
    nr_windows: int,
    stride: float
):
    # Various sizes of the windows
    window_sizes: np.array = np.linspace(min_win_size, max_win_size, nr_windows, dtype=int)

    result = []

    # For each window-size, get all the sub
    for win_size in window_sizes:
        sub_images = slide_single_window_over_img(img, win_size, stride)
        result.append(sub_images)

    return torch.cat(result, dim=0)

def apply_window_resize(img: torch.Tensor, win_size: int):
    pil_transform = transforms.ToPILImage()

    img_transforms = transforms.Compose([
        pil_transform,
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    return img_transforms(img)

def slide_single_window_over_img(
    img: torch.Tensor,
    win_size: int,
    stride_pct: float = 0.2
):
    img = torch.squeeze(img)

    # Image dimensions
    img_width: int = img.shape[1]
    img_height: int = img.shape[2]

    sub_images = []

    step_size = int(np.floor(win_size * stride_pct))

    for y in range(0, img_height, step_size):
        for x in range(0, img_width, step_size):
            sub_image = img[:, x: x + win_size, y: y + win_size]

            resized_sub_image = apply_window_resize(sub_image, win_size)

            sub_images.append(resized_sub_image)

    return torch.stack(sub_images)

def visualize_tensor(img_tensor: torch.Tensor):
    pil_transformer = transforms.ToPILImage()
    pil_transformer(img_tensor).show()

def save_images(torch_tensors: torch.Tensor, path_to_folder: str):
    rand_filenames = str(uuid.uuid4())[:8]
    pil_transformer = transforms.ToPILImage()
    image_folder = f"results/{path_to_folder}/debug/images/{rand_filenames}/"
    os.makedirs(image_folder, exist_ok=True)

    for i, img in enumerate(torch_tensors):
        pil_img = pil_transformer(img)
        pil_img.save(f"{image_folder}/{rand_filenames}_{i}.jpg")

    return torch_tensors
