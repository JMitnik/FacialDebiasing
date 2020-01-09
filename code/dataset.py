from torch.utils.data import Dataset as TorchDataset, ConcatDataset, DataLoader, random_split, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from setup import config
import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable, Optional
from enum import Enum

# Default transform
default_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

class DataLabel(Enum):
    POSITIVE = 1
    NEGATIVE = 0

class CelebDataset(TorchDataset):
    """Dataset for CelebA"""

    def __init__(self, path_to_images: str, path_to_bbox: str, transform: Callable = default_transform):
        self.df_images: pd.DataFrame = pd.read_table(path_to_bbox, delim_whitespace=True)
        self.path_to_images: str = path_to_images
        self.transform = transform

    def __getitem__(self, idx: int):
        """Retrieves the cropped images and resizes them to dimensions (64, 64)

        Arguments:
            index

        Returns:
            (tensor, int) -- Image and class
        """
        img: Image = Image.open(os.path.join(self.path_to_images,
                                      self.df_images.iloc[idx].image_id))

        img: Image = self.transform(img)
        label: int = DataLabel.POSITIVE.value

        return img, label

    def __len__(self):
        return len(self.df_images)

class ImagenetDataset(ImageFolder):
    def __init__(self, path_to_images: str, transform: Callable = default_transform):
        super().__init__(path_to_images, transform)

    def __getitem__(self, idx: int):
        # Override label with negative
        img, _ = super().__getitem__(idx)
        return (img, DataLabel.NEGATIVE.value)

def split_dataset(dataset, train_size: float):
    nr_train: int = int(np.floor(train_size * len(dataset)))
    nr_valid: int = len(dataset) - nr_train

    return random_split(dataset, [nr_train, nr_valid])

def train_and_valid_loaders(batch_size: int, shuffle: bool = True, train_size: float = 0.8):
    # Create and concatenate multiple sources of data
    imagenet_dataset: Dataset = ImagenetDataset(config.path_to_imagenet_images)
    celeb_dataset: Dataset = CelebDataset(config.path_to_celeba_images, config.path_to_celeba_bbox_file)

    # Split into train and valid datasets
    imagenet_train, imagenet_valid = split_dataset(imagenet_dataset, train_size)
    celeb_train, celeb_valid = split_dataset(celeb_dataset, train_size)

    # Concat the sources of data
    dataset_train: Dataset = ConcatDataset([imagenet_train, celeb_train])
    dataset_valid: Dataset = ConcatDataset([imagenet_valid, celeb_valid])

    # Define the loaders
    train_loader: DataLoader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    valid_loader: DataLoader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader
