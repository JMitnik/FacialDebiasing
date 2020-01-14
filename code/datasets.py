import torch
from torch.utils.data import Dataset as TorchDataset, ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler, BatchSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from setup import config
import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable, Optional, Tuple
from enum import Enum

from torch import float64

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

        img = self.transform(img)
        label: int = DataLabel.POSITIVE.value

        return img, label, idx

    def sample(self, amount: int):
        max_idx: int = len(self)
        idxs: np.array = np.random.choice(np.linspace(0, max_idx - 1), amount)

        return [self.__getitem__(idx) for idx in idxs]

    def __len__(self):
        return len(self.df_images)

class ImagenetDataset(ImageFolder):
    def __init__(self, path_to_images: str, transform: Callable = default_transform):
        super().__init__(path_to_images, transform)

    def __getitem__(self, idx: int):
        # Override label with negative
        img, _ = super().__getitem__(idx)
        return (img, DataLabel.NEGATIVE.value, idx)

    def sample(self, amount: int):
        max_idx: int = len(self)
        idxs: np.array = np.random.choice(np.linspace(0, max_idx - 1), amount)

        return [self.__getitem__(idx) for idx in idxs]

def split_dataset(dataset, train_size: float, max_images: Optional[int] = None):
    # Shuffle indices of the dataset
    idxs: np.array = np.arange(len(dataset))
    np.random.seed(config.random_seed)
    np.random.shuffle(idxs)

    # Sample sub-selection
    sampled_idxs: np.array = idxs if not max_images else idxs[:min(max_images, len(idxs))]

    # Split dataset
    split: int = int(np.floor(train_size * len(sampled_idxs)))
    train_idxs = sampled_idxs[:split]
    valid_idxs = sampled_idxs[split:]

    # Subsample dataset with given validation indices
    train_data = Subset(dataset, train_idxs)
    valid_data = Subset(dataset, valid_idxs)

    return train_data, valid_data

def concat_datasets(dataset_a, dataset_b, proportion_a):
    proportion_b: float = 1 - proportion_a

    # Calculate amount of dataset
    nr_dataset_a: int = int(np.floor(proportion_a * len(dataset_a)))
    nr_dataset_b: int = int(np.floor(proportion_b * len(dataset_b)))

    # Subsample the datasets
    sampled_dataset_a = Subset(dataset_a, np.arange(nr_dataset_a))
    sampled_dataset_b = Subset(dataset_b, np.arange(nr_dataset_b))

    return ConcatDataset([sampled_dataset_a, sampled_dataset_b])

def train_and_valid_loaders(
    batch_size: int,
    shuffle: bool = True,
    train_size: float = 0.8,
    proportion_faces: float = 0.5,
    max_images: Optional[int] = None,
    reduce_bias: bool = False
):
    # Create the datasets
    imagenet_dataset: Dataset = ImagenetDataset(config.path_to_imagenet_images)
    celeb_dataset: Dataset = CelebDataset(config.path_to_celeba_images, config.path_to_celeba_bbox_file)

    # Split both datasets into training and validation
    celeb_train, celeb_valid = split_dataset(celeb_dataset, train_size, max_images)
    imagenet_train, imagenet_valid = split_dataset(imagenet_dataset, train_size, max_images)

    # Concat the sources of data by their proportions
    dataset_train: Dataset = concat_datasets(celeb_train, imagenet_train, proportion_faces)
    dataset_valid: Dataset = concat_datasets(celeb_valid, imagenet_valid, proportion_faces)

    # Nonfaces loaders
    train_nonfaces_loader: DataLoader = DataLoader(imagenet_train, batch_size=batch_size, shuffle=shuffle, num_workers=5)
    valid_nonfaces_loader: DataLoader = DataLoader(imagenet_valid, batch_size=batch_size, shuffle=False, num_workers=5)

    # Define the sampler
    weights_sampler_train = WeightedRandomSampler(torch.ones(len(celeb_train), dtype=float64), batch_size, replacement=False)
    batch_sampler = BatchSampler(weights_sampler_train, batch_size, drop_last=True)

    # Define the face loaders
    train_faces_loader: DataLoader = DataLoader(celeb_train, batch_sampler=batch_sampler)
    valid_faces_loader: DataLoader = DataLoader(celeb_valid, batch_size=batch_size, shuffle=shuffle, num_workers=5)

    train_loaders: Tuple[DataLoader, DataLoader] = (train_faces_loader, train_nonfaces_loader)
    valid_loaders: Tuple[DataLoader, DataLoader] = (valid_faces_loader, valid_nonfaces_loader)

    return train_loaders, valid_loaders

def sample_dataset(dataset: Dataset, nr_samples: int):
    max_nr_items: int = min(nr_samples, len(dataset))
    idxs = np.random.permutation(np.arange(len(dataset)))[:max_nr_items]

    return torch.stack([dataset[idx][0] for idx in idxs])

def sample_idxs_from_loader(idxs, data_loader, label):
    if label == 1:
        dataset = data_loader.dataset.datasets[0].dataset.dataset
    else:
        dataset = data_loader.dataset.datasets[1].dataset.dataset

    return torch.stack([dataset[idx.item()][0] for idx in idxs])

