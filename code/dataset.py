import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler, BatchSampler, SequentialSampler
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import RandomSampler
from setup import config
import numpy as np
from typing import Optional, List, NamedTuple, Union

from datasets.generic import CountryEnum, DataLoaderTuple, GenderEnum, SkinColorEnum
from datasets.celeb_a import CelebDataset
from datasets.imagenet import ImagenetDataset
from datasets.ppb import PPBDataset

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

def concat_datasets(dataset_a, dataset_b, proportion_a: Optional[float] = None):
    if proportion_a:
        proportion_b = 1 - proportion_a
        # Calculate amount of dataset
        nr_dataset_a: int = int(np.floor(proportion_a * len(dataset_a)))
        nr_dataset_b: int = int(np.floor(proportion_b * len(dataset_b)))

    else:
        nr_dataset_a = len(dataset_a)
        nr_dataset_b = len(dataset_b)

    # Subsample the datasets
    sampled_dataset_a = Subset(dataset_a, np.arange(nr_dataset_a))
    sampled_dataset_b = Subset(dataset_b, np.arange(nr_dataset_b))

    return ConcatDataset([sampled_dataset_a, sampled_dataset_b])

def make_train_and_valid_loaders(
    batch_size: int,
    max_images: int,
    shuffle: bool = True,
    train_size: float = 0.8,
    proportion_faces: float = 0.5,
    enable_debias: bool = True,
    sample_bias_with_replacement: bool = True,
):
    nr_images: Optional[int] = max_images if max_images >= 0 else None

    # Create the datasets
    imagenet_dataset: Dataset = ImagenetDataset(config.path_to_imagenet_images)
    celeb_dataset: Dataset = CelebDataset(config.path_to_celeba_images, config.path_to_celeba_bbox_file)

    # Split both datasets into training and validation
    celeb_train, celeb_valid = split_dataset(celeb_dataset, train_size, nr_images)
    imagenet_train, imagenet_valid = split_dataset(imagenet_dataset, train_size, nr_images)

    # Nonfaces loaders
    train_nonfaces_loader: DataLoader = DataLoader(imagenet_train, batch_size=int(batch_size / 2), shuffle=shuffle, num_workers=config.num_workers)
    valid_nonfaces_loader: DataLoader = DataLoader(imagenet_valid, batch_size=int(batch_size / 2), shuffle=False, num_workers=config.num_workers)

    # Init some weights
    init_weights = torch.rand(len(celeb_train)).tolist()

    # Define samplers: random for non-debias, weighed for debiasing
    random_train_sampler = RandomSampler(celeb_train)
    weights_sampler_train = WeightedRandomSampler(init_weights, len(celeb_train), replacement=sample_bias_with_replacement)

    train_sampler = weights_sampler_train if enable_debias else random_train_sampler

    # Define the face loaders
    train_faces_loader: DataLoader = DataLoader(celeb_train, sampler=train_sampler, batch_size=int(batch_size / 2), num_workers=config.num_workers)
    valid_faces_loader: DataLoader = DataLoader(celeb_valid, batch_size=int(batch_size / 2), shuffle=shuffle, num_workers=config.num_workers)

    train_loaders: DataLoaderTuple = DataLoaderTuple(train_faces_loader, train_nonfaces_loader)
    valid_loaders: DataLoaderTuple = DataLoaderTuple(valid_faces_loader, valid_nonfaces_loader)

    return train_loaders, valid_loaders

def make_eval_loader(
    filter_exclude_gender: List[Union[GenderEnum, str]] = [],
    filter_exclude_country: List[Union[CountryEnum, str]] = [],
    filter_exclude_skin_color: List[Union[SkinColorEnum, str]] = [],
    proportion_faces: float = 0.5,
    batch_size: int = 16
):

    # Define faces dataset
    pbb_dataset = PPBDataset(
        path_to_images=config.path_to_eval_face_images,
        path_to_metadata=config.path_to_eval_metadata,
        filter_excl_country=filter_exclude_country,
        filter_excl_gender=filter_exclude_gender,
        filter_excl_skin_color=filter_exclude_skin_color,
        nr_sub_images=config.batch_size
    )

    # Concat and wrap with loader
    data_loader = DataLoader(pbb_dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)

    return data_loader

def subsample_dataset(dataset: Dataset, nr_subsamples: int):
    idxs = np.arange(nr_subsamples)
    return Subset(dataset, idxs)


def sample_dataset(dataset: Dataset, nr_samples: int):
    max_nr_items: int = min(nr_samples, len(dataset))
    idxs = np.random.permutation(np.arange(len(dataset)))[:max_nr_items]

    return torch.stack([dataset[idx][0] for idx in idxs])

def sample_idxs_from_loaders(idxs, data_loaders, label):
    if label == 1:
        dataset = data_loaders.faces.dataset.dataset
    else:
        dataset = data_loaders.nonfaces.dataset.dataset

    return torch.stack([dataset[idx.item()][0] for idx in idxs])

def sample_idxs_from_loader(idxs, data_loader, label):
    if label == 1:
        dataset = data_loader.dataset.dataset
    else:
        dataset = data_loader.dataset.dataset

    return torch.stack([dataset[idx.item()][0] for idx in idxs])

def make_hist_loader(dataset, batch_size):
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    return DataLoader(dataset, batch_sampler=batch_sampler)
