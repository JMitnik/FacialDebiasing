import torch
from torch.utils.data import Dataset as TorchDataset, ConcatDataset, DataLoader, Dataset, Sampler, WeightedRandomSampler, BatchSampler, SequentialSampler
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import RandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from setup import config
import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable, Optional, List, NamedTuple, Union
from enum import Enum
from torch import float64

from .generic import CountryEnum, SkinColorEnum, default_transform, DataLabel, GenderEnum

class PBBDataset(TorchDataset):
    def __init__(
        self,
        path_to_images: str,
        path_to_metadata: str,
        filter_excl_gender: List[Union[GenderEnum, str]] = [],
        filter_excl_country: List[Union[CountryEnum, str]] = [],
        filter_excl_skin_color: List[Union[SkinColorEnum, str]] = [],
        transform: Callable = default_transform
    ):
        self.path_to_images: str = path_to_images
        self.path_to_metadata: str = path_to_metadata
        self.filter_excl_gender: List[Union[GenderEnum, str]] = filter_excl_gender
        self.filter_excl_country: List[Union[CountryEnum, str]] = filter_excl_country
        self.filter_excl_skin_color: List[Union[SkinColorEnum, str]] = filter_excl_skin_color
        self.transform = transform

        self.df_metadata = self._apply_filters_to_metadata(pd.read_csv(self.path_to_metadata))


    def _apply_filters_to_metadata(self, df: pd.DataFrame):
        result = df

        if len(self.filter_excl_country):
            result = result.query('country not in @self.filter_excl_country')

        if len(self.filter_excl_gender):
            result = result.query('gender not in @self.filter_excl_gender')

        if len(self.filter_excl_skin_color):
            result = result.query('bi.fitz not in @self.filter_excl_gender')

        return result

    def __getitem__(self, idx: int):
        img: Image = Image.open(os.path.join(self.path_to_images,
                                self.df_metadata.iloc[idx].filename))

        img = self.transform(img)

        label = DataLabel.POSITIVE.value

        return img, label, idx

    def __len__(self):
        return len(self.df_metadata)
