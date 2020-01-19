import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np
import pandas as pd
import os
from PIL import Image
from typing import Callable, Optional, List, Union
from .generic import CountryEnum, SkinColorEnum, default_transform, DataLabel, GenderEnum, slide_windows_over_img

class PPBDataset(TorchDataset):
    def __init__(
        self,
        path_to_images: str,
        path_to_metadata: str,
        filter_excl_gender: List[Union[GenderEnum, str]] = [],
        filter_excl_country: List[Union[CountryEnum, str]] = [],
        filter_excl_skin_color: List[Union[SkinColorEnum, str]] = [],
        nr_sub_images: int = -1,
        transform: Callable = default_transform
    ):
        self.path_to_images: str = path_to_images
        self.path_to_metadata: str = path_to_metadata
        self.filter_excl_gender: List[Union[GenderEnum, str]] = filter_excl_gender
        self.filter_excl_country: List[Union[CountryEnum, str]] = filter_excl_country
        self.filter_excl_skin_color: List[Union[SkinColorEnum, str]] = filter_excl_skin_color
        self.transform = transform

        self.nr_sub_images: Optional[int] = None if nr_sub_images < 0 else nr_sub_images

        self.df_metadata = self._apply_filters_to_metadata(pd.read_csv(self.path_to_metadata))


    def _apply_filters_to_metadata(self, df: pd.DataFrame):
        result = df

        if len(self.filter_excl_country):
            result = result.query('country not in @self.filter_excl_country')

        if len(self.filter_excl_gender):
            result = result.query('gender not in @self.filter_excl_gender')

        if len(self.filter_excl_skin_color):
            result = result.query('bi_fitz not in @self.filter_excl_skin_color')

        return result

    def __getitem__(self, idx: int):
        img: Image = Image.open(os.path.join(self.path_to_images,
                                self.df_metadata.iloc[idx].filename))

        img = self.transform(img)

        if self.nr_sub_images:
            imgs = slide_windows_over_img(img, min_win_size=30, max_win_size=64, nr_windows=self.nr_sub_images)
        else:
            imgs = img.unsqueeze(0)

        label = DataLabel.POSITIVE.value

        imgs = imgs.squeeze()
        return (imgs, label, idx)

    def __len__(self):
        return len(self.df_metadata)
