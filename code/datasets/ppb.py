import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np
import pandas as pd
import os
from PIL import Image
from typing import Callable, Optional, List, Union
from .generic import CountryEnum, SkinColorEnum, default_transform, DataLabel, GenderEnum, slide_windows_over_img
from setup import config

class PPBDataset(TorchDataset):
    def __init__(
        self,
        path_to_images: str,
        path_to_metadata: str,
        filter_excl_gender: List[Union[GenderEnum, str]] = [],
        filter_excl_country: List[Union[CountryEnum, str]] = [],
        filter_excl_skin_color: List[Union[SkinColorEnum, str]] = [],
        nr_windows: int = 10,
        batch_size: int = -1,
        transform: Callable = default_transform
    ):
        self.path_to_images: str = path_to_images
        self.path_to_metadata: str = path_to_metadata
        self.filter_excl_gender: List[Union[GenderEnum, str]] = filter_excl_gender
        self.filter_excl_country: List[Union[CountryEnum, str]] = filter_excl_country
        self.filter_excl_skin_color: List[Union[SkinColorEnum, str]] = filter_excl_skin_color
        self.transform = transform

        self.nr_windows: int = nr_windows
        self.batch_size: Optional[int] = None if batch_size < 0 else batch_size

        self.df_metadata: pd.DataFrame = self._apply_filters_to_metadata(pd.read_csv(self.path_to_metadata))


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

        if self.batch_size:
            imgs = slide_windows_over_img(img, min_win_size=config.eval_min_size, max_win_size=config.eval_max_size, nr_windows=self.nr_windows)
            imgs = torch.split(imgs, self.batch_size)
        else:
            imgs = img.unsqueeze(0)

        label = DataLabel.POSITIVE.value

        return (imgs, label, idx)

    def __len__(self):
        return len(self.df_metadata)
