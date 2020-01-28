import torch
from torch.utils.data import Dataset as TorchDataset
from .generic import GenericImageDataset
import numpy as np
import pandas as pd
import os
from PIL import Image
from typing import Callable, Optional, List, Union
from .data_utils import CountryEnum, DatasetOutput, SkinColorEnum, default_transform, DataLabel, GenderEnum, slide_windows_over_img
from setup import config

class PPBDataset(GenericImageDataset):
    def __init__(
        self,
        path_to_images: str,
        path_to_metadata: str,
        filter_excl_gender: List[str] = [],
        filter_excl_country: List[str] = [],
        filter_excl_skin_color: List[str] = [],
        nr_windows: int = 10,
        batch_size: int = -1,
        transform: Callable = default_transform,
        get_sub_images: bool = False,
        stride: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.path_to_images: str = path_to_images
        self.path_to_metadata: str = path_to_metadata
        self.filter_excl_gender: List[str] = filter_excl_gender
        self.filter_excl_country: List[str] = filter_excl_country
        self.filter_excl_skin_color: List[str] = filter_excl_skin_color
        self.transform = transform

        self.nr_windows: int = nr_windows
        self.batch_size: Optional[int] = None if batch_size < 0 else batch_size
        self.stride = stride

        self.get_sub_images = get_sub_images
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

    def __getitem__(self, idx: int, stride: float = 0.2):
        img: Image = Image.open(os.path.join(self.path_to_images,
                                self.df_metadata.iloc[idx].filename))

        img = self.transform(img)

        if self.get_sub_images:
            sub_images = slide_windows_over_img(img, min_win_size=config.eval_min_size,
                                          max_win_size=config.eval_max_size,
                                          nr_windows=self.nr_windows,
                                          stride=self.stride)
            sub_images = torch.split(sub_images, self.batch_size)
        else:
            sub_images = torch.tensor(0)

        label = DataLabel.POSITIVE.value

        return DatasetOutput(
            image=img,
            label=label,
            idx=idx,
            sub_images=sub_images
        )

    def __len__(self):
        return len(self.df_metadata)
