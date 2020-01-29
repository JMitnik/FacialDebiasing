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
        path_to_metadata: str,
        filter_excl_gender: List[str] = [],
        filter_excl_country: List[str] = [],
        filter_excl_skin_color: List[str] = [],
        **kwargs
    ):
        super().__init__(**kwargs)

        # Path to metadata
        self.path_to_metadata: str = path_to_metadata

        # Filters
        self.filter_excl_gender: List[str] = filter_excl_gender
        self.filter_excl_country: List[str] = filter_excl_country
        self.filter_excl_skin_color: List[str] = filter_excl_skin_color

        # Store is a Dataframe based on the metadata
        self.store: pd.DataFrame = self._apply_filters_to_metadata(pd.read_csv(self.path_to_metadata))

        self.classification_label = 1

    def _apply_filters_to_metadata(self, df: pd.DataFrame):
        result = df

        if len(self.filter_excl_country):
            result = result.query('country not in @self.filter_excl_country')

        if len(self.filter_excl_gender):
            result = result.query('gender not in @self.filter_excl_gender')

        if len(self.filter_excl_skin_color):
            result = result.query('bi_fitz not in @self.filter_excl_skin_color')

        return result

    def read_image(self, idx: int):
        return Image.open(os.path.join(
            self.path_to_images,
            self.store
        ))

    def __len__(self):
        return len(self.store)
