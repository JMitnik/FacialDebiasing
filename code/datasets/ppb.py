import torch
from torch.utils.data import Dataset as TorchDataset
from .generic import GenericImageDataset
import numpy as np
from logger import logger
import pandas as pd
import os
from PIL import Image
from typing import Callable, Optional, List, Union
from .data_utils import CountryEnum, DatasetOutput, SkinColorEnum, default_transform, DataLabel, GenderEnum, slide_windows_over_img

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

    def init_store(self, path_to_metadata):
        if not os.path.exists(path_to_metadata):
            logger.error(f"Path to metadata (and probably PPB) does not exist at {path_to_metadata}!")
            raise Exception

        try:
            store = self._apply_filters_to_metadata(pd.read_csv(path_to_metadata, delim_whitespace=True))
            return store
        except:
            logger.error(
                f"Unable to read the metadata file located at {path_to_metadata}"
            )


    def _apply_filters_to_metadata(self, df: pd.DataFrame):
        """Allows filters to filter out countries, skin-colors and genders."""
        result = df

        if len(self.filter_excl_country):
            result = result.query('country not in @self.filter_excl_country')

        if len(self.filter_excl_gender):
            result = result.query('gender not in @self.filter_excl_gender')

        if len(self.filter_excl_skin_color):
            try:
                result = result.query('bi_fitz not in @self.filter_excl_skin_color')
            except:
                logger.error("bi_fitz can't be found in the metadata datadframe",
                             next_step="The skin color wont be applied",
                             tip="Rename the bi.fitz column to be bi_fitz in the metadata csv")

        return result

    def read_image(self, idx: int):
        return Image.open(os.path.join(
            self.path_to_images,
            self.store.iloc[idx].filename
        ))

    def __len__(self):
        return len(self.store)
