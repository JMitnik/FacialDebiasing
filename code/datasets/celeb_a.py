import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Callable

from torch import float64

from .data_utils import default_transform, DataLabel, DatasetOutput

# Default transform
default_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

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
        # Read img into PIL.Image format
        img: Image = Image.open(os.path.join(
            self.path_to_images,
            self.df_images.iloc[idx].image_id)
        )

        # Transform to tensor
        img = self.transform(img)
        label: int = DataLabel.POSITIVE.value
        sub_images = torch.tensor(0)

        return DatasetOutput(img, label, idx, sub_images=sub_images)

    def sample(self, amount: int):
        max_idx: int = len(self)
        idxs: np.array = np.random.choice(np.linspace(0, max_idx - 1), amount)

        return [self.__getitem__(idx) for idx in idxs]

    def __len__(self):
        return len(self.df_images)
